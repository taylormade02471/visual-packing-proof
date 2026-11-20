from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import CatalogAsset, CatalogVariant
from .monitoring import ModelMonitor
from .packing import PackageItem, select_carton, verify_item
from .search import EmbeddingSearchEngine, EmptyIndexError, MobileNetV3Encoder, SearchResponse


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
PROJECT_DIR = PACKAGE_DIR.parents[1]


@dataclass
class ProofServices:
    engine: EmbeddingSearchEngine
    assets: list[CatalogAsset]
    samples: list[dict[str, str]]
    benchmark: dict[str, object]
    monitor: ModelMonitor

    @property
    def baseline_centroid(self) -> torch.Tensor | None:
        if self.engine.embeddings.numel() == 0:
            return None
        centroid = self.engine.embeddings.mean(dim=0)
        return centroid / torch.linalg.vector_norm(centroid).clamp_min(1e-12)

    def find_variant(self, sku: str) -> tuple[CatalogAsset, CatalogVariant] | None:
        for asset in self.assets:
            for variant in asset.variants:
                if variant.sku == sku:
                    return asset, variant
        return None


def load_services(artifacts_dir: str | Path | None = None) -> ProofServices:
    root = Path(artifacts_dir) if artifacts_dir else PROJECT_DIR / "artifacts"
    records = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    assets = [
        CatalogAsset(
            sha256=record["asset_id"],
            image_bytes=b"",
            image_url=record["image_url"],
            product_ids=tuple(record["product_ids"]),
            product_names=tuple(record["product_names"]),
            categories=tuple(record["categories"]),
            variants=tuple(CatalogVariant.model_validate(value) for value in record["variants"]),
        )
        for record in records
    ]
    engine = EmbeddingSearchEngine(MobileNetV3Encoder())
    engine.assets = assets
    engine.embeddings = torch.load(root / "embeddings.pt", map_location="cpu", weights_only=True)
    samples = [
        {
            "asset_id": record["asset_id"],
            "name": record["product_names"][0],
            "sku": record["variants"][0]["sku"],
            "sample_url": record["sample_url"],
        }
        for record in records
        if record.get("sample_url")
    ]
    benchmark = json.loads((root / "benchmark_report.json").read_text(encoding="utf-8"))
    return ProofServices(engine, assets, samples, benchmark, ModelMonitor())


def _search_payload(response: SearchResponse) -> dict[str, object]:
    return {
        "latency_ms": response.latency_ms,
        "latency_scope": "hot model and in-memory search path",
        "matches": [asdict(match) for match in response.matches],
    }


def create_app(
    services: ProofServices | None = None,
    *,
    max_upload_bytes: int = 8 * 1024 * 1024,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or load_services()
        yield

    app = FastAPI(
        title="Stone Wick Visual Search and Packing AI",
        version="0.1.0",
        description="Reviewer-testable PyTorch visual retrieval and safe packing verification proof.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    async def image_payload(file: UploadFile) -> bytes:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(415, "Upload must be an image.")
        payload = await file.read(max_upload_bytes + 1)
        if len(payload) > max_upload_bytes:
            raise HTTPException(413, f"Image exceeds the {max_upload_bytes}-byte limit.")
        if not payload:
            raise HTTPException(422, "Image is empty.")
        return payload

    def current_services() -> ProofServices:
        return app.state.services

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        proof = current_services()
        return {
            "status": "ok",
            "indexed_assets": len(proof.assets),
            "indexed_variants": sum(len(asset.variants) for asset in proof.assets),
            "encoder": proof.engine.encoder.name,
        }

    @app.get("/api/catalog")
    def catalog() -> list[dict[str, str]]:
        return current_services().samples

    @app.get("/api/evidence")
    def evidence() -> dict[str, object]:
        return current_services().benchmark

    @app.get("/api/metrics")
    def metrics() -> dict[str, object]:
        return current_services().monitor.summary()

    @app.post("/api/search")
    async def visual_search(
        file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=12)
    ) -> dict[str, object]:
        proof = current_services()
        payload = await image_payload(file)
        try:
            response = proof.engine.search(payload, top_k=top_k)
        except EmptyIndexError as error:
            raise HTTPException(503, "Visual search index is unavailable.") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        top_score = response.matches[0].score if response.matches else 0.0
        proof.monitor.record(
            response.latency_ms,
            top_score,
            response.query_embedding,
            proof.baseline_centroid,
        )
        return _search_payload(response)

    @app.post("/api/packing/verify")
    async def packing_verify(
        file: UploadFile = File(...),
        expected_sku: str = Form(...),
        barcode_sku: str | None = Form(None),
    ) -> dict[str, object]:
        proof = current_services()
        match = proof.find_variant(expected_sku)
        if not match:
            raise HTTPException(404, "Expected SKU is not in the proof catalog.")
        asset, variant = match
        payload = await image_payload(file)
        try:
            response = proof.engine.search(payload, top_k=5)
        except EmptyIndexError as error:
            raise HTTPException(503, "Visual search index is unavailable.") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        decision = verify_item(
            response,
            expected_sku,
            asset.sha256,
            barcode_sku=barcode_sku,
            visual_variant_count=len(asset.variants),
        )
        carton = (
            select_carton([PackageItem(expected_sku, variant.dimensions_cm, variant.fragile)])
            if decision.verified
            else None
        )
        top_score = response.matches[0].score if response.matches else decision.confidence
        proof.monitor.record(
            response.latency_ms,
            top_score,
            response.query_embedding if response.query_embedding.numel() else None,
            proof.baseline_centroid,
            decision=decision.decision,
        )
        return {
            "search": _search_payload(response),
            "verification": asdict(decision),
            "recommended_carton": asdict(carton) if carton else None,
        }

    return app


app = create_app()
