import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from visual_packing.api import ProofServices, create_app
from visual_packing.catalog import CatalogAsset, CatalogVariant
from visual_packing.monitoring import ModelMonitor
from visual_packing.search import EmbeddingSearchEngine


def png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


class ColorEncoder:
    name = "test-color"

    def encode(self, payload: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(payload)) as image:
            return torch.tensor(image.convert("RGB").resize((1, 1)).getpixel((0, 0)), dtype=torch.float32)


def make_services() -> ProofServices:
    variant_red = CatalogVariant(sku="SKU-RED", name="1 oz Boston Round", dimensions_cm=(4, 4, 10))
    variant_blue = CatalogVariant(sku="SKU-BLUE", name="Wax Melt", dimensions_cm=(11, 8, 3), fragile=False)
    assets = [
        CatalogAsset("red", png((255, 0, 0)), "https://example/red.png", ("red",), ("Red Fragrance",), ("Fragrance",), (variant_red,)),
        CatalogAsset("blue", png((0, 0, 255)), "https://example/blue.png", ("blue",), ("Blue Wax",), ("Wax Melt",), (variant_blue,)),
    ]
    engine = EmbeddingSearchEngine(ColorEncoder())
    engine.index(assets)
    return ProofServices(
        engine=engine,
        assets=assets,
        samples=[
            {"asset_id": "red", "name": "Red Fragrance", "sample_url": "/static/samples/red.png", "sku": "SKU-RED"}
        ],
        benchmark={"p95_latency_ms": 29.9, "recall_at_5": 0.971429, "limitations": "Proof benchmark."},
        monitor=ModelMonitor(window_size=10),
    )


def test_monitor_is_bounded_and_reports_low_confidence_hold_and_drift() -> None:
    monitor = ModelMonitor(window_size=3, low_confidence_threshold=0.8, drift_similarity_floor=0.65)
    baseline = torch.tensor([1.0, 0.0])
    for _ in range(4):
        monitor.record(10.0, 0.5, torch.tensor([0.0, 1.0]), baseline, decision="HUMAN_REVIEW_HOLD")
    summary = monitor.summary()
    assert summary["samples_tracked"] == 3
    assert summary["low_confidence_count"] == 3
    assert summary["hold_rate"] == 1.0
    assert summary["drift_alert"] is True


def test_hold_rate_uses_only_decision_events() -> None:
    monitor = ModelMonitor(window_size=10)
    monitor.record(10.0, 0.9, None, None)
    monitor.record(12.0, 0.7, None, None, decision="HUMAN_REVIEW_HOLD")
    assert monitor.summary()["hold_rate"] == 1.0


def test_public_page_health_catalog_and_evidence() -> None:
    with TestClient(create_app(make_services())) as client:
        assert "Visual Search" in client.get("/").text
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["indexed_assets"] == 2
        assert client.get("/api/catalog").json()[0]["sku"] == "SKU-RED"
        assert client.get("/api/evidence").json()["p95_latency_ms"] == 29.9


def test_visual_search_accepts_image_and_updates_metrics() -> None:
    with TestClient(create_app(make_services())) as client:
        result = client.post(
            "/api/search?top_k=2",
            files={"file": ("red.png", png((250, 5, 5)), "image/png")},
        )
        assert result.status_code == 200
        assert result.json()["matches"][0]["asset_id"] == "red"
        assert client.get("/api/metrics").json()["samples_tracked"] == 1


def test_search_rejects_non_image_and_oversized_input() -> None:
    with TestClient(create_app(make_services(), max_upload_bytes=50)) as client:
        wrong_type = client.post(
            "/api/search", files={"file": ("note.txt", b"hello", "text/plain")}
        )
        assert wrong_type.status_code == 415
        too_large = client.post(
            "/api/search", files={"file": ("large.png", png((1, 2, 3)), "image/png")}
        )
        assert too_large.status_code == 413


def test_packing_route_rejects_wrong_barcode_and_unknown_sku() -> None:
    with TestClient(create_app(make_services())) as client:
        result = client.post(
            "/api/packing/verify",
            data={"expected_sku": "SKU-RED", "barcode_sku": "SKU-BLUE"},
            files={"file": ("red.png", png((255, 0, 0)), "image/png")},
        )
        assert result.status_code == 200
        assert result.json()["verification"]["decision"] == "REJECT_MISMATCH"
        assert result.json()["recommended_carton"] is None
        assert result.json()["search"]["matches"]
        missing = client.post(
            "/api/packing/verify",
            data={"expected_sku": "DOES-NOT-EXIST"},
            files={"file": ("red.png", png((255, 0, 0)), "image/png")},
        )
        assert missing.status_code == 404


def test_empty_index_returns_bounded_service_error() -> None:
    services = make_services()
    services.engine.assets = []
    services.engine.embeddings = torch.empty((0, 0))
    with TestClient(create_app(services)) as client:
        result = client.post(
            "/api/search", files={"file": ("red.png", png((255, 0, 0)), "image/png")}
        )
        assert result.status_code == 503
        assert result.json()["detail"] == "Visual search index is unavailable."
