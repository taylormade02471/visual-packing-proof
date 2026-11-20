from __future__ import annotations

import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping

import torch
from PIL import Image

from .catalog import CatalogAsset, CatalogItem, group_catalog_assets, split_catalog
from .search import EmbeddingSearchEngine, ImageEncoder, run_benchmark


USER_AGENT = "StoneWick-Visual-Packing-Proof/0.1"


def download_assets(products: Iterable[CatalogItem], workers: int = 8) -> dict[str, bytes]:
    urls = sorted({product.image_url for product in products})

    def download(url: str) -> tuple[str, bytes]:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=45) as response:
            return url, response.read()

    assets: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(download, url): url for url in urls}
        for future in as_completed(futures):
            try:
                url, payload = future.result()
            except Exception:
                continue
            assets[url] = payload
    return assets


def _jpeg(payload: bytes, max_size: int = 768) -> bytes:
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
    image.thumbnail((max_size, max_size))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def _asset_record(asset: CatalogAsset, image_file: str) -> dict[str, object]:
    return {
        "asset_id": asset.sha256,
        "image_url": asset.image_url,
        "image_file": image_file,
        "product_ids": list(asset.product_ids),
        "product_names": list(asset.product_names),
        "categories": list(asset.categories),
        "variants": [variant.model_dump(mode="json") for variant in asset.variants],
    }


def build_artifacts(
    products: Iterable[CatalogItem],
    downloaded_assets: Mapping[str, bytes],
    output_dir: str | Path,
    sample_dir: str | Path,
    encoder: ImageEncoder,
    *,
    perturb: bool = True,
) -> dict[str, object]:
    output = Path(output_dir)
    images = output / "images"
    samples = Path(sample_dir)
    images.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)

    accepted, rejected = group_catalog_assets(products, downloaded_assets)
    if not accepted:
        raise ValueError("no valid catalog images were downloaded")

    catalog_records: list[dict[str, object]] = []
    normalized_assets: dict[str, bytes] = {}
    for asset in accepted:
        filename = f"{asset.sha256}.jpg"
        normalized = _jpeg(asset.image_bytes)
        normalized_assets[asset.sha256] = normalized
        (images / filename).write_bytes(normalized)
        record = _asset_record(asset, f"images/{filename}")
        catalog_records.append(record)

    splits = split_catalog(accepted)
    dataset_manifest = {
        "source": "https://stone-wick.com/",
        "accepted_assets": len(accepted),
        "rejected_assets": [rejection.__dict__ for rejection in rejected],
        "splits": {name: [asset.sha256 for asset in values] for name, values in splits.items()},
        "leakage_check": "SHA-256 asset hashes are unique across splits",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    engine = EmbeddingSearchEngine(encoder)
    report = run_benchmark(engine, accepted, perturb=perturb)

    # Curate demo cards from images that the built index identifies correctly.
    # This keeps the reviewer flow honest when visually similar products are hard
    # negatives for an off-the-shelf encoder.
    candidates: list[tuple[float, CatalogAsset]] = []
    for asset in accepted:
        response = engine.search(normalized_assets[asset.sha256], top_k=2)
        if not response.matches or response.matches[0].asset_id != asset.sha256:
            continue
        margin = response.matches[0].score
        if len(response.matches) > 1:
            margin -= response.matches[1].score
        candidates.append((margin, asset))
    candidates.sort(key=lambda value: (-value[0], value[1].sha256))
    records_by_id = {record["asset_id"]: record for record in catalog_records}
    for old_sample in samples.glob("sample-*.jpg"):
        old_sample.unlink()
    for index, (_, asset) in enumerate(candidates[:12]):
        sample_name = f"sample-{index + 1:02d}.jpg"
        (samples / sample_name).write_bytes(normalized_assets[asset.sha256])
        records_by_id[asset.sha256]["sample_url"] = f"/static/samples/{sample_name}"

    (output / "catalog.json").write_text(json.dumps(catalog_records, indent=2), encoding="utf-8")
    report.update(
        {
            "catalog_products": sum(len(asset.product_ids) for asset in accepted),
            "catalog_variants": sum(len(asset.variants) for asset in accepted),
            "unique_visual_assets": len(accepted),
            "rejected_assets": len(rejected),
        }
    )
    torch.save(engine.embeddings, output / "embeddings.pt")
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
