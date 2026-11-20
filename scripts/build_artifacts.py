from __future__ import annotations

import json
from pathlib import Path

from visual_packing.benchmark import build_artifacts, download_assets
from visual_packing.catalog import load_source_catalog
from visual_packing.search import MobileNetV3Encoder


def main() -> None:
    products = load_source_catalog("data/catalog.source.json")
    downloads = download_assets(products)
    print(f"Downloaded {len(downloads)} of {len(products)} catalog images")
    report = build_artifacts(
        products,
        downloads,
        Path("artifacts"),
        Path("src/visual_packing/static/samples"),
        MobileNetV3Encoder(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
