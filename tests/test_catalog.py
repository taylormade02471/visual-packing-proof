import hashlib
import io
import json

import pytest
from PIL import Image
from pydantic import ValidationError

from visual_packing.catalog import (
    CatalogItem,
    group_catalog_assets,
    load_source_catalog,
    parse_bundle_catalog,
    split_catalog,
)


def png(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(stream, format="PNG")
    return stream.getvalue()


def item(product_id: str, image_url: str, sku: str, category: str = "Fragrance") -> CatalogItem:
    return CatalogItem.model_validate(
        {
            "product_id": product_id,
            "name": f"Product {product_id}",
            "category": category,
            "image_url": image_url,
            "variants": [
                {
                    "sku": sku,
                    "name": "1 oz Boston Round",
                    "dimensions_cm": [4, 4, 10],
                    "fragile": True,
                }
            ],
        }
    )


def test_catalog_schema_rejects_missing_identity() -> None:
    with pytest.raises(ValidationError):
        CatalogItem.model_validate({"name": "Missing ID", "variants": []})


def test_load_source_catalog_rejects_duplicate_product_ids(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    payload = [item("p1", "https://example/p1.png", "SKU-1").model_dump(mode="json")] * 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate product_id"):
        load_source_catalog(path)


def test_shared_image_is_one_asset_with_all_products_and_variants() -> None:
    shared = png((90, 40, 20))
    accepted, rejected = group_catalog_assets(
        [
            item("p1", "https://example/shared.png", "SKU-1"),
            item("p2", "https://example/shared.png", "SKU-2"),
        ],
        {"https://example/shared.png": shared},
    )
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].product_ids == ("p1", "p2")
    assert {variant.sku for variant in accepted[0].variants} == {"SKU-1", "SKU-2"}
    assert accepted[0].sha256 == hashlib.sha256(shared).hexdigest()


def test_corrupt_image_is_rejected_with_reason() -> None:
    accepted, rejected = group_catalog_assets(
        [item("bad", "https://example/bad.png", "BAD-1")],
        {"https://example/bad.png": b"not an image"},
    )
    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0].product_id == "bad"
    assert rejected[0].reason == "invalid_image"


def test_split_is_deterministic_and_hashes_never_cross_splits() -> None:
    products = [item(f"p{i}", f"https://example/{i}.png", f"SKU-{i}") for i in range(20)]
    assets = {f"https://example/{i}.png": png((i, i * 2, i * 3)) for i in range(20)}
    accepted, rejected = group_catalog_assets(products, assets)
    assert rejected == []

    first = split_catalog(accepted, seed=42)
    second = split_catalog(list(reversed(accepted)), seed=42)
    assert {key: [a.sha256 for a in value] for key, value in first.items()} == {
        key: [a.sha256 for a in value] for key, value in second.items()
    }
    assert all(first[name] for name in ("train", "validation", "test"))
    split_hashes = [{asset.sha256 for asset in first[name]} for name in ("train", "validation", "test")]
    assert split_hashes[0].isdisjoint(split_hashes[1])
    assert split_hashes[0].isdisjoint(split_hashes[2])
    assert split_hashes[1].isdisjoint(split_hashes[2])
    assert sum(map(len, split_hashes)) == len(accepted)


def test_parse_bundle_catalog_extracts_products_without_evaluating_javascript() -> None:
    bundle = """var vp=[{id:`amber-candle`,handle:`amber-candle`,name:`Amber Candle`,category:`Candle`,size:`2 variants`,price:12,image:`https://cdn.example/amber.png`,aura:`#000`,variants:[`8 oz Candle`,`Wax Melt`]}],yp={};"""
    products = parse_bundle_catalog(bundle)
    assert len(products) == 1
    assert products[0].product_id == "amber-candle"
    assert products[0].image_url == "https://cdn.example/amber.png"
    assert [variant.name for variant in products[0].variants] == ["8 oz Candle", "Wax Melt"]
    assert all(variant.sku.startswith("SW-AMBER-CANDLE-") for variant in products[0].variants)
