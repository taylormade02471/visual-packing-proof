from __future__ import annotations

import hashlib
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator


FORMAT_DIMENSIONS_CM: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("1/3 oz boston round", (3, 3, 8)),
    ("1/2 oz boston round", (3, 3, 9)),
    ("1 oz boston round", (4, 4, 10)),
    ("2 oz boston round", (5, 5, 12)),
    ("1 oz roll on", (3, 3, 11)),
    ("cologne spray", (5, 4, 13)),
    ("body mist", (5, 5, 16)),
    ("body wash", (6, 6, 18)),
    ("body lotion", (6, 6, 18)),
    ("lotion", (6, 6, 18)),
    ("massage oil", (5, 5, 15)),
    ("body butter", (9, 9, 7)),
    ("sugar scrub", (9, 9, 8)),
    ("air freshener", (4, 4, 12)),
    ("gift set", (24, 18, 10)),
    ("wax melt", (11, 8, 3)),
    ("candle", (9, 9, 11)),
)


class CatalogVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dimensions_cm: tuple[float, float, float]
    fragile: bool = True

    @field_validator("dimensions_cm")
    @classmethod
    def positive_dimensions(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(dimension <= 0 for dimension in value):
            raise ValueError("dimensions must be positive")
        return value


class CatalogItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    image_url: str = Field(pattern=r"^https://")
    variants: tuple[CatalogVariant, ...] = Field(min_length=1)


@dataclass(frozen=True)
class CatalogAsset:
    sha256: str
    image_bytes: bytes
    image_url: str
    product_ids: tuple[str, ...]
    product_names: tuple[str, ...]
    categories: tuple[str, ...]
    variants: tuple[CatalogVariant, ...]


@dataclass(frozen=True)
class AssetRejection:
    product_id: str
    image_url: str
    reason: str


def load_source_catalog(path: str | Path) -> list[CatalogItem]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("catalog root must be a list")
    products = [CatalogItem.model_validate(record) for record in payload]
    product_ids = [product.product_id for product in products]
    duplicates = sorted({value for value in product_ids if product_ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate product_id: {', '.join(duplicates)}")
    return products


def _valid_image(image_bytes: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def group_catalog_assets(
    products: Iterable[CatalogItem], assets: Mapping[str, bytes]
) -> tuple[list[CatalogAsset], list[AssetRejection]]:
    grouped: dict[str, dict[str, object]] = {}
    rejected: list[AssetRejection] = []
    for product in sorted(products, key=lambda record: record.product_id):
        image_bytes = assets.get(product.image_url)
        if image_bytes is None:
            rejected.append(AssetRejection(product.product_id, product.image_url, "missing_asset"))
            continue
        if not _valid_image(image_bytes):
            rejected.append(AssetRejection(product.product_id, product.image_url, "invalid_image"))
            continue
        digest = hashlib.sha256(image_bytes).hexdigest()
        entry = grouped.setdefault(
            digest,
            {
                "bytes": image_bytes,
                "url": product.image_url,
                "ids": [],
                "names": [],
                "categories": [],
                "variants": [],
            },
        )
        entry["ids"].append(product.product_id)  # type: ignore[union-attr]
        entry["names"].append(product.name)  # type: ignore[union-attr]
        entry["categories"].append(product.category)  # type: ignore[union-attr]
        entry["variants"].extend(product.variants)  # type: ignore[union-attr]

    accepted = [
        CatalogAsset(
            sha256=digest,
            image_bytes=entry["bytes"],  # type: ignore[arg-type]
            image_url=entry["url"],  # type: ignore[arg-type]
            product_ids=tuple(entry["ids"]),  # type: ignore[arg-type]
            product_names=tuple(entry["names"]),  # type: ignore[arg-type]
            categories=tuple(entry["categories"]),  # type: ignore[arg-type]
            variants=tuple(entry["variants"]),  # type: ignore[arg-type]
        )
        for digest, entry in sorted(grouped.items())
    ]
    return accepted, rejected


def split_catalog(
    assets: Iterable[CatalogAsset], seed: int = 42
) -> dict[str, list[CatalogAsset]]:
    records = sorted(assets, key=lambda asset: asset.sha256)
    random.Random(seed).shuffle(records)
    count = len(records)
    if count < 3:
        return {"train": records, "validation": [], "test": []}
    validation_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    if validation_count + test_count >= count:
        validation_count = test_count = 1
    train_count = count - validation_count - test_count
    result = {
        "train": records[:train_count],
        "validation": records[train_count : train_count + validation_count],
        "test": records[train_count + validation_count :],
    }
    for values in result.values():
        values.sort(key=lambda asset: asset.sha256)
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def _dimensions_for(variant_name: str, category: str) -> tuple[float, float, float]:
    combined = f"{variant_name} {category}".lower()
    for token, dimensions in FORMAT_DIMENSIONS_CM:
        if token in combined:
            return dimensions
    return (9, 9, 11)


def parse_bundle_catalog(bundle_text: str) -> list[CatalogItem]:
    """Extract the public product literal without executing remote JavaScript."""
    marker = "var vp=["
    start = bundle_text.find(marker)
    if start < 0:
        raise ValueError("Stone Wick product catalog marker was not found")
    end = bundle_text.find("],yp={", start)
    if end < 0:
        raise ValueError("Stone Wick product catalog terminator was not found")
    catalog_text = bundle_text[start + len(marker) : end + 1]
    pattern = re.compile(
        r"\{id:`(?P<id>[^`]+)`,handle:`(?P<handle>[^`]+)`,name:`(?P<name>[^`]+)`,"
        r"category:`(?P<category>[^`]+)`.*?image:`(?P<image>https://[^`]+)`.*?"
        r"variants:\[(?P<variants>.*?)\]\}",
        re.DOTALL,
    )
    products: list[CatalogItem] = []
    for match in pattern.finditer(catalog_text):
        variant_names = re.findall(r"`([^`]+)`", match.group("variants")) or ["Standard"]
        handle = match.group("handle")
        category = match.group("category")
        variants = tuple(
            CatalogVariant(
                sku=f"SW-{_slug(handle)}-{_slug(variant_name)}",
                name=variant_name,
                dimensions_cm=_dimensions_for(variant_name, category),
                fragile="wax melt" not in variant_name.lower(),
            )
            for variant_name in variant_names
        )
        products.append(
            CatalogItem(
                product_id=match.group("id"),
                name=match.group("name"),
                category=category,
                image_url=match.group("image"),
                variants=variants,
            )
        )
    if not products:
        raise ValueError("Stone Wick product catalog contained no parseable products")
    return products
