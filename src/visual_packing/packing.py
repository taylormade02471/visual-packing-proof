from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

from .search import SearchResponse


@dataclass(frozen=True)
class VerificationDecision:
    verified: bool
    decision: str
    tier: str
    confidence: float
    reason: str


def verify_item(
    search: SearchResponse,
    expected_sku: str,
    expected_asset_id: str,
    *,
    barcode_sku: str | None = None,
    visual_variant_count: int = 1,
    score_threshold: float = 0.85,
    margin_threshold: float = 0.05,
) -> VerificationDecision:
    if barcode_sku:
        if barcode_sku == expected_sku:
            return VerificationDecision(
                True,
                "PASS_PACK",
                "BARCODE_MATCH",
                1.0,
                "Scanned barcode exactly matches the expected SKU.",
            )
        return VerificationDecision(
            False,
            "REJECT_MISMATCH",
            "BARCODE_CHECK",
            1.0,
            f"Scanned barcode {barcode_sku} does not match expected SKU {expected_sku}.",
        )

    if not search.matches:
        return VerificationDecision(
            False,
            "HUMAN_REVIEW_HOLD",
            "SAFETY_GATE",
            0.0,
            "Visual search returned no supported catalog match.",
        )

    top = search.matches[0]
    second_score = search.matches[1].score if len(search.matches) > 1 else 0.0
    margin = top.score - second_score
    if top.score < score_threshold or margin < margin_threshold:
        return VerificationDecision(
            False,
            "HUMAN_REVIEW_HOLD",
            "SAFETY_GATE",
            top.score,
            (
                f"Visual evidence is uncertain: score {top.score:.3f}, margin {margin:.3f}. "
                "The item was not auto-approved."
            ),
        )
    if top.asset_id != expected_asset_id:
        return VerificationDecision(
            False,
            "REJECT_MISMATCH",
            "VISUAL_MISMATCH",
            top.score,
            f"Top visual asset {top.asset_id} does not match expected asset {expected_asset_id}.",
        )
    if visual_variant_count > 1:
        return VerificationDecision(
            False,
            "HUMAN_REVIEW_HOLD",
            "VARIANT_EVIDENCE_REQUIRED",
            top.score,
            (
                f"The matched image represents multiple SKU variants ({visual_variant_count}). "
                "A matching barcode or variant-specific evidence is required."
            ),
        )
    if top.asset_id == expected_asset_id:
        return VerificationDecision(
            True,
            "PASS_PACK",
            "VISUAL_FAST_PATH",
            top.score,
            f"Expected asset ranked first with a {margin:.3f} confidence margin.",
        )
    raise AssertionError("unreachable verification state")


@dataclass(frozen=True)
class PackageItem:
    sku: str
    dimensions_cm: tuple[float, float, float]
    fragile: bool = True

    def __post_init__(self) -> None:
        if any(dimension <= 0 for dimension in self.dimensions_cm):
            raise ValueError("package dimensions must be positive")


@dataclass(frozen=True)
class CartonRecommendation:
    box_id: str
    box_dimensions_cm: tuple[float, float, float] | None
    required_dimensions_cm: tuple[float, float, float]
    protective_padding_cm: float
    fill_efficiency: float


STANDARD_CARTONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("SMALL_MAILER", (22, 16, 8)),
    ("MEDIUM_CARTON", (35, 25, 15)),
    ("LARGE_CARTON", (50, 40, 30)),
    ("XL_CARTON", (65, 50, 40)),
)


def _fits(required: tuple[float, float, float], box: tuple[float, float, float]) -> bool:
    return any(all(item <= limit for item, limit in zip(required, orientation, strict=True)) for orientation in permutations(box))


def select_carton(items: Iterable[PackageItem], fragile_padding_cm: float = 4.0) -> CartonRecommendation:
    records = list(items)
    if not records:
        raise ValueError("at least one package item is required")
    if fragile_padding_cm < 0:
        raise ValueError("fragile_padding_cm cannot be negative")

    oriented: list[tuple[float, float, float]] = []
    raw_volume = 0.0
    any_fragile = False
    for item in records:
        raw_volume += item.dimensions_cm[0] * item.dimensions_cm[1] * item.dimensions_cm[2]
        padding = fragile_padding_cm if item.fragile else 0.0
        any_fragile = any_fragile or item.fragile
        padded = tuple(dimension + padding for dimension in item.dimensions_cm)
        oriented.append(tuple(sorted(padded, reverse=True)))

    required = (
        max(dimensions[0] for dimensions in oriented),
        max(dimensions[1] for dimensions in oriented),
        sum(dimensions[2] for dimensions in oriented),
    )
    for box_id, dimensions in sorted(
        STANDARD_CARTONS, key=lambda record: record[1][0] * record[1][1] * record[1][2]
    ):
        if _fits(required, dimensions):
            box_volume = dimensions[0] * dimensions[1] * dimensions[2]
            return CartonRecommendation(
                box_id,
                dimensions,
                required,
                fragile_padding_cm if any_fragile else 0.0,
                round(raw_volume / box_volume, 4),
            )
    return CartonRecommendation(
        "CUSTOM_PACKOUT",
        None,
        required,
        fragile_padding_cm if any_fragile else 0.0,
        0.0,
    )
