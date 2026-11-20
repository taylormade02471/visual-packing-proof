import torch

from visual_packing.packing import PackageItem, select_carton, verify_item
from visual_packing.search import SearchMatch, SearchResponse


def response(*matches: tuple[str, float]) -> SearchResponse:
    return SearchResponse(
        latency_ms=12.0,
        matches=tuple(
            SearchMatch(
                asset_id=asset_id,
                product_ids=(asset_id,),
                product_names=(asset_id.title(),),
                categories=("Fragrance",),
                score=score,
            )
            for asset_id, score in matches
        ),
        query_embedding=torch.tensor([1.0, 0.0]),
    )


def test_matching_barcode_accepts_with_deterministic_evidence() -> None:
    result = verify_item(response(("other", 0.99)), "SKU-RED", "red", barcode_sku="SKU-RED")
    assert result.verified is True
    assert result.decision == "PASS_PACK"
    assert result.tier == "BARCODE_MATCH"
    assert result.confidence == 1.0


def test_wrong_barcode_rejects_before_visual_evidence() -> None:
    result = verify_item(response(("red", 0.99)), "SKU-RED", "red", barcode_sku="SKU-BLUE")
    assert result.verified is False
    assert result.decision == "REJECT_MISMATCH"
    assert result.tier == "BARCODE_CHECK"


def test_confident_visual_match_passes() -> None:
    result = verify_item(response(("red", 0.94), ("blue", 0.70)), "SKU-RED", "red")
    assert result.verified is True
    assert result.decision == "PASS_PACK"
    assert result.tier == "VISUAL_FAST_PATH"


def test_visual_match_for_shared_variant_image_holds_without_barcode() -> None:
    result = verify_item(
        response(("red", 0.99), ("blue", 0.60)),
        "SKU-RED-1OZ",
        "red",
        visual_variant_count=2,
    )
    assert result.verified is False
    assert result.decision == "HUMAN_REVIEW_HOLD"
    assert "multiple sku variants" in result.reason.lower()


def test_confident_wrong_product_rejects() -> None:
    result = verify_item(response(("blue", 0.95), ("red", 0.60)), "SKU-RED", "red")
    assert result.verified is False
    assert result.decision == "REJECT_MISMATCH"
    assert "blue" in result.reason


def test_ambiguous_or_empty_visual_result_holds_for_human_review() -> None:
    ambiguous = verify_item(response(("red", 0.90), ("blue", 0.88)), "SKU-RED", "red")
    assert ambiguous.decision == "HUMAN_REVIEW_HOLD"
    assert ambiguous.verified is False
    empty = verify_item(response(), "SKU-RED", "red")
    assert empty.decision == "HUMAN_REVIEW_HOLD"


def test_ambiguous_wrong_top_result_holds_instead_of_rejecting() -> None:
    result = verify_item(response(("blue", 0.95), ("red", 0.94)), "SKU-RED", "red")
    assert result.decision == "HUMAN_REVIEW_HOLD"
    assert result.verified is False


def test_carton_fit_is_orientation_aware() -> None:
    result = select_carton([PackageItem("SKU-1", (20, 10, 5), fragile=False)])
    assert result.box_id == "SMALL_MAILER"
    assert result.fill_efficiency > 0


def test_fragile_padding_can_move_item_to_larger_carton() -> None:
    unpadded = select_carton([PackageItem("SKU-1", (18, 12, 5), fragile=False)])
    padded = select_carton([PackageItem("SKU-1", (18, 12, 5), fragile=True)])
    assert unpadded.box_id == "SMALL_MAILER"
    assert padded.box_id == "MEDIUM_CARTON"
    assert padded.protective_padding_cm == 4.0


def test_no_standard_fit_returns_custom_packout() -> None:
    result = select_carton([PackageItem("BIG", (90, 80, 70), fragile=True)])
    assert result.box_id == "CUSTOM_PACKOUT"
    assert result.box_dimensions_cm is None
    assert result.fill_efficiency == 0.0
