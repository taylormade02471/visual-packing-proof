import io

import pytest
import torch
from PIL import Image

from visual_packing.benchmark import build_artifacts
from visual_packing.catalog import CatalogAsset, CatalogVariant
from visual_packing.search import EmbeddingSearchEngine, EmptyIndexError, HybridVisualEncoder, run_benchmark


def image_bytes(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, format="PNG")
    return stream.getvalue()


class ColorEncoder:
    name = "test-color-encoder"

    def encode(self, payload: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(payload)) as image:
            red, green, blue = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
        return torch.tensor([red, green, blue], dtype=torch.float32)


def asset(identifier: str, color: tuple[int, int, int]) -> CatalogAsset:
    payload = image_bytes(color)
    return CatalogAsset(
        sha256=identifier,
        image_bytes=payload,
        image_url=f"https://example/{identifier}.png",
        product_ids=(identifier,),
        product_names=(f"Product {identifier}",),
        categories=("Test",),
        variants=(CatalogVariant(sku=f"SKU-{identifier}", name="Standard", dimensions_cm=(4, 4, 10)),),
    )


def test_index_embeddings_are_l2_normalized() -> None:
    engine = EmbeddingSearchEngine(ColorEncoder())
    engine.index([asset("red", (255, 0, 0)), asset("green", (0, 255, 0))])
    norms = torch.linalg.vector_norm(engine.embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_index_can_average_offline_camera_views_without_adding_results() -> None:
    class CountingEncoder(ColorEncoder):
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, payload: bytes) -> torch.Tensor:
            self.calls += 1
            return super().encode(payload)

    encoder = CountingEncoder()
    engine = EmbeddingSearchEngine(encoder)
    engine.index([asset("red", (255, 0, 0))], augmentations_per_asset=2)
    assert encoder.calls == 3
    assert len(engine.assets) == 1
    assert engine.embeddings.shape[0] == 1
    assert torch.allclose(torch.linalg.vector_norm(engine.embeddings, dim=1), torch.ones(1))


def test_search_ranks_the_matching_product_first() -> None:
    engine = EmbeddingSearchEngine(ColorEncoder())
    engine.index([asset("red", (255, 0, 0)), asset("green", (0, 255, 0))])
    response = engine.search(image_bytes((250, 5, 5)), top_k=2)
    assert response.matches[0].product_ids == ("red",)
    assert response.matches[0].score > response.matches[1].score
    assert response.latency_ms >= 0


def test_search_rejects_empty_index_and_bounds_top_k() -> None:
    engine = EmbeddingSearchEngine(ColorEncoder())
    with pytest.raises(EmptyIndexError):
        engine.search(image_bytes((1, 2, 3)))
    engine.index([asset("red", (255, 0, 0))])
    assert len(engine.search(image_bytes((255, 0, 0)), top_k=50).matches) == 1
    with pytest.raises(ValueError, match="top_k"):
        engine.search(image_bytes((255, 0, 0)), top_k=0)


def test_zero_vector_is_rejected() -> None:
    class ZeroEncoder:
        name = "zero"

        def encode(self, payload: bytes) -> torch.Tensor:
            return torch.zeros(3)

    engine = EmbeddingSearchEngine(ZeroEncoder())
    with pytest.raises(ValueError, match="zero-length"):
        engine.index([asset("red", (255, 0, 0))])


def test_hybrid_encoder_adds_finite_color_signature() -> None:
    class BaseEncoder:
        name = "base"

        def encode(self, payload: bytes) -> torch.Tensor:
            return torch.tensor([1.0, 2.0, 3.0])

    encoder = HybridVisualEncoder(BaseEncoder(), histogram_bins=8, histogram_weight=1.5)
    vector = encoder.encode(image_bytes((255, 0, 0)))
    assert vector.numel() == 3 + (3 * 8)
    assert torch.isfinite(vector).all()
    assert encoder.name == "base+rgb-histogram-8"


def test_benchmark_records_reproducible_metrics_and_limitations() -> None:
    assets = [asset("red", (255, 0, 0)), asset("green", (0, 255, 0)), asset("blue", (0, 0, 255))]
    engine = EmbeddingSearchEngine(ColorEncoder())
    report = run_benchmark(engine, assets, perturb=False)
    assert report["queries_tested"] == 3
    assert report["recall_at_1"] == 1.0
    assert report["recall_at_5"] == 1.0
    assert report["p95_latency_ms"] >= 0
    assert report["encoder"] == "test-color-encoder"
    assert report["latency_scope"] == "hot model and in-memory search path"
    assert "not a production accuracy claim" in report["limitations"].lower()


def test_build_artifacts_writes_manifest_splits_embeddings_and_samples(tmp_path) -> None:
    from visual_packing.catalog import CatalogItem

    products = [
        CatalogItem(
            product_id=color,
            name=color.title(),
            category="Test",
            image_url=f"https://example/{color}.png",
            variants=(CatalogVariant(sku=f"SKU-{color}", name="Standard", dimensions_cm=(4, 4, 10)),),
        )
        for color in ("red", "green", "blue")
    ]
    payloads = {
        "https://example/red.png": image_bytes((255, 0, 0)),
        "https://example/green.png": image_bytes((0, 255, 0)),
        "https://example/blue.png": image_bytes((0, 0, 255)),
    }
    output = tmp_path / "artifacts"
    samples = tmp_path / "samples"
    report = build_artifacts(products, payloads, output, samples, ColorEncoder(), perturb=False)
    assert report["queries_tested"] == 3
    assert (output / "catalog.json").is_file()
    assert (output / "dataset_manifest.json").is_file()
    assert (output / "embeddings.pt").is_file()
    assert (output / "benchmark_report.json").is_file()
    assert len(list(samples.glob("*.jpg"))) == 3
