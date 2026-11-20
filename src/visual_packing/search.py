from __future__ import annotations

import io
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

import numpy as np
import torch
from PIL import Image, ImageEnhance, UnidentifiedImageError
from torchvision import models

from .catalog import CatalogAsset


class EmptyIndexError(RuntimeError):
    pass


class ImageEncoder(Protocol):
    name: str

    def encode(self, payload: bytes) -> torch.Tensor: ...


class MobileNetV3Encoder:
    name = "torchvision/mobilenet_v3_large-imagenet1k-v2"

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        backbone = models.mobilenet_v3_large(weights=weights)
        self.model = torch.nn.Sequential(backbone.features, backbone.avgpool, torch.nn.Flatten()).to(self.device)
        self.model.eval()
        self.transform = weights.transforms()
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 224, 224, device=self.device))

    @torch.inference_mode()
    def encode(self, payload: bytes) -> torch.Tensor:
        try:
            with Image.open(io.BytesIO(payload)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError("invalid image") from error
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze(0).detach().cpu().to(torch.float32)


class HybridVisualEncoder:
    """Combine semantic PyTorch features with a crop/rotation-resistant RGB histogram."""

    def __init__(
        self,
        base: ImageEncoder,
        *,
        histogram_bins: int = 16,
        histogram_weight: float = 1.5,
    ) -> None:
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        if histogram_weight <= 0:
            raise ValueError("histogram_weight must be positive")
        self.base = base
        self.histogram_bins = histogram_bins
        self.histogram_weight = histogram_weight
        self.name = f"{base.name}+rgb-histogram-{histogram_bins}"

    def encode(self, payload: bytes) -> torch.Tensor:
        base_vector = self.base.encode(payload).detach().cpu().to(torch.float32).flatten()
        base_norm = torch.linalg.vector_norm(base_vector)
        if not torch.isfinite(base_norm) or base_norm <= 0:
            raise ValueError("base encoder returned a zero-length or non-finite embedding")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                image = source.convert("RGB").resize((128, 128))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError("invalid image") from error
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()) / 255.0
        histograms = []
        for channel in range(3):
            histogram = torch.histc(
                pixels[:, :, channel], bins=self.histogram_bins, min=0.0, max=1.0
            )
            histograms.append(histogram / histogram.sum().clamp_min(1.0))
        color_vector = torch.cat(histograms)
        color_vector = color_vector / torch.linalg.vector_norm(color_vector).clamp_min(1e-12)
        return torch.cat((base_vector / base_norm, color_vector * self.histogram_weight))


@dataclass(frozen=True)
class SearchMatch:
    asset_id: str
    product_ids: tuple[str, ...]
    product_names: tuple[str, ...]
    categories: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class SearchResponse:
    latency_ms: float
    matches: tuple[SearchMatch, ...]
    query_embedding: torch.Tensor


class EmbeddingSearchEngine:
    def __init__(self, encoder: ImageEncoder) -> None:
        self.encoder = encoder
        self.assets: list[CatalogAsset] = []
        self.embeddings = torch.empty((0, 0), dtype=torch.float32)

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        flat = vector.detach().cpu().to(torch.float32).flatten()
        norm = torch.linalg.vector_norm(flat)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("encoder returned a zero-length or non-finite embedding")
        return flat / norm

    def index(self, assets: Iterable[CatalogAsset], *, augmentations_per_asset: int = 0) -> None:
        if augmentations_per_asset < 0:
            raise ValueError("augmentations_per_asset cannot be negative")
        self.assets = list(assets)
        if not self.assets:
            self.embeddings = torch.empty((0, 0), dtype=torch.float32)
            return
        vectors: list[torch.Tensor] = []
        for asset_position, asset in enumerate(self.assets):
            views = [self._normalize(self.encoder.encode(asset.image_bytes))]
            for augmentation in range(augmentations_per_asset):
                transformed = camera_perturbation(
                    asset.image_bytes, seed=10_000 + asset_position * 101 + augmentation
                )
                views.append(self._normalize(self.encoder.encode(transformed)))
            vectors.append(self._normalize(torch.stack(views).mean(dim=0)))
        dimensions = {vector.numel() for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("encoder returned inconsistent embedding dimensions")
        self.embeddings = torch.stack(vectors)

    def search(self, image_bytes: bytes, top_k: int = 5) -> SearchResponse:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.assets or self.embeddings.numel() == 0:
            raise EmptyIndexError("visual search index is empty")
        started = time.perf_counter()
        query = self._normalize(self.encoder.encode(image_bytes))
        if query.numel() != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match index")
        scores = torch.mv(self.embeddings, query)
        count = min(top_k, len(self.assets))
        values, indices = torch.topk(scores, count, largest=True, sorted=True)
        matches = tuple(
            SearchMatch(
                asset_id=self.assets[index].sha256,
                product_ids=self.assets[index].product_ids,
                product_names=self.assets[index].product_names,
                categories=self.assets[index].categories,
                score=round(float(score), 6),
            )
            for score, index in zip(values.tolist(), indices.tolist(), strict=True)
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(round(latency_ms, 3), matches, query)


def camera_perturbation(payload: bytes, seed: int = 42) -> bytes:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("invalid image") from error
    width, height = image.size
    inset_x = max(1, round(width * 0.04))
    inset_y = max(1, round(height * 0.04))
    image = image.crop((inset_x, inset_y, width - inset_x, height - inset_y))
    angle = 3 + (seed % 4)
    image = image.rotate(angle, fillcolor=(245, 245, 245))
    image = ImageEnhance.Brightness(image).enhance(1.05 + (seed % 3) * 0.02)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def run_benchmark(
    engine: EmbeddingSearchEngine,
    assets: Iterable[CatalogAsset],
    *,
    perturb: bool = True,
    gallery_augmentations: int = 3,
) -> dict[str, object]:
    records = list(assets)
    if not records:
        raise ValueError("benchmark requires at least one catalog asset")
    engine.index(records, augmentations_per_asset=gallery_augmentations)
    latencies: list[float] = []
    hits_at_1 = 0
    hits_at_5 = 0
    for position, asset in enumerate(records):
        query = camera_perturbation(asset.image_bytes, seed=position) if perturb else asset.image_bytes
        response = engine.search(query, top_k=5)
        latencies.append(response.latency_ms)
        expected = set(asset.product_ids)
        returned = [set(match.product_ids) for match in response.matches]
        hits_at_1 += int(bool(returned and expected.intersection(returned[0])))
        hits_at_5 += int(any(expected.intersection(candidate) for candidate in returned))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "encoder": engine.encoder.name,
        "environment": f"{platform.system()} {platform.machine()} / torch {torch.__version__}",
        "queries_tested": len(records),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
        "sla_under_200ms_hot_path": bool(np.percentile(latencies, 95) < 200.0),
        "recall_at_1": round(hits_at_1 / len(records), 6),
        "recall_at_5": round(hits_at_5 / len(records), 6),
        "latency_scope": "hot model and in-memory search path",
        "limitations": (
            "Camera-style perturbations of the owner's public catalog are a proof benchmark, "
            "not a production accuracy claim. Client warehouse images and traffic are required "
            "to validate >99.9% SKU accuracy and conversion lift."
        ),
    }
