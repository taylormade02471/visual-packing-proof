from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch


@dataclass(frozen=True)
class MonitorEvent:
    latency_ms: float
    top_score: float
    centroid_similarity: float | None
    decision: str | None


class ModelMonitor:
    def __init__(
        self,
        window_size: int = 200,
        low_confidence_threshold: float = 0.80,
        drift_similarity_floor: float = 0.65,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.events: deque[MonitorEvent] = deque(maxlen=window_size)
        self.low_confidence_threshold = low_confidence_threshold
        self.drift_similarity_floor = drift_similarity_floor
        self._lock = Lock()

    def record(
        self,
        latency_ms: float,
        top_score: float,
        query_embedding: torch.Tensor | None,
        baseline_centroid: torch.Tensor | None,
        *,
        decision: str | None = None,
    ) -> None:
        similarity: float | None = None
        if query_embedding is not None and baseline_centroid is not None:
            query = query_embedding.flatten().to(torch.float32)
            baseline = baseline_centroid.flatten().to(torch.float32)
            if query.numel() == baseline.numel():
                similarity = float(torch.nn.functional.cosine_similarity(query, baseline, dim=0))
        with self._lock:
            self.events.append(MonitorEvent(latency_ms, top_score, similarity, decision))

    def summary(self) -> dict[str, object]:
        with self._lock:
            events = list(self.events)
        if not events:
            return {
                "samples_tracked": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "low_confidence_count": 0,
                "hold_rate": 0.0,
                "mean_centroid_similarity": None,
                "drift_alert": False,
            }
        latencies = [event.latency_ms for event in events]
        similarities = [event.centroid_similarity for event in events if event.centroid_similarity is not None]
        mean_similarity = float(np.mean(similarities)) if similarities else None
        decision_events = [event for event in events if event.decision is not None]
        hold_count = sum(event.decision == "HUMAN_REVIEW_HOLD" for event in decision_events)
        return {
            "samples_tracked": len(events),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "low_confidence_count": sum(
                event.top_score < self.low_confidence_threshold for event in events
            ),
            "hold_rate": round(hold_count / len(decision_events), 4) if decision_events else 0.0,
            "mean_centroid_similarity": round(mean_similarity, 4) if mean_similarity is not None else None,
            "drift_alert": bool(mean_similarity is not None and mean_similarity < self.drift_similarity_floor),
        }
