"""Filter Cascade — orchestrates the 3-stage defense-in-depth pipeline.

Stage 1: Structural Filter (regex/heuristic) — < 5ms
Stage 2: Language VAE Anomaly Scorer       — 15-35ms
Stage 3: Vector Guardrail (cosine sim)     — < 10ms

Short-circuits on block at any stage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from vae_guardrail.config import Settings, get_settings
from vae_guardrail.filters.structural import StructuralFilter, StructuralResult
from vae_guardrail.filters.vector_guard import VectorGuard, VectorResult
from vae_guardrail.scoring.scorer import AnomalyScorer, AnomalyResult

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    SHADOW_BLOCK = "shadow_block"  # would block, but shadow mode is on


@dataclass
class StageResult:
    """Individual stage result with timing."""

    name: str
    passed: bool
    latency_ms: float
    details: dict = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Final result of the full cascade."""

    verdict: Verdict
    stages: list[StageResult]
    total_latency_ms: float
    blocked_by: str | None = None


class FilterCascade:
    """Orchestrate the 3-stage defense-in-depth filter cascade.

    Parameters
    ----------
    scorer : AnomalyScorer
        Trained VAE anomaly scorer (Stage 2).
    settings : Settings, optional
        Application settings.
    """

    def __init__(
        self,
        scorer: AnomalyScorer,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Stage 1: Structural
        self.structural = StructuralFilter(
            max_length=self.settings.max_prompt_length,
        )

        # Stage 2: VAE scorer (passed in)
        self.scorer = scorer

        # Stage 3: Vector guard
        self.vector_guard = VectorGuard(
            threshold=self.settings.vector_similarity_threshold,
        )
        # Initialize vector guard with scorer's embedding function
        self.vector_guard.initialize(scorer.get_embedding)

        self.shadow_mode = self.settings.shadow_mode

    def _make_verdict(self, would_block: bool) -> Verdict:
        if not would_block:
            return Verdict.PASS
        return Verdict.SHADOW_BLOCK if self.shadow_mode else Verdict.BLOCK

    def validate(self, text: str) -> CascadeResult:
        """Run a prompt through all three stages.

        Returns early if any stage blocks.
        """
        t_start = time.perf_counter()
        stages: list[StageResult] = []

        # ── Stage 1: Structural Filter ──────────────────────────────
        t0 = time.perf_counter()
        structural_result: StructuralResult = self.structural.check(text)
        stage1_ms = (time.perf_counter() - t0) * 1000

        stages.append(StageResult(
            name="structural",
            passed=structural_result.passed,
            latency_ms=stage1_ms,
            details={
                "score": structural_result.score,
                "matches": [
                    {"name": m.pattern_name, "severity": m.severity}
                    for m in structural_result.matches
                ],
            },
        ))

        if not structural_result.passed:
            total = (time.perf_counter() - t_start) * 1000
            return CascadeResult(
                verdict=self._make_verdict(would_block=True),
                stages=stages,
                total_latency_ms=total,
                blocked_by="structural",
            )

        # ── Stage 2: VAE Anomaly Scorer ─────────────────────────────
        t0 = time.perf_counter()
        anomaly_result: AnomalyResult = self.scorer.score(text)
        stage2_ms = (time.perf_counter() - t0) * 1000

        stages.append(StageResult(
            name="vae_anomaly",
            passed=not anomaly_result.is_anomaly,
            latency_ms=stage2_ms,
            details={
                "reconstruction_loss": anomaly_result.reconstruction_loss,
                "mahalanobis_distance": anomaly_result.mahalanobis_distance,
                "combined_score": anomaly_result.combined_score,
                "threshold": anomaly_result.threshold,
            },
        ))

        if anomaly_result.is_anomaly:
            total = (time.perf_counter() - t_start) * 1000
            return CascadeResult(
                verdict=self._make_verdict(would_block=True),
                stages=stages,
                total_latency_ms=total,
                blocked_by="vae_anomaly",
            )

        # ── Stage 3: Vector Guardrail ───────────────────────────────
        t0 = time.perf_counter()
        embedding = self.scorer.get_embedding(text)
        vector_result: VectorResult = self.vector_guard.check(embedding)
        stage3_ms = (time.perf_counter() - t0) * 1000

        stages.append(StageResult(
            name="vector_guard",
            passed=vector_result.passed,
            latency_ms=stage3_ms,
            details={
                "max_similarity": vector_result.max_similarity,
                "closest_attack": vector_result.closest_attack,
                "threshold": vector_result.threshold,
            },
        ))

        if not vector_result.passed:
            total = (time.perf_counter() - t_start) * 1000
            return CascadeResult(
                verdict=self._make_verdict(would_block=True),
                stages=stages,
                total_latency_ms=total,
                blocked_by="vector_guard",
            )

        # ── All passed ──────────────────────────────────────────────
        total = (time.perf_counter() - t_start) * 1000
        return CascadeResult(
            verdict=Verdict.PASS,
            stages=stages,
            total_latency_ms=total,
            blocked_by=None,
        )
