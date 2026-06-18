"""Pydantic request/response schemas for the API server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidateRequest(BaseModel):
    """Request body for the /v1/validate endpoint."""

    text: str = Field(..., min_length=1, max_length=10000, description="Prompt text to validate")


class StageVerdictSchema(BaseModel):
    """Per-stage verdict detail."""

    name: str
    passed: bool
    latency_ms: float
    details: dict = Field(default_factory=dict)


class ValidateResponse(BaseModel):
    """Response from the /v1/validate endpoint."""

    verdict: str = Field(..., description="pass | block | shadow_block")
    blocked_by: str | None = Field(None, description="Stage that caused the block")
    total_latency_ms: float
    stages: list[StageVerdictSchema]


class HealthResponse(BaseModel):
    """Response from the /v1/health endpoint."""

    status: str = "ok"
    model_loaded: bool = True
    shadow_mode: bool = True
    device: str = "cpu"


class ConfigUpdateRequest(BaseModel):
    """Request body for runtime threshold updates."""

    vae_anomaly_threshold: float | None = None
    vector_similarity_threshold: float | None = None
    shadow_mode: bool | None = None


class ConfigUpdateResponse(BaseModel):
    """Response from the /v1/config endpoint."""

    message: str = "Configuration updated"
    current: dict = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Basic metrics for the /v1/metrics endpoint."""

    total_requests: int = 0
    total_blocks: int = 0
    total_passes: int = 0
    total_shadow_blocks: int = 0
    avg_latency_ms: float = 0.0
