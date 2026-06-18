"""FastAPI server for the VAE Guardrail.

Endpoints:
    POST /v1/validate  — validate a prompt through the cascade
    GET  /v1/health    — health check
    GET  /v1/metrics   — Prometheus-compatible metrics
    POST /v1/config    — runtime threshold hot-reloading
    GET  /dashboard    — monitoring dashboard (HTML)

Usage::

    python -m vae_guardrail.api.server
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from vae_guardrail.api.schemas import (
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    HealthResponse,
    MetricsResponse,
    StageVerdictSchema,
    ValidateRequest,
    ValidateResponse,
)
from vae_guardrail.config import get_settings
from vae_guardrail.filters.cascade import CascadeResult, FilterCascade, Verdict
from vae_guardrail.scoring.scorer import AnomalyScorer

logger = logging.getLogger(__name__)

# ── Global state ────────────────────────────────────────────────────
_cascade: FilterCascade | None = None
_metrics = {
    "total_requests": 0,
    "total_blocks": 0,
    "total_passes": 0,
    "total_shadow_blocks": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and cascade on startup."""
    global _cascade

    settings = get_settings()
    logger.info("Loading VAE model from %s ...", settings.checkpoint_path)

    if not settings.checkpoint_path.exists():
        logger.error(
            "Checkpoint not found at %s. Train the model first: "
            "python -m vae_guardrail.training.train",
            settings.checkpoint_path,
        )
        raise FileNotFoundError(f"Checkpoint not found: {settings.checkpoint_path}")

    if not settings.calibration_path.exists():
        logger.error(
            "Calibration not found at %s. Run calibration first: "
            "python -m vae_guardrail.scoring.calibration",
            settings.calibration_path,
        )
        raise FileNotFoundError(f"Calibration not found: {settings.calibration_path}")

    scorer = AnomalyScorer(settings=settings)
    _cascade = FilterCascade(scorer=scorer, settings=settings)

    logger.info("[OK] VAE Guardrail ready -- shadow_mode=%s", settings.shadow_mode)
    yield
    logger.info("Shutting down VAE Guardrail.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="VAE Guardrail",
        description="Language VAE Prompt Injection Guardrail API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── POST /v1/validate ───────────────────────────────────────────
    @app.post("/v1/validate", response_model=ValidateResponse)
    async def validate(request: ValidateRequest) -> ValidateResponse:
        if _cascade is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result: CascadeResult = _cascade.validate(request.text)

        # Update metrics
        _metrics["total_requests"] += 1
        _metrics["total_latency_ms"] += result.total_latency_ms

        if result.verdict == Verdict.PASS:
            _metrics["total_passes"] += 1
        elif result.verdict == Verdict.BLOCK:
            _metrics["total_blocks"] += 1
        elif result.verdict == Verdict.SHADOW_BLOCK:
            _metrics["total_shadow_blocks"] += 1

        return ValidateResponse(
            verdict=result.verdict.value,
            blocked_by=result.blocked_by,
            total_latency_ms=round(result.total_latency_ms, 2),
            stages=[
                StageVerdictSchema(
                    name=s.name,
                    passed=s.passed,
                    latency_ms=round(s.latency_ms, 2),
                    details=s.details,
                )
                for s in result.stages
            ],
        )

    # ── GET /v1/health ──────────────────────────────────────────────
    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok" if _cascade is not None else "not_ready",
            model_loaded=_cascade is not None,
            shadow_mode=settings.shadow_mode,
            device=settings.resolve_device(),
        )

    # ── GET /v1/metrics ─────────────────────────────────────────────
    @app.get("/v1/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        total = _metrics["total_requests"]
        return MetricsResponse(
            total_requests=total,
            total_blocks=_metrics["total_blocks"],
            total_passes=_metrics["total_passes"],
            total_shadow_blocks=_metrics["total_shadow_blocks"],
            avg_latency_ms=round(
                _metrics["total_latency_ms"] / max(total, 1), 2
            ),
        )

    # ── POST /v1/config ─────────────────────────────────────────────
    @app.post("/v1/config", response_model=ConfigUpdateResponse)
    async def update_config(request: ConfigUpdateRequest) -> ConfigUpdateResponse:
        if _cascade is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        settings = get_settings()
        updated = {}

        if request.vae_anomaly_threshold is not None:
            _cascade.scorer.threshold = request.vae_anomaly_threshold
            updated["vae_anomaly_threshold"] = request.vae_anomaly_threshold

        if request.vector_similarity_threshold is not None:
            _cascade.vector_guard.threshold = request.vector_similarity_threshold
            updated["vector_similarity_threshold"] = request.vector_similarity_threshold

        if request.shadow_mode is not None:
            _cascade.shadow_mode = request.shadow_mode
            updated["shadow_mode"] = request.shadow_mode

        return ConfigUpdateResponse(
            message=f"Updated: {list(updated.keys())}" if updated else "No changes",
            current={
                "vae_anomaly_threshold": _cascade.scorer.threshold,
                "vector_similarity_threshold": _cascade.vector_guard.threshold,
                "shadow_mode": _cascade.shadow_mode,
            },
        )

    # ── GET /dashboard ──────────────────────────────────────────────
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        settings = get_settings()
        total = _metrics["total_requests"]
        avg_lat = _metrics["total_latency_ms"] / max(total, 1)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VAE Guardrail Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s ease;
        }}
        .card:hover {{ transform: translateY(-4px); }}
        .card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }}
        .card .label {{ font-size: 0.85rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }}
        .pass .value {{ color: #4ade80; }}
        .block .value {{ color: #f87171; }}
        .shadow .value {{ color: #fbbf24; }}
        .latency .value {{ color: #60a5fa; }}
        .status {{
            background: rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .status span {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge-on {{ background: rgba(74,222,128,0.2); color: #4ade80; }}
        .badge-off {{ background: rgba(248,113,113,0.2); color: #f87171; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ VAE Guardrail Dashboard</h1>
        <div class="cards">
            <div class="card pass">
                <div class="label">Passed</div>
                <div class="value">{_metrics['total_passes']}</div>
            </div>
            <div class="card block">
                <div class="label">Blocked</div>
                <div class="value">{_metrics['total_blocks']}</div>
            </div>
            <div class="card shadow">
                <div class="label">Shadow Blocked</div>
                <div class="value">{_metrics['total_shadow_blocks']}</div>
            </div>
            <div class="card latency">
                <div class="label">Avg Latency</div>
                <div class="value">{avg_lat:.1f}ms</div>
            </div>
        </div>
        <div class="status">
            <p><strong>Total Requests:</strong> {total}</p>
            <p><strong>Device:</strong> {settings.resolve_device()}</p>
            <p><strong>Shadow Mode:</strong>
                <span class="{'badge-on' if settings.shadow_mode else 'badge-off'}">
                    {'ON' if settings.shadow_mode else 'OFF'}
                </span>
            </p>
            <p><strong>Model:</strong> {settings.model_name}</p>
        </div>
    </div>
    <script>setTimeout(() => location.reload(), 10000);</script>
</body>
</html>"""

    return app


def main() -> None:
    """Entry point for ``python -m vae_guardrail.api.server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = get_settings()
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )


if __name__ == "__main__":
    main()
