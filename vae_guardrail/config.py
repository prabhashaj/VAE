"""Centralized configuration via pydantic-settings.

Loads values from .env / environment variables with sensible defaults.
Use ``get_settings()`` for a cached singleton.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Model & Checkpoints ─────────────────────────────────────────────
    model_name: str = "distilbert-base-uncased"
    checkpoint_path: Path = Path("./checkpoints/vae_guardrail_best.pt")
    calibration_path: Path = Path("./checkpoints/calibration_stats.json")

    # ── VAE Architecture ────────────────────────────────────────────────
    latent_dim: int = 128
    hidden_dim: int = 512
    encoder_hidden: int = 768  # must match transformer hidden size

    # ── Device ──────────────────────────────────────────────────────────
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # ── Anomaly Thresholds ──────────────────────────────────────────────
    vae_anomaly_threshold: float = 0.64
    vector_similarity_threshold: float = 0.95
    max_prompt_length: int = 2048

    # ── Operational Mode ────────────────────────────────────────────────
    shadow_mode: bool = True
    log_level: str = "INFO"

    # ── API Server ──────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1

    # ── Data Paths ──────────────────────────────────────────────────────
    data_dir: Path = Path("./data")

    # ── Helpers ─────────────────────────────────────────────────────────
    def resolve_device(self) -> str:
        """Return the concrete torch device string."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
