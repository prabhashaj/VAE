"""Anomaly scorer using trained VAE reconstruction loss + Mahalanobis distance.

The scorer loads a trained VAE checkpoint and calibration statistics, then
computes a combined anomaly score for each input prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from vae_guardrail.config import Settings, get_settings
from vae_guardrail.model.vae import SentenceVAE

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly scoring on a single prompt."""

    reconstruction_loss: float
    mahalanobis_distance: float
    combined_score: float
    is_anomaly: bool
    threshold: float


class AnomalyScorer:
    """Score prompts for anomalousness using a trained VAE.

    The combined score is a weighted sum of normalized reconstruction loss
    and Mahalanobis distance in the latent space.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Path to the trained VAE checkpoint.
    calibration_path : Path, optional
        Path to calibration statistics JSON.
    settings : Settings, optional
        Application settings (uses singleton if not provided).
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        calibration_path: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.device = self.settings.resolve_device()

        cp = checkpoint_path or self.settings.checkpoint_path
        cal = calibration_path or self.settings.calibration_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)

        # Load model
        self.model = SentenceVAE(
            model_name=self.settings.model_name,
            hidden_dim=self.settings.hidden_dim,
            latent_dim=self.settings.latent_dim,
        )
        checkpoint = torch.load(cp, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load calibration statistics
        with open(cal, "r", encoding="utf-8") as f:
            cal_data = json.load(f)

        self.recon_mean = cal_data["reconstruction_loss"]["mean"]
        self.recon_std = cal_data["reconstruction_loss"]["std"]
        self.latent_mean = torch.tensor(cal_data["latent"]["mean"], dtype=torch.float32).to(
            self.device
        )
        self.latent_cov_inv = torch.tensor(
            cal_data["latent"]["cov_inv"], dtype=torch.float32
        ).to(self.device)

        self.threshold = self.settings.vae_anomaly_threshold

        logger.info(
            "AnomalyScorer loaded — device=%s, threshold=%.3f", self.device, self.threshold
        )

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def score(self, text: str) -> AnomalyResult:
        """Score a single text prompt for anomalousness."""
        tokens = self._tokenize(text)
        x_recon, cls_embed, mu, logvar = self.model(
            tokens["input_ids"], tokens["attention_mask"]
        )

        # Reconstruction loss (per-sample MSE)
        recon_loss = torch.nn.functional.mse_loss(x_recon, cls_embed).item()

        # Mahalanobis distance in latent space
        z, _, _ = self.model.encode(tokens["input_ids"], tokens["attention_mask"])
        diff = z.squeeze(0) - self.latent_mean  # (latent_dim,)
        mahal = torch.sqrt(diff @ self.latent_cov_inv @ diff).item()

        # Normalize reconstruction loss using calibration stats
        recon_z = (recon_loss - self.recon_mean) / max(self.recon_std, 1e-8)

        # Combined score: weighted sum
        combined = 0.6 * recon_z + 0.4 * (mahal / 10.0)  # mahal typically larger scale

        return AnomalyResult(
            reconstruction_loss=recon_loss,
            mahalanobis_distance=mahal,
            combined_score=combined,
            is_anomaly=combined > self.threshold,
            threshold=self.threshold,
        )

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> list[AnomalyResult]:
        """Score a batch of prompts."""
        return [self.score(t) for t in texts]

    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        """Get the [CLS] embedding for a text (used by VectorGuard)."""
        tokens = self._tokenize(text)
        cls_embed, _, _ = self.model.encoder(
            tokens["input_ids"], tokens["attention_mask"]
        )
        return cls_embed.squeeze(0).cpu().numpy()
