"""Calibration: compute latent statistics from training data and suggest thresholds.

Runs the trained VAE over the training set to collect:
- Reconstruction loss distribution
- Latent space mean and covariance (for Mahalanobis distance)
- Suggested thresholds at various percentiles

Usage::

    python -m vae_guardrail.scoring.calibration \\
        --checkpoint checkpoints/vae_guardrail_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_guardrail.config import get_settings
from vae_guardrail.model.vae import SentenceVAE
from vae_guardrail.training.dataset import PromptDataset

logger = logging.getLogger(__name__)


def calibrate(
    checkpoint_path: str | Path,
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    batch_size: int = 64,
    max_length: int = 128,
) -> dict:
    """Run calibration and return/save statistics.

    Parameters
    ----------
    checkpoint_path
        Path to trained VAE checkpoint.
    data_path
        Path to the benign training JSONL (defaults to settings.data_dir / train_benign.jsonl).
    output_path
        Where to save calibration JSON (defaults to settings.calibration_path).
    """
    settings = get_settings()
    device = settings.resolve_device()

    data_path = Path(data_path) if data_path else settings.data_dir / "train_benign.jsonl"
    output_path = Path(output_path) if output_path else settings.calibration_path

    # Load model
    model = SentenceVAE(
        model_name=settings.model_name,
        hidden_dim=settings.hidden_dim,
        latent_dim=settings.latent_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load data
    dataset = PromptDataset(data_path, model_name=settings.model_name, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Running calibration on %d samples...", len(dataset))

    all_recon_losses: list[float] = []
    all_latents: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            x_recon, cls_embed, mu, logvar = model(input_ids, attention_mask)
            z, _, _ = model.encode(input_ids, attention_mask)

            # Per-sample reconstruction loss
            for i in range(x_recon.size(0)):
                loss = torch.nn.functional.mse_loss(x_recon[i], cls_embed[i]).item()
                all_recon_losses.append(loss)

            all_latents.append(z.cpu().numpy())

    recon_array = np.array(all_recon_losses)
    latent_array = np.concatenate(all_latents, axis=0)  # (N, latent_dim)

    # Reconstruction loss statistics
    recon_stats = {
        "mean": float(recon_array.mean()),
        "std": float(recon_array.std()),
        "min": float(recon_array.min()),
        "max": float(recon_array.max()),
        "p50": float(np.percentile(recon_array, 50)),
        "p90": float(np.percentile(recon_array, 90)),
        "p95": float(np.percentile(recon_array, 95)),
        "p99": float(np.percentile(recon_array, 99)),
    }

    # Latent space statistics
    latent_mean = latent_array.mean(axis=0)  # (latent_dim,)
    latent_cov = np.cov(latent_array, rowvar=False)  # (latent_dim, latent_dim)

    # Regularize covariance for numerical stability
    latent_cov += np.eye(latent_cov.shape[0]) * 1e-6
    latent_cov_inv = np.linalg.inv(latent_cov)

    # Compute Mahalanobis distances for all training samples
    diffs = latent_array - latent_mean  # (N, latent_dim)
    mahal_distances = np.sqrt(np.sum(diffs @ latent_cov_inv * diffs, axis=1))

    mahal_stats = {
        "mean": float(mahal_distances.mean()),
        "std": float(mahal_distances.std()),
        "p50": float(np.percentile(mahal_distances, 50)),
        "p90": float(np.percentile(mahal_distances, 90)),
        "p95": float(np.percentile(mahal_distances, 95)),
        "p99": float(np.percentile(mahal_distances, 99)),
    }

    # Suggested thresholds (combined score)
    recon_z = (recon_array - recon_stats["mean"]) / max(recon_stats["std"], 1e-8)
    combined = 0.6 * recon_z + 0.4 * (mahal_distances / 10.0)

    suggested_thresholds = {
        "p90": float(np.percentile(combined, 90)),
        "p95": float(np.percentile(combined, 95)),
        "p99": float(np.percentile(combined, 99)),
        "recommended": float(np.percentile(combined, 95)),
    }

    # Assemble calibration data
    calibration = {
        "reconstruction_loss": recon_stats,
        "mahalanobis": mahal_stats,
        "latent": {
            "mean": latent_mean.tolist(),
            "cov_inv": latent_cov_inv.tolist(),
        },
        "suggested_thresholds": suggested_thresholds,
        "n_samples": len(dataset),
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    logger.info("[OK] Calibration saved to %s", output_path)
    logger.info("   Reconstruction loss — mean=%.4f, std=%.4f, p95=%.4f",
                recon_stats["mean"], recon_stats["std"], recon_stats["p95"])
    logger.info("   Mahalanobis distance — mean=%.2f, std=%.2f, p95=%.2f",
                mahal_stats["mean"], mahal_stats["std"], mahal_stats["p95"])
    logger.info("   Suggested threshold (p95 combined): %.4f",
                suggested_thresholds["recommended"])

    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate VAE anomaly thresholds")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vae_guardrail_best.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument("--data", type=str, default=None, help="Path to benign JSONL")
    parser.add_argument("--output", type=str, default=None, help="Output calibration JSON path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    calibrate(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
