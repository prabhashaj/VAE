"""Training pipeline for the Sentence VAE.

Features:
- β-annealing: KLD weight linearly ramps 0 → 1 over the first 30 % of training
- Cosine LR schedule with warm-up
- Early stopping on validation loss (patience configurable)
- Checkpoint saving (best + latest)

Usage::

    python -m vae_guardrail.training.train --epochs 10
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from vae_guardrail.config import get_settings
from vae_guardrail.model.vae import SentenceVAE, vae_loss
from vae_guardrail.training.dataset import PromptDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full VAE training loop."""

    def __init__(
        self,
        model: SentenceVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        lr: float = 2e-4,
        epochs: int = 10,
        beta_anneal_ratio: float = 0.3,
        patience: int = 5,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer — only train non-frozen params
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # β-annealing schedule
        total_steps = epochs * len(train_loader)
        self.anneal_steps = int(total_steps * beta_anneal_ratio)

        # Tracking
        self.best_val_loss = float("inf")
        self.no_improve = 0
        self.global_step = 0

    def _get_beta(self) -> float:
        """Linear β annealing: 0 → 1 over ``anneal_steps``."""
        if self.anneal_steps == 0:
            return 1.0
        return min(1.0, self.global_step / self.anneal_steps)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            x_recon, cls_embed, mu, logvar = self.model(input_ids, attention_mask)
            beta = self._get_beta()
            loss, recon, kld = vae_loss(x_recon, cls_embed, mu, logvar, beta=beta)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1
            self.global_step += 1

        return {
            "loss": total_loss / n_batches,
            "recon": total_recon / n_batches,
            "kld": total_kld / n_batches,
            "beta": self._get_beta(),
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            x_recon, cls_embed, mu, logvar = self.model(input_ids, attention_mask)
            loss, recon, kld = vae_loss(x_recon, cls_embed, mu, logvar, beta=1.0)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "recon": total_recon / n_batches,
            "kld": total_kld / n_batches,
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "global_step": self.global_step,
            },
            path,
        )

    def train(self) -> None:
        """Run the full training loop."""
        logger.info("Starting training — %d epochs, device=%s", self.epochs, self.device)
        logger.info(
            "Trainable params: %s",
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}",
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            self.scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                "Epoch %d/%d [%.1fs]  "
                "train_loss=%.4f (recon=%.4f kld=%.4f β=%.3f)  "
                "val_loss=%.4f (recon=%.4f kld=%.4f)",
                epoch,
                self.epochs,
                elapsed,
                train_metrics["loss"],
                train_metrics["recon"],
                train_metrics["kld"],
                train_metrics["beta"],
                val_metrics["loss"],
                val_metrics["recon"],
                val_metrics["kld"],
            )

            # Save latest
            self._save_checkpoint(
                self.checkpoint_dir / "vae_guardrail_latest.pt", epoch, val_metrics["loss"]
            )

            # Best model tracking
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.no_improve = 0
                self._save_checkpoint(
                    self.checkpoint_dir / "vae_guardrail_best.pt", epoch, val_metrics["loss"]
                )
                logger.info("  [*] New best model saved (val_loss=%.4f)", val_metrics["loss"])
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    logger.info("  [STOP] Early stopping -- no improvement for %d epochs", self.patience)
                    break

        logger.info("Training complete. Best val_loss=%.4f", self.best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Sentence VAE")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    device = settings.resolve_device()

    # Check if data exists; if not, generate it
    train_path = data_dir / "train_benign.jsonl"
    eval_path = data_dir / "eval_benign.jsonl"

    if not train_path.exists():
        logger.info("Training data not found — generating synthetic data...")
        from vae_guardrail.training.generate_data import build_dataset
        build_dataset(output_dir=data_dir)

    # Load datasets
    logger.info("Loading datasets...")
    train_ds = PromptDataset(train_path, model_name=settings.model_name, max_length=args.max_length)
    val_ds = PromptDataset(eval_path, model_name=settings.model_name, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # Build model
    model = SentenceVAE(
        model_name=settings.model_name,
        hidden_dim=settings.hidden_dim,
        latent_dim=settings.latent_dim,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
