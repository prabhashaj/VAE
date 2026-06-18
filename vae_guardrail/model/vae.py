"""Sentence-level Variational Autoencoder for anomaly detection.

The encoder uses a frozen DistilBERT to produce [CLS] embeddings, then maps
them through an MLP to a latent distribution (μ, log σ²).  The decoder
reconstructs the [CLS] embedding from a sampled latent vector.

High reconstruction error on unseen prompts signals anomalous (potentially
injected) content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class SentenceEncoder(nn.Module):
    """Frozen transformer → MLP → (μ, log σ²)."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # Freeze transformer weights — we only train the projection head
        for param in self.transformer.parameters():
            param.requires_grad = False

        transformer_dim: int = self.transformer.config.hidden_size  # 768 for distilbert

        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cls_embedding, mu, logvar)."""
        with torch.no_grad():
            transformer_out = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
        # [CLS] is the first token
        cls_embed = transformer_out.last_hidden_state[:, 0, :]  # (B, 768)

        h = self.projection(cls_embed)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return cls_embed, mu, logvar


class SentenceDecoder(nn.Module):
    """Latent z → reconstructed [CLS] embedding."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class SentenceVAE(nn.Module):
    """Full VAE: encode → reparameterize → decode."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 512,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = SentenceEncoder(model_name, hidden_dim, latent_dim)
        output_dim = self.encoder.transformer.config.hidden_size
        self.decoder = SentenceDecoder(latent_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ  (reparameterization trick)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z, mu, logvar)."""
        cls_embed, mu, logvar = self.encoder(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct [CLS] embedding from latent z."""
        return self.decoder(z)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        x_recon : reconstructed [CLS] embedding
        cls_embed : original [CLS] embedding (target)
        mu : latent mean
        logvar : latent log-variance
        """
        cls_embed, mu, logvar = self.encoder(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, cls_embed, mu, logvar


def vae_loss(
    x_recon: torch.Tensor,
    x_orig: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute β-VAE loss = MSE reconstruction + β · KL divergence.

    Returns (total_loss, recon_loss, kld_loss).
    """
    recon_loss = nn.functional.mse_loss(x_recon, x_orig, reduction="mean")
    # KLD = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld_loss
    return total, recon_loss, kld_loss
