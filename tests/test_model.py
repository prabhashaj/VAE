"""Unit tests for the SentenceVAE model."""

import pytest
import torch

from vae_guardrail.model.vae import SentenceVAE, vae_loss


@pytest.fixture(scope="module")
def model():
    """Build a small VAE for testing (uses real DistilBERT, so a bit slow)."""
    return SentenceVAE(
        model_name="distilbert-base-uncased",
        hidden_dim=64,
        latent_dim=32,
    )


@pytest.fixture(scope="module")
def dummy_input():
    """Dummy tokenized input (batch_size=2, seq_len=16)."""
    return {
        "input_ids": torch.randint(0, 30522, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
    }


class TestSentenceVAE:

    def test_forward_shapes(self, model: SentenceVAE, dummy_input: dict):
        x_recon, cls_embed, mu, logvar = model(**dummy_input)
        B = 2
        assert x_recon.shape == (B, 768), f"x_recon shape: {x_recon.shape}"
        assert cls_embed.shape == (B, 768), f"cls_embed shape: {cls_embed.shape}"
        assert mu.shape == (B, 32), f"mu shape: {mu.shape}"
        assert logvar.shape == (B, 32), f"logvar shape: {logvar.shape}"

    def test_encode_shapes(self, model: SentenceVAE, dummy_input: dict):
        z, mu, logvar = model.encode(**dummy_input)
        assert z.shape == (2, 32)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)

    def test_decode_shapes(self, model: SentenceVAE):
        z = torch.randn(2, 32)
        out = model.decode(z)
        assert out.shape == (2, 768)

    def test_reparameterize_stochastic(self, model: SentenceVAE):
        mu = torch.zeros(5, 32)
        logvar = torch.zeros(5, 32)
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        # With non-zero std, samples should differ
        assert not torch.allclose(z1, z2), "Reparameterize should be stochastic"

    def test_frozen_transformer(self, model: SentenceVAE):
        for param in model.encoder.transformer.parameters():
            assert not param.requires_grad, "Transformer params should be frozen"

    def test_trainable_params_exist(self, model: SentenceVAE):
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0, "Model should have trainable parameters"


class TestVAELoss:

    def test_loss_returns_three_tensors(self):
        x_recon = torch.randn(4, 768)
        x_orig = torch.randn(4, 768)
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        total, recon, kld = vae_loss(x_recon, x_orig, mu, logvar)
        assert total.dim() == 0, "Total loss should be scalar"
        assert recon.dim() == 0, "Recon loss should be scalar"
        assert kld.dim() == 0, "KLD loss should be scalar"

    def test_loss_non_negative(self):
        x = torch.randn(4, 768)
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        total, recon, kld = vae_loss(x, x, mu, logvar)
        assert recon.item() >= 0
        assert kld.item() >= 0

    def test_perfect_reconstruction(self):
        x = torch.randn(4, 768)
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        total, recon, kld = vae_loss(x, x, mu, logvar, beta=0.0)
        assert recon.item() < 1e-6, "Perfect reconstruction should have ~0 recon loss"

    def test_beta_scaling(self):
        x_recon = torch.randn(4, 768)
        x_orig = torch.randn(4, 768)
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)

        total_b0, _, _ = vae_loss(x_recon, x_orig, mu, logvar, beta=0.0)
        total_b1, _, _ = vae_loss(x_recon, x_orig, mu, logvar, beta=1.0)

        # With beta=0, total == recon_only; with beta=1, total >= recon_only
        assert total_b1.item() >= total_b0.item() - 1e-6
