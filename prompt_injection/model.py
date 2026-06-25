"""MLP classifier that sits on top of frozen sentence-transformer embeddings.

Architecture:
    Input (384-dim all-MiniLM embedding)
      → Linear → BatchNorm → GELU → Dropout
      → Linear → BatchNorm → GELU → Dropout
      → Linear (2-class logits)
"""
import torch
import torch.nn as nn
from prompt_injection.config import EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT, NUM_CLASSES


class MLPClassifier(nn.Module):
    """Feed-forward MLP binary classifier over sentence embeddings."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch, input_dim) — pre-computed embeddings.
        Returns:
            logits of shape (batch, num_classes).
        """
        return self.network(x)

    # ── helpers ──────────────────────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (batch, num_classes)."""
        with torch.no_grad():
            return torch.softmax(self(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices (batch,)."""
        return self.predict_proba(x).argmax(dim=-1)
