"""Improved MLP classifier head for prompt injection detection.

Architecture (with USE_LEXICAL_FEATURES=True):
    Input (384 + 25 = 409-dim: MiniLM embedding + lexical features)
      → Linear(409, 512) → LayerNorm → GELU → Dropout(0.2)
      → Linear(512, 384) → LayerNorm → GELU → Dropout(0.2)  [+ residual proj]
      → Linear(384, 256) → LayerNorm → GELU → Dropout(0.2)  [+ residual proj]
      → Linear(256, 128) → LayerNorm → GELU → Dropout(0.2)
      → Linear(128, 2) (logits)

Key improvements over previous version:
  - LayerNorm instead of BatchNorm: stable for variable sequence lengths,
    no issues with small batches or batch statistics drift at inference.
  - Residual connections: prevent information collapse through depth.
  - Deeper network [512, 384, 256, 128] to handle 25-dim lexical features.
  - Kaiming init tuned for GELU (uses 'relu' mode as close approximation).
  - TemperatureScaler: post-hoc calibration via a single learned temperature T.
  - predict_with_uncertainty(): 3-way output (benign / injection / uncertain)
    using a calibrated confidence threshold.
"""
import torch
import torch.nn as nn

from prompt_injection.config import (
    MLP_INPUT_DIM, HIDDEN_DIMS, DROPOUT, NUM_CLASSES,
    CONFIDENCE_THRESHOLD, UNCERTAINTY_LOW,
)


class ResidualBlock(nn.Module):
    """A single LayerNorm + GELU + Dropout block with optional residual shortcut."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)

        # Residual projection if dims differ
        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class MLPClassifier(nn.Module):
    """Feed-forward MLP binary classifier over sentence embeddings + lexical features."""

    def __init__(
        self,
        input_dim: int = MLP_INPUT_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Build residual blocks
        blocks: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout))
            prev_dim = h_dim

        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Linear(prev_dim, num_classes)

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
            x: float tensor of shape (batch, input_dim) — concatenated
               encoder embedding + lexical features.
        Returns:
            logits of shape (batch, num_classes).
        """
        return self.head(self.blocks(x))

    # ── helpers ──────────────────────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (batch, num_classes)."""
        with torch.no_grad():
            return torch.softmax(self(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices (batch,)."""
        return self.predict_proba(x).argmax(dim=-1)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        threshold: float = CONFIDENCE_THRESHOLD,
        uncertain_low: float = UNCERTAINTY_LOW,
    ) -> list[dict]:
        """
        Calibrated 3-way prediction per sample.

        Returns a list of dicts with keys:
            label       : int  — 0=benign, 1=injection, 2=uncertain
            label_name  : str  — 'benign' | 'injection' | 'uncertain'
            injection_prob : float
            benign_prob    : float
        """
        probs = self.predict_proba(x)  # (batch, 2)
        results = []
        for p in probs.cpu().tolist():
            benign_p, inj_p = p[0], p[1]
            if inj_p >= threshold:
                label, name = 1, "injection"
            elif inj_p <= uncertain_low:
                label, name = 0, "benign"
            else:
                label, name = 2, "uncertain"
            results.append({
                "label": label,
                "label_name": name,
                "injection_prob": round(inj_p, 4),
                "benign_prob": round(benign_p, 4),
            })
        return results


class TemperatureScaler(nn.Module):
    """Post-hoc calibration wrapper using temperature scaling (Guo et al., 2017).

    Learns a single scalar temperature T on the validation set after all
    training phases. Calibrated softmax:
        p_calibrated = softmax(logits / T)

    Usage:
        scaler = TemperatureScaler(model)
        scaler.calibrate(val_loader, device)   # fits T
        probs = scaler.predict_proba(x)        # calibrated probs
    """

    def __init__(self, model: MLPClassifier) -> None:
        super().__init__()
        self.model = model
        # Initialize temperature to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits."""
        logits = self.model(x)
        return logits / self.temperature.clamp(min=0.1)  # clamp to avoid div-by-zero

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities (batch, num_classes)."""
        with torch.no_grad():
            return torch.softmax(self(x), dim=-1)

    def calibrate(
        self,
        val_loader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature T by minimizing NLL on the validation set.

        Returns the fitted temperature value.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        # Collect all logits and labels first (no need to recompute each step)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                emb = emb.to(device)
                all_logits.append(self.model(emb).to(device))
                all_labels.append(lbl.to(device))
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def _eval_closure():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature.clamp(min=0.1)
            loss = nll_criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(_eval_closure)

        fitted_T = float(self.temperature.item())
        print(f"  Temperature calibration: T={fitted_T:.4f}")
        return fitted_T
