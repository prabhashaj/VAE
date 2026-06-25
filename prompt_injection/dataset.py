"""Dataset class and embedding helpers for the prompt injection classifier.

Each JSONL line has the schema: {"text": "...", "label": 0|1}
  label 0 → benign
  label 1 → prompt injection
"""
import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

from prompt_injection.config import EMBEDDING_MODEL


# ─── Embedding helper ─────────────────────────────────────────────────────────

def load_encoder(model_name: str = EMBEDDING_MODEL, device: str = "cpu") -> SentenceTransformer:
    """Load the sentence-transformer encoder (cached locally after first use)."""
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: list[str],
    encoder: SentenceTransformer,
    batch_size: int = 128,
    show_progress: bool = False,
) -> torch.Tensor:
    """Encode a list of strings → float32 tensor of shape (N, embedding_dim)."""
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalize; improves linear separability
    )
    return embeddings.float()


# ─── JSONL I/O ────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> tuple[list[str], list[int]]:
    """Return (texts, labels) lists from a JSONL file."""
    texts, labels = [], []
    for record in iter_jsonl(path):
        texts.append(record["text"])
        labels.append(int(record["label"]))
    return texts, labels


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """Pre-computed embedding dataset.  Pass ``encoder`` to compute on the fly,
    or pass ``embeddings`` tensor directly (preferred for training speed)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        encoder: SentenceTransformer | None = None,
        embeddings: torch.Tensor | None = None,
        batch_size: int = 128,
    ) -> None:
        if embeddings is not None:
            self.embeddings = embeddings
        elif encoder is not None:
            self.embeddings = encode_texts(
                texts, encoder, batch_size=batch_size, show_progress=True
            )
        else:
            raise ValueError("Either encoder or pre-computed embeddings must be supplied.")

        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.embeddings) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
