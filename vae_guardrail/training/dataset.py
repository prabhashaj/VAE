"""PyTorch Dataset for loading JSONL prompt data."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PromptDataset(Dataset):
    """Load prompts from a JSONL file and tokenize them.

    Each line in the JSONL file must have at least a ``"text"`` field.

    Parameters
    ----------
    path : str or Path
        Path to the JSONL file.
    model_name : str
        HuggingFace tokenizer identifier.
    max_length : int
        Maximum token length (prompts are truncated/padded to this).
    """

    def __init__(
        self,
        path: str | Path,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
    ) -> None:
        self.path = Path(path)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load all texts
        self.records: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Pre-tokenize for speed
        texts = [r["text"] for r in self.records]
        self._encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
        }

    @property
    def texts(self) -> list[str]:
        """Return the raw text of all samples."""
        return [r["text"] for r in self.records]

    @property
    def labels(self) -> list[str]:
        """Return labels (benign/attack) if present."""
        return [r.get("label", "unknown") for r in self.records]
