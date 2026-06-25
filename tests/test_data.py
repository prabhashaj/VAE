"""Tests for data generation and the Dataset class."""
import json
import tempfile
from pathlib import Path
import torch
import pytest

from prompt_injection.generate_data import _make_benign, _make_injection, generate
from prompt_injection.dataset import load_jsonl, EmbeddingDataset
from prompt_injection.config import EMBEDDING_DIM


def test_make_benign_returns_string():
    s = _make_benign()
    assert isinstance(s, str) and len(s) > 5


def test_make_injection_returns_string():
    s = _make_injection()
    assert isinstance(s, str) and len(s) > 5


def test_generate_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_f = Path(tmpdir) / "train.jsonl"
        eval_f  = Path(tmpdir) / "eval.jsonl"
        generate(
            train_file=train_f, eval_file=eval_f,
            n_train_benign=10, n_train_injection=10,
            n_eval_benign=5,  n_eval_injection=5,
        )
        assert train_f.exists() and eval_f.exists()
        train_lines = train_f.read_text().strip().splitlines()
        assert len(train_lines) == 20
        for line in train_lines:
            rec = json.loads(line)
            assert "text" in rec and "label" in rec
            assert rec["label"] in (0, 1)


def test_load_jsonl():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"text": f"sample {i}", "label": i % 2}) + "\n")
        path = Path(f.name)
    texts, labels = load_jsonl(path)
    assert len(texts) == 5
    assert all(isinstance(t, str) for t in texts)
    assert all(l in (0, 1) for l in labels)


def test_embedding_dataset_with_precomputed():
    n, dim = 8, EMBEDDING_DIM
    embeddings = torch.randn(n, dim)
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    ds = EmbeddingDataset([""] * n, labels, embeddings=embeddings)
    assert len(ds) == n
    emb, lbl = ds[0]
    assert emb.shape == (dim,)
    assert lbl.item() in (0, 1)
