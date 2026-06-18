"""Tests for the synthetic data generator."""

import json
from pathlib import Path

import pytest

from vae_guardrail.training.generate_data import build_dataset


@pytest.fixture
def data_dir(tmp_path):
    """Generate data to a temp directory."""
    build_dataset(
        output_dir=tmp_path,
        n_benign=100,
        n_attacks=50,
        train_ratio=0.8,
        seed=42,
    )
    return tmp_path


class TestDataGeneration:

    def test_files_created(self, data_dir: Path):
        assert (data_dir / "train_benign.jsonl").exists()
        assert (data_dir / "eval_benign.jsonl").exists()
        assert (data_dir / "eval_attacks.jsonl").exists()

    def test_train_split_size(self, data_dir: Path):
        with open(data_dir / "train_benign.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 80  # 80% of 100

    def test_eval_benign_split_size(self, data_dir: Path):
        with open(data_dir / "eval_benign.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 20  # 20% of 100

    def test_attack_count(self, data_dir: Path):
        with open(data_dir / "eval_attacks.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Integer division can produce slightly fewer; just check it's close
        assert 45 <= len(lines) <= 50

    def test_benign_labels(self, data_dir: Path):
        with open(data_dir / "train_benign.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                assert rec["label"] == "benign"
                assert "text" in rec
                assert len(rec["text"]) > 0

    def test_attack_labels(self, data_dir: Path):
        with open(data_dir / "eval_attacks.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                assert rec["label"] == "attack"
                assert "category" in rec

    def test_reproducible(self, tmp_path: Path):
        """Same seed should produce identical data."""
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        build_dataset(output_dir=d1, n_benign=50, n_attacks=20, seed=99)
        build_dataset(output_dir=d2, n_benign=50, n_attacks=20, seed=99)

        for fname in ["train_benign.jsonl", "eval_benign.jsonl", "eval_attacks.jsonl"]:
            assert (d1 / fname).read_text() == (d2 / fname).read_text()
