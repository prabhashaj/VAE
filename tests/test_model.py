"""Tests for the MLPClassifier."""
import torch
import pytest
from prompt_injection.model import MLPClassifier
from prompt_injection.config import EMBEDDING_DIM, NUM_CLASSES


def test_output_shape():
    model = MLPClassifier()
    x = torch.randn(8, EMBEDDING_DIM)
    logits = model(x)
    assert logits.shape == (8, NUM_CLASSES)


def test_predict_proba_sums_to_one():
    model = MLPClassifier()
    x = torch.randn(4, EMBEDDING_DIM)
    probs = model.predict_proba(x)
    assert probs.shape == (4, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_predict_class_indices():
    model = MLPClassifier()
    x = torch.randn(16, EMBEDDING_DIM)
    preds = model.predict(x)
    assert preds.shape == (16,)
    assert set(preds.tolist()).issubset({0, 1})


def test_single_sample():
    model = MLPClassifier()
    x = torch.randn(1, EMBEDDING_DIM)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_custom_hidden_dims():
    model = MLPClassifier(hidden_dims=[64, 32])
    x = torch.randn(4, EMBEDDING_DIM)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)
