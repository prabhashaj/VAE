"""Standalone evaluation script.

Computes precision, recall, F1, accuracy, and a confusion matrix on any JSONL file.

Run:
    python -m prompt_injection.evaluate                          # uses default eval.jsonl
    python -m prompt_injection.evaluate --data data/eval.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from prompt_injection.config import EVAL_FILE, MODEL_PATH, EMBEDDING_MODEL, BATCH_SIZE
from prompt_injection.dataset import load_jsonl, load_encoder, EmbeddingDataset
from prompt_injection.model import MLPClassifier
from torch.utils.data import DataLoader


def evaluate(
    data_file: Path = EVAL_FILE,
    model_path: Path = MODEL_PATH,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}. Train first.")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}.")

    print(f"Loading checkpoint: {model_path}")
    model = MLPClassifier().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Trained at epoch {ckpt['epoch']} — val F1 at save: {ckpt['val_f1']:.4f}\n")

    print(f"Loading data: {data_file}")
    texts, labels = load_jsonl(data_file)
    print(f"  {len(texts):,} samples ({labels.count(0):,} benign / {labels.count(1):,} injection)\n")

    encoder = load_encoder(EMBEDDING_MODEL, device=str(device))
    dataset = EmbeddingDataset(texts, labels, encoder=encoder)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for embeddings, batch_labels in loader:
            embeddings = embeddings.to(device)
            probs = model.predict_proba(embeddings)        # (B, 2)
            preds = probs.argmax(dim=-1)
            all_probs.extend(probs[:, 1].cpu().tolist())  # injection probability
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(
        all_labels, all_preds,
        target_names=["benign", "injection"],
        digits=4,
    ))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>10}  {'benign':>8}  {'injection':>10}")
    print(f"  {'benign':>10}  {cm[0, 0]:>8}  {cm[0, 1]:>10}")
    print(f"  {'injection':>10}  {cm[1, 0]:>8}  {cm[1, 1]:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prompt injection classifier")
    parser.add_argument("--data",  type=Path, default=EVAL_FILE,  help="JSONL eval file")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Checkpoint path")
    args = parser.parse_args()
    evaluate(args.data, args.model)
