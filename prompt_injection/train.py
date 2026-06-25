"""Training loop for the prompt injection MLP classifier.

Pipeline:
  1. Load/generate JSONL data
  2. Encode all texts with all-MiniLM-L6-v2  (done once, kept in RAM)
  3. Train MLP on embeddings with early stopping
  4. Save best checkpoint

Run:
    python -m prompt_injection.train
"""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from prompt_injection.config import (
    TRAIN_FILE, EVAL_FILE, MODEL_PATH, CHECKPOINT_DIR,
    EMBEDDING_MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, EARLY_STOP_PAT, SEED,
)
from prompt_injection.dataset import load_jsonl, load_encoder, EmbeddingDataset
from prompt_injection.model import MLPClassifier


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: MLPClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(
    train_file: Path = TRAIN_FILE,
    eval_file: Path = EVAL_FILE,
    model_path: Path = MODEL_PATH,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Validate data files ──────────────────────────────────────────────────
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_file}. "
            "Run: python -m prompt_injection.generate_data"
        )
    if not eval_file.exists():
        raise FileNotFoundError(
            f"Eval data not found at {eval_file}. "
            "Run: python -m prompt_injection.generate_data"
        )

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_texts, train_labels = load_jsonl(train_file)
    eval_texts, eval_labels   = load_jsonl(eval_file)
    print(f"  Train: {len(train_texts):,} samples  "
          f"({train_labels.count(0):,} benign / {train_labels.count(1):,} injection)")
    print(f"  Eval : {len(eval_texts):,} samples  "
          f"({eval_labels.count(0):,} benign / {eval_labels.count(1):,} injection)")

    # ── Encode texts ─────────────────────────────────────────────────────────
    print(f"\nEncoding texts with '{EMBEDDING_MODEL}'...")
    encoder = load_encoder(EMBEDDING_MODEL, device=str(device))
    t0 = time.time()
    train_dataset = EmbeddingDataset(train_texts, train_labels, encoder=encoder)
    eval_dataset  = EmbeddingDataset(eval_texts,  eval_labels,  encoder=encoder)
    print(f"  Encoding took {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # ── Model, optimizer, loss ───────────────────────────────────────────────
    model = MLPClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Class-weighted loss — handles imbalance from multi-source mixing
    n_benign    = train_labels.count(0)
    n_injection = train_labels.count(1)
    n_total     = n_benign + n_injection
    w_benign    = n_total / (2.0 * n_benign)    if n_benign    > 0 else 1.0
    w_injection = n_total / (2.0 * n_injection) if n_injection > 0 else 1.0
    class_weights = torch.tensor([w_benign, w_injection], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: benign={w_benign:.3f}  injection={w_injection:.3f}")

    # ── Training loop ────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1    = 0.0
    no_improve = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val F1':>8}  {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, eval_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        flag = " *" if val_f1 > best_f1 else ""
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {val_f1:>8.4f}  {lr:>10.2e}{flag}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                },
                model_path,
            )
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PAT} epochs).")
                break

    print(f"\nBest Val F1: {best_f1:.4f}  -- checkpoint saved to {model_path}")

    # ── Final eval on best checkpoint ────────────────────────────────────────
    print("\nFinal evaluation (best checkpoint)...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels_lst = [], []
    with torch.no_grad():
        for embeddings, labels in eval_loader:
            embeddings = embeddings.to(device)
            preds = model(embeddings).argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels_lst.extend(labels.tolist())

    print(classification_report(
        all_labels_lst, all_preds,
        target_names=["benign", "injection"],
        digits=4,
    ))


if __name__ == "__main__":
    train()
