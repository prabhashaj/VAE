"""Training loop for the prompt injection MLP classifier (v3).

Pipeline:
  1. Load/generate JSONL data
  2. Encode all texts with all-MiniLM-L6-v2  (done once, kept in RAM)
  3. Concatenate handcrafted lexical features (25-dim)
  4. Phase 1: Train MLP with FocalLoss + Mixup + OneCycleLR + label smoothing
  5. Phase 2: Fine-tune on hard negatives (both FP benign and FN injection)
  6. Temperature calibration via LBFGS on val set
  7. Threshold calibration via F1 sweep on val set
  8. Save best checkpoint with calibrated threshold + temperature

Run:
    python -m prompt_injection.train
"""
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from prompt_injection.config import (
    TRAIN_FILE, EVAL_FILE, MODEL_PATH, CHECKPOINT_DIR,
    EMBEDDING_MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, EARLY_STOP_PAT, SEED, LABEL_SMOOTHING,
    PHASE2_EPOCHS, PHASE2_LR, HARD_NEG_UPSAMPLE, HARD_FN_UPSAMPLE,
    CONFIDENCE_THRESHOLD, USE_LEXICAL_FEATURES,
    MIXUP_ALPHA, FOCAL_GAMMA, FOCAL_ALPHA,
)
from prompt_injection.dataset import (
    load_jsonl, load_encoder, EmbeddingDataset, build_dataset_cached
)
from prompt_injection.model import MLPClassifier, TemperatureScaler


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for class-imbalanced training.

    FL(p) = −α · (1 − pₜ)^γ · log(pₜ)

    Down-weights easy examples and focuses gradient on hard ones.
    With γ=2, α=0.25 this is the standard RetinaNet configuration.
    Supports optional class weights and label smoothing.
    """

    def __init__(
        self,
        gamma: float = FOCAL_GAMMA,
        alpha: float = FOCAL_ALPHA,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.weight          = weight    # class weight tensor (num_classes,)
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with label smoothing gives per-sample loss
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Focal modulation: (1 - p_t)^gamma
        probs = torch.softmax(logits, dim=-1)
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup (Zhang et al., 2018) to a batch of embeddings.

    Interpolates pairs of samples in embedding space:
        x̃ = λ·xᵢ + (1−λ)·xⱼ
        ỹ = λ·yᵢ + (1−λ)·yⱼ,  λ ~ Beta(α, α)

    Returns mixed embeddings and soft (one-hot) labels.
    """
    if alpha <= 0.0:
        # No mixup — return one-hot labels
        soft_labels = F.one_hot(labels, num_classes=num_classes).float()
        return embeddings, soft_labels

    lam = float(np.random.beta(alpha, alpha))
    batch_size = embeddings.size(0)
    idx = torch.randperm(batch_size, device=embeddings.device)

    mixed_emb = lam * embeddings + (1.0 - lam) * embeddings[idx]

    # Convert labels to one-hot and mix
    y_a = F.one_hot(labels, num_classes=num_classes).float()
    y_b = F.one_hot(labels[idx], num_classes=num_classes).float()
    mixed_labels = lam * y_a + (1.0 - lam) * y_b

    return mixed_emb, mixed_labels


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(
    model: MLPClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,       # if provided, stepped once per batch (e.g. OneCycleLR)
    use_mixup: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for embeddings, labels in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        if use_mixup and MIXUP_ALPHA > 0:
            mixed_emb, mixed_labels = mixup_batch(embeddings, labels)
            optimizer.zero_grad()
            logits = model(mixed_emb)
            # Mixed CE: sum of CE against each mixed label component
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
        else:
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()          # OneCycleLR: step per batch
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: MLPClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Returns (avg_loss, macro_f1, all_preds, all_labels)."""
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
    return avg_loss, macro_f1, all_preds, all_labels


# ─── Threshold calibration ────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Sweep injection probability thresholds [0.35, 0.80] and return the
    threshold that maximises macro-F1 on the validation set.
    """
    model.eval()
    all_probs, all_labels = [], []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        probs = model.predict_proba(embeddings)  # (B, 2)
        all_probs.extend(probs[:, 1].cpu().tolist())
        all_labels.extend(labels.tolist())

    best_thresh = 0.50
    best_f1     = 0.0
    for thresh in np.arange(0.35, 0.85, 0.01):
        preds = [1 if p >= thresh else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = float(thresh)

    print(f"  Calibrated threshold: {best_thresh:.2f}  (val macro-F1={best_f1:.4f})")
    return best_thresh


# ─── Hard negative mining ─────────────────────────────────────────────────────

@torch.no_grad()
def mine_hard_negatives(
    model: MLPClassifier,
    eval_dataset: EmbeddingDataset,
    device: torch.device,
    threshold: float = CONFIDENCE_THRESHOLD,
    fp_upsample: int = HARD_NEG_UPSAMPLE,
    fn_upsample: int = HARD_FN_UPSAMPLE,
) -> EmbeddingDataset | None:
    """
    Find hard examples the model gets wrong and return an upsampled dataset
    for Phase-2 fine-tuning.

    Mines two categories:
      - False Positives (FP): benign samples predicted as injection → upsampled ×fp_upsample
      - False Negatives (FN): injection samples predicted as benign → upsampled ×fn_upsample
        (higher factor: a missed attack is more dangerous than a false alarm)

    Returns None if no hard examples are found.
    """
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    fp_embs:   list[torch.Tensor] = []  # benign → wrongly injection
    fp_labels: list[int]          = []
    fn_embs:   list[torch.Tensor] = []  # injection → wrongly benign
    fn_labels: list[int]          = []

    offset = 0
    for embs, lbls in loader:
        embs_dev = embs.to(device)
        probs = model.predict_proba(embs_dev)[:, 1].cpu()

        for i, (prob, lbl) in enumerate(zip(probs.tolist(), lbls.tolist())):
            if lbl == 0 and prob >= threshold:
                # Benign but predicted injection (false positive)
                fp_embs.append(eval_dataset.embeddings[offset + i])
                fp_labels.append(0)
            elif lbl == 1 and prob < threshold:
                # Injection but predicted benign (false negative — more dangerous)
                fn_embs.append(eval_dataset.embeddings[offset + i])
                fn_labels.append(1)
        offset += len(lbls)

    n_fp = len(fp_embs)
    n_fn = len(fn_embs)

    if n_fp == 0 and n_fn == 0:
        print("  No hard negatives found — skipping Phase 2.")
        return None

    print(f"  Hard FP (benign->injection): {n_fp:,} -> upsampled x{fp_upsample}")
    print(f"  Hard FN (injection->benign): {n_fn:,} -> upsampled x{fn_upsample}")

    all_embs:   list[torch.Tensor] = []
    all_labels: list[int]          = []

    if fp_embs:
        all_embs.extend(fp_embs * fp_upsample)
        all_labels.extend(fp_labels * fp_upsample)
    if fn_embs:
        all_embs.extend(fn_embs * fn_upsample)
        all_labels.extend(fn_labels * fn_upsample)

    emb_tensor = torch.stack(all_embs)
    lbl_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Wrap in a simple EmbeddingDataset-compatible object
    ds = EmbeddingDataset.__new__(EmbeddingDataset)
    ds.embeddings = emb_tensor
    ds.labels     = lbl_tensor
    return ds


# ─── Main ─────────────────────────────────────────────────────────────────────

def train(
    train_file: Path = TRAIN_FILE,
    eval_file: Path = EVAL_FILE,
    model_path: Path = MODEL_PATH,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {EMBEDDING_MODEL}")
    print(f"Lexical features: {USE_LEXICAL_FEATURES}")

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
    eval_texts,  eval_labels  = load_jsonl(eval_file)
    print(f"  Train: {len(train_texts):,} samples  "
          f"({train_labels.count(0):,} benign / {train_labels.count(1):,} injection)")
    print(f"  Eval : {len(eval_texts):,} samples  "
          f"({eval_labels.count(0):,} benign / {eval_labels.count(1):,} injection)")

    # ── Encode texts ─────────────────────────────────────────────────────────
    # GPU fp16 encoding + disk caching:
    #   • First run : encoder on GPU in fp16 (~220 MB VRAM), batch=16
    #                 saves .pt cache to data/embed_cache/
    #   • Later runs: loads cache directly (≈5 s), skips encoding
    # After encoding, encoder is deleted to free VRAM for MLP training.
    print(f"\nEncoding texts with '{EMBEDDING_MODEL}' (GPU fp16 + disk cache)...")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    encoder = load_encoder(EMBEDDING_MODEL, device=str(device))
    t0 = time.time()
    train_dataset = build_dataset_cached(
        train_texts, train_labels, train_file, encoder, gpu_batch_size=256
    )
    eval_dataset = build_dataset_cached(
        eval_texts, eval_labels, eval_file, encoder, gpu_batch_size=256
    )
    # Free encoder from VRAM — not needed for MLP training
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        print(f"  VRAM free after encoder release: {free_mb:.0f} MB")
    print(f"  Encoding/loading took {time.time() - t0:.1f}s")
    print(f"  Feature dim: {train_dataset.embeddings.shape[1]}")

    # pin_memory: keeps embedding tensors in pinned host RAM for fast GPU transfer
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=use_pin)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=use_pin)

    # ── Model, optimizer, loss ───────────────────────────────────────────────
    input_dim = train_dataset.embeddings.shape[1]
    model = MLPClassifier(input_dim=input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} trainable parameters  (input_dim={input_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # OneCycleLR: better convergence + implicit warm-up, no manual scheduling needed
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Compute class weights for FocalLoss (same formula as before)
    n_benign    = train_labels.count(0)
    n_injection = train_labels.count(1)
    n_total     = n_benign + n_injection
    w_benign    = n_total / (2.0 * n_benign)    if n_benign    > 0 else 1.0
    w_injection = n_total / (2.0 * n_injection) if n_injection > 0 else 1.0
    class_weights = torch.tensor([w_benign, w_injection], dtype=torch.float32).to(device)

    # Phase 1 uses FocalLoss instead of CrossEntropyLoss
    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA,
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    # Eval criterion: plain CE (no focal) for stable val loss monitoring
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    print(f"  Class weights: benign={w_benign:.3f}  injection={w_injection:.3f}")
    print(f"  FocalLoss: gamma={FOCAL_GAMMA}  alpha={FOCAL_ALPHA}")
    print(f"  Mixup alpha: {MIXUP_ALPHA}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")

    # ── Phase 1: Training loop ────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1    = 0.0
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"  PHASE 1 — Full balanced training ({EPOCHS} epochs max)")
    print(f"{'='*65}")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val F1':>8}  {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler,   # OneCycleLR stepped per batch
            use_mixup=True,
        )
        val_loss, val_f1, _, _ = evaluate(model, eval_loader, eval_criterion, device)

        lr = optimizer.param_groups[0]["lr"]

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
                    "input_dim": input_dim,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "temperature": 1.0,
                },
                model_path,
            )
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PAT} epochs).")
                break

    print(f"\nPhase 1 Best Val F1: {best_f1:.4f}")

    # ── Reload best Phase-1 checkpoint ───────────────────────────────────────
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Phase 2: Hard negative fine-tuning ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PHASE 2 — Hard negative fine-tuning ({PHASE2_EPOCHS} epochs)")
    print(f"{'='*65}")

    # First calibrate threshold so we can identify hard negatives correctly
    initial_threshold = calibrate_threshold(model, eval_loader, device)
    hard_neg_ds = mine_hard_negatives(
        model, eval_dataset, device,
        threshold=initial_threshold,
        fp_upsample=HARD_NEG_UPSAMPLE,
        fn_upsample=HARD_FN_UPSAMPLE,
    )

    if hard_neg_ds is not None:
        # Mix hard negatives into training
        phase2_embs   = torch.cat([train_dataset.embeddings, hard_neg_ds.embeddings])
        phase2_labels = torch.cat([train_dataset.labels,     hard_neg_ds.labels])
        phase2_ds     = EmbeddingDataset.__new__(EmbeddingDataset)
        phase2_ds.embeddings = phase2_embs
        phase2_ds.labels     = phase2_labels
        phase2_loader = DataLoader(phase2_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        p2_optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
        p2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(p2_optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)
        # Equal weight in Phase 2 — symmetric treatment of both classes
        p2_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        p2_best_f1 = best_f1
        print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val F1':>8}")
        print("-" * 45)

        for epoch in range(1, PHASE2_EPOCHS + 1):
            train_loss = train_epoch(
                model, phase2_loader, p2_optimizer, p2_criterion, device, use_mixup=False
            )
            val_loss, val_f1, _, _ = evaluate(model, eval_loader, p2_criterion, device)
            p2_scheduler.step()

            flag = " *" if val_f1 > p2_best_f1 else ""
            print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {val_f1:>8.4f}{flag}")

            if val_f1 > p2_best_f1:
                p2_best_f1 = val_f1
                torch.save(
                    {
                        "epoch": f"p2_{epoch}",
                        "model_state_dict": model.state_dict(),
                        "val_f1": val_f1,
                        "val_loss": val_loss,
                        "input_dim": input_dim,
                        "threshold": CONFIDENCE_THRESHOLD,
                        "temperature": 1.0,
                    },
                    model_path,
                )

        print(f"\nPhase 2 Best Val F1: {p2_best_f1:.4f}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # ── Temperature calibration ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TEMPERATURE CALIBRATION (Guo et al., 2017)")
    print(f"{'='*65}")
    scaler = TemperatureScaler(model).to(device)
    fitted_T = scaler.calibrate(eval_loader, device)

    # ── Threshold calibration (final, on temperature-scaled model) ───────────
    print(f"\n{'='*65}")
    print(f"  Calibrating inference threshold on validation set...")
    print(f"{'='*65}")
    best_threshold = calibrate_threshold(scaler, eval_loader, device)

    # Update saved checkpoint with calibrated threshold + temperature
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt["threshold"]   = best_threshold
    ckpt["temperature"] = fitted_T
    ckpt["input_dim"]   = input_dim
    torch.save(ckpt, model_path)
    print(f"  Checkpoint updated with threshold={best_threshold:.2f}  temperature={fitted_T:.4f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nFinal evaluation (best checkpoint with calibrated threshold + temperature)...")
    scaler.eval()
    all_preds_raw, all_probs, all_labels_lst = [], [], []
    with torch.no_grad():
        for embeddings, labels in eval_loader:
            embeddings = embeddings.to(device)
            probs = scaler.predict_proba(embeddings)
            raw_preds = (probs[:, 1] >= best_threshold).long()
            all_preds_raw.extend(raw_preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels_lst.extend(labels.tolist())

    print(f"\nWith calibrated threshold={best_threshold:.2f}, T={fitted_T:.4f}:")
    print(classification_report(
        all_labels_lst, all_preds_raw,
        target_names=["benign", "injection"],
        digits=4,
    ))

    # Also show argmax (raw) results for comparison
    model.eval()
    all_preds_argmax = []
    with torch.no_grad():
        for embeddings, _ in eval_loader:
            embeddings = embeddings.to(device)
            preds = model(embeddings).argmax(dim=-1)
            all_preds_argmax.extend(preds.cpu().tolist())
    print("With argmax (threshold=0.50, no temperature):")
    print(classification_report(
        all_labels_lst, all_preds_argmax,
        target_names=["benign", "injection"],
        digits=4,
    ))


if __name__ == "__main__":
    train()
