"""Dataset class and embedding helpers for the prompt injection classifier.

Each JSONL line has the schema: {"text": "...", "label": 0|1}
  label 0 → benign
  label 1 → prompt injection

GPU-accelerated encoding strategy for 4 GB GPUs:
  - Encoder runs on GPU in fp16 (half precision) to halve VRAM usage.
    mpnet fp16 = ~220 MB vs ~438 MB fp32; attention activations also halved.
  - Batch size of 16 keeps peak VRAM under ~1.5 GB for the encoder pass.
  - Output is immediately cast back to fp32 on CPU for stable MLP training.
  - Embeddings are disk-cached after first computation: subsequent runs
    skip encoding entirely and load in ~5 seconds instead of ~1 hour.

Final feature vector: cat([encoder_embedding (768), lexical_features (16)])
"""
import gc
import hashlib
import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prompt_injection.config import EMBEDDING_MODEL, USE_LEXICAL_FEATURES, DATA_DIR, ENCODE_FP16


# ─── Embedding cache ──────────────────────────────────────────────────────────
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"


def _cache_key(data_file: Path, model_name: str, use_lexical: bool) -> str:
    """
    Stable cache key based on:
      - First 4 KB of the data file (catches regenerated data)
      - File size (catches appended data)
      - Model name and lexical flag
    """
    with open(data_file, "rb") as f:
        header = f.read(4096)
    size = data_file.stat().st_size
    digest = hashlib.md5(
        f"{header}|{size}|{model_name}|{use_lexical}".encode()
    ).hexdigest()[:12]
    stem = data_file.stem   # "train" or "eval"
    return f"{stem}_{digest}.pt"


def _save_cache(path: Path, embeddings: torch.Tensor) -> None:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved embedding cache: {path.name}  ({size_mb:.0f} MB)")


def _load_cache(path: Path) -> torch.Tensor:
    emb = torch.load(path, map_location="cpu", weights_only=False)
    size_mb = path.stat().st_size / 1e6
    print(f"  Loaded embedding cache: {path.name}  ({size_mb:.0f} MB)")
    return emb


# ─── Encoder loader ───────────────────────────────────────────────────────────

def load_encoder(model_name: str = EMBEDDING_MODEL, device: str = "cpu") -> SentenceTransformer:
    """Load the sentence-transformer encoder (HF-cached locally after first use)."""
    return SentenceTransformer(model_name, device=device)


# ─── GPU fp16 chunked encoding ────────────────────────────────────────────────

@torch.no_grad()
def encode_texts_gpu_fp16(
    texts: list[str],
    encoder: SentenceTransformer,
    batch_size: int = 16,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Encode texts on GPU using fp16 to stay within 4 GB VRAM.

    Strategy:
      1. Cast encoder weights to fp16  → ~220 MB VRAM (vs 438 MB fp32)
      2. Encode in small batches       → attention activations tiny (~80 MB/batch)
      3. Convert output to fp32 on CPU → stable downstream MLP training
      4. Restore encoder to fp32       → leave model in clean state

    Returns float32 tensor of shape (N, embedding_dim) on CPU.
    """
    device = next(encoder.parameters()).device

    # Cast encoder to fp16 to halve VRAM for weights + activations if ENCODE_FP16 is True
    if ENCODE_FP16:
        encoder.half()

    all_embs: list[torch.Tensor] = []
    n = len(texts)
    desc = "Encoding (GPU fp16)" if ENCODE_FP16 else "Encoding (GPU fp32)"
    batches = range(0, n, batch_size)

    try:
        for i in tqdm(batches, desc=desc, unit="batch", disable=not show_progress):
            batch = texts[i : i + batch_size]
            emb = encoder.encode(
                batch,
                batch_size=len(batch),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Immediately move to CPU as float32 to free VRAM
            all_embs.append(emb.float().cpu())

            # Periodically clear VRAM cache
            if device.type == "cuda" and (i // batch_size) % 50 == 0:
                torch.cuda.empty_cache()

    finally:
        # Always restore encoder to fp32 (don't leave it mutated)
        if ENCODE_FP16:
            encoder.float()

    return torch.cat(all_embs, dim=0)


def encode_texts(
    texts: list[str],
    encoder: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Standard encoding (fp32). Used at inference time for small batches.
    For large offline encoding use encode_texts_gpu_fp16().
    """
    device = next(encoder.parameters()).device
    if device.type == "cuda":
        # Even at inference, use fp16 to avoid OOM on 4 GB GPU
        return encode_texts_gpu_fp16(
            texts, encoder,
            batch_size=min(batch_size, 32),
            show_progress=show_progress,
        )
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return embeddings.float()


def encode_with_features(
    texts: list[str],
    encoder: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = False,
    use_lexical: bool = USE_LEXICAL_FEATURES,
) -> torch.Tensor:
    """
    Encode texts and concatenate handcrafted lexical features.

    Returns float32 CPU tensor of shape:
      (N, embedding_dim)                    if use_lexical=False
      (N, embedding_dim + LEXICAL_FEAT_DIM) if use_lexical=True
    """
    emb = encode_texts(texts, encoder, batch_size=batch_size, show_progress=show_progress)
    emb = emb.cpu()

    if not use_lexical:
        return emb

    from prompt_injection.feature_engineering import compute_features_batch
    feats = compute_features_batch(texts)  # (N, 16) on CPU
    return torch.cat([emb, feats], dim=1)


# ─── Cached dataset builder ───────────────────────────────────────────────────

def build_dataset_cached(
    texts: list[str],
    labels: list[int],
    data_file: Path,
    encoder: SentenceTransformer,
    gpu_batch_size: int = 16,
    use_lexical: bool = USE_LEXICAL_FEATURES,
) -> "EmbeddingDataset":
    """
    Build an EmbeddingDataset with disk caching.

    First call: encodes on GPU (fp16) + computes lexical features → saves .pt cache.
    Subsequent calls: loads .pt cache directly (skip all encoding).

    Cache is invalidated automatically when the data file content changes.
    """
    key  = _cache_key(data_file, EMBEDDING_MODEL, use_lexical)
    path = EMBED_CACHE_DIR / key

    if path.exists():
        emb = _load_cache(path)
    else:
        print(f"  Cache miss — encoding {len(texts):,} texts on GPU fp16...")
        device = next(encoder.parameters()).device

        # Encode with GPU fp16
        raw_emb = encode_texts_gpu_fp16(
            texts, encoder,
            batch_size=gpu_batch_size,
            show_progress=True,
        )
        torch.cuda.empty_cache()
        gc.collect()

        if use_lexical:
            print("  Computing lexical features...")
            from prompt_injection.feature_engineering import compute_features_batch
            feats = compute_features_batch(texts)  # CPU, fast
            emb = torch.cat([raw_emb, feats], dim=1)
            del raw_emb
        else:
            emb = raw_emb

        _save_cache(path, emb)

    # Wrap in dataset object
    ds = EmbeddingDataset.__new__(EmbeddingDataset)
    ds.embeddings = emb
    ds.labels     = torch.tensor(labels, dtype=torch.long)
    assert len(ds.embeddings) == len(ds.labels)
    return ds


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
    """Pre-computed embedding dataset.

    Instantiate via ``build_dataset_cached()`` for the training pipeline
    (GPU fp16 + disk cache).  Pass ``embeddings`` directly for small tests.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        encoder: SentenceTransformer | None = None,
        embeddings: torch.Tensor | None = None,
        batch_size: int = 16,
        use_lexical: bool = USE_LEXICAL_FEATURES,
    ) -> None:
        if embeddings is not None:
            self.embeddings = embeddings
        elif encoder is not None:
            self.embeddings = encode_with_features(
                texts, encoder,
                batch_size=batch_size,
                show_progress=True,
                use_lexical=use_lexical,
            )
        else:
            raise ValueError("Either encoder or pre-computed embeddings must be supplied.")

        self.labels = torch.tensor(labels, dtype=torch.long)
        assert len(self.embeddings) == len(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
