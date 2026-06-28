"""Central configuration for the prompt injection classifier."""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE  = DATA_DIR / "eval.jsonl"
MODEL_PATH = CHECKPOINT_DIR / "mlp_classifier.pt"

# ─── Embedding model ──────────────────────────────────────────────────────────
# Using all-MiniLM-L6-v2 (22M params, 384-dim) — 5x faster than mpnet on
# GTX 1650 Max-Q, encodes 200K texts in ~15 min vs 11 hours.
# The lexical feature layer (16-dim) compensates for the smaller encoder by
# providing direct access to obfuscation/keyword signals the encoder misses.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ─── Handcrafted lexical features ─────────────────────────────────────────────
# 16-dim feature vector appended to encoder embedding.
# These capture signals the MiniLM encoder collapses:
#   base64 patterns, unicode escapes, injection keyword density,
#   entropy anomalies, imperative commands, override patterns, etc.
USE_LEXICAL_FEATURES = True
LEXICAL_FEATURE_DIM  = 16

# Total MLP input = EMBEDDING_DIM + LEXICAL_FEATURE_DIM
MLP_INPUT_DIM = EMBEDDING_DIM + (LEXICAL_FEATURE_DIM if USE_LEXICAL_FEATURES else 0)

# ─── MLP architecture ─────────────────────────────────────────────────────────
# Wider and deeper than the original [256, 128] to exploit the 400-dim input.
# LayerNorm + residual connections (see model.py) prevent collapse through depth.
HIDDEN_DIMS = [384, 256, 128]
DROPOUT     = 0.20
NUM_CLASSES = 2             # 0 = benign, 1 = injection

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 512          # larger batches — embeddings are small (400-dim)
EPOCHS          = 50
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PAT  = 8
LABEL_SMOOTHING = 0.05
SEED            = 42

# Phase-2 fine-tuning on hard negatives
PHASE2_EPOCHS     = 10
PHASE2_LR         = 5e-5
HARD_NEG_UPSAMPLE = 3

# ─── Inference calibration ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
UNCERTAINTY_LOW      = 0.45

# ─── Data ─────────────────────────────────────────────────────────────────────
EVAL_RATIO = 0.15
