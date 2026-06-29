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
# The lexical feature layer (25-dim) compensates for the smaller encoder by
# providing direct access to obfuscation/keyword signals the encoder misses.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
ENCODE_FP16     = False

# ─── Handcrafted lexical features ─────────────────────────────────────────────
# 25-dim feature vector appended to encoder embedding.
# Features f1-f16 (original): base64 patterns, unicode escapes, injection keyword
#   density, entropy anomalies, imperative commands, override patterns, etc.
# Features f17-f24 (IMPROVEMENTS.md): instruction hierarchy spoofing, nested
#   brackets, token smuggling, multilingual switch, sentence length variance,
#   number density, prompt framing phrases, ellipsis separators.
# Feature f25 (new): gibberish score — detects random character strings.
USE_LEXICAL_FEATURES = True
LEXICAL_FEATURE_DIM  = 25

# Total MLP input = EMBEDDING_DIM + LEXICAL_FEATURE_DIM
MLP_INPUT_DIM = EMBEDDING_DIM + (LEXICAL_FEATURE_DIM if USE_LEXICAL_FEATURES else 0)

# ─── MLP architecture ─────────────────────────────────────────────────────────
# Deeper network [512, 384, 256, 128] to handle the richer 409-dim input.
# LayerNorm + residual connections (see model.py) prevent collapse through depth.
HIDDEN_DIMS = [512, 384, 256, 128]
DROPOUT     = 0.20
NUM_CLASSES = 2             # 0 = benign, 1 = injection

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 512          # larger batches — embeddings are small (409-dim)
EPOCHS          = 50
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PAT  = 8
LABEL_SMOOTHING = 0.05
SEED            = 42

# Phase-2 fine-tuning on hard negatives
PHASE2_EPOCHS     = 10
PHASE2_LR         = 5e-5
HARD_NEG_UPSAMPLE = 3      # false-positive benign samples upsampled ×3
HARD_FN_UPSAMPLE  = 5      # false-negative injection samples upsampled ×5 (more dangerous)

# ─── Mixup training (Zhang et al., 2018) ──────────────────────────────────────
# Interpolates pairs of training samples in embedding space.
# Set to 0.0 to disable.
MIXUP_ALPHA = 0.2

# ─── Focal Loss (Lin et al., 2017) ────────────────────────────────────────────
# Down-weights easy examples, focuses gradient on hard ones.
# Used in Phase 1 instead of CrossEntropyLoss.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# ─── Inference calibration ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
UNCERTAINTY_LOW      = 0.35   # lowered from 0.45: short/benign text → confident benign

# ─── Keyword-based override guard ─────────────────────────────────────────────
# If a high-confidence injection keyword is detected (case-insensitive),
# the injection_prob is raised to at least this floor value.
# Bypasses the uncertainty bucket entirely for keyword-triggered samples.
KEYWORD_CONFIDENCE_FLOOR = 0.90

# ─── Gibberish guard ──────────────────────────────────────────────────────────
# If the gibberish score exceeds this threshold AND no injection keywords
# are present, the sample is hard-clamped to benign.
GIBBERISH_THRESHOLD = 0.80

# ─── Data ─────────────────────────────────────────────────────────────────────
EVAL_RATIO = 0.15
