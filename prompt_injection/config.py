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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, strong
EMBEDDING_DIM   = 384

# ─── MLP architecture ─────────────────────────────────────────────────────────
HIDDEN_DIMS = [256, 128]   # hidden layer widths
DROPOUT     = 0.3
NUM_CLASSES = 2            # 0 = benign, 1 = injection

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 256
EPOCHS         = 30
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 1e-4
EARLY_STOP_PAT = 5          # patience (epochs without val improvement)
SEED           = 42

# ─── Data download ────────────────────────────────────────────────────────────
# Sources pulled from Hugging Face (see prompt_injection/generate_data.py):
#   deepset/prompt-injections
#   watchdogsrox/Mirror-Prompt-Injection-Dataset
#   fka/awesome-chatgpt-prompts  (benign)
EVAL_RATIO = 0.15   # fraction of data held out for evaluation
