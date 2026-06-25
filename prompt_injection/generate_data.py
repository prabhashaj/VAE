"""Real-data prompt injection dataset downloader — no synthetic data.

Pulls from verified HuggingFace datasets covering diverse attack vectors:
  jailbreaks, DoS, indirect injection, roleplay escapes, data exfiltration,
  obfuscation, context manipulation, in-the-wild attacks, toxic chat.

FREE (no login required):
  watchdogsrox/Mirror-Prompt-Injection-Dataset  ~5K  injection+benign
  deepset/prompt-injections                      ~660  injection+benign
  jackhhao/jailbreak-classification             ~1.3K  injection+benign
  TrustAIRLab/in-the-wild-jailbreak-prompts     ~2K  injection+benign
  lmsys/toxic-chat                              ~10K  injection+benign
  rubend18/ChatGPT-Jailbreak-Prompts             ~78  injection
  PKU-Alignment/BeaverTails                    ~330K  injection+benign
  tatsu-lab/alpaca                              ~52K  benign
  databricks/databricks-dolly-15k              ~15K  benign
  HuggingFaceH4/no_robots                      ~10K  benign
  fka/awesome-chatgpt-prompts                   ~2K  benign
  LibrAI/do-not-answer                          ~939  injection

GATED (requires: huggingface-cli login):
  hackaprompt/hackaprompt-dataset              ~600K  injection
  allenai/wildjailbreak                         ~52K  injection+benign

NOTE: To unlock gated datasets run:
  pip install huggingface_hub
  huggingface-cli login
Then re-run this script to get up to 200K samples.

Run:
    python -m prompt_injection.generate_data
    python -m prompt_injection.generate_data --target 200000
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from prompt_injection.config import TRAIN_FILE, EVAL_FILE, SEED

random.seed(SEED)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_bool(val) -> bool:
    """Robustly convert any HuggingFace boolean variant to Python bool."""
    if val is None:
        return True        # default to safe if unknown
    try:
        return bool(val)
    except Exception:
        return True


def _truncate(text: str, max_len: int = 2000) -> str:
    return text[:max_len].strip()


# ─── Injection datasets ───────────────────────────────────────────────────────

def iter_mirror() -> Iterator[dict]:
    """watchdogsrox/Mirror-Prompt-Injection-Dataset
    Schema: text, label (0/1), category
    ~4,995 injection + ~4,995 benign — mirror-paired to avoid keyword shortcuts.
    Attack types: ignore, extraction, persona, indirect.
    """
    print("  Loading Mirror-Prompt-Injection-Dataset ...")
    try:
        ds = load_dataset("watchdogsrox/Mirror-Prompt-Injection-Dataset")
        n = 0
        for split in ds:
            for row in ds[split]:
                text  = (row.get("text") or "").strip()
                label = int(row.get("label", 0))
                cat   = str(row.get("category") or ("injection" if label else "benign"))
                if text:
                    yield {"text": _truncate(text), "label": label,
                           "_src": "mirror", "_cat": cat}
                    n += 1
        print(f"    mirror: {n:,} records")
    except Exception as e:
        print(f"    WARNING mirror: {e}")


def iter_deepset() -> Iterator[dict]:
    """deepset/prompt-injections
    Schema: text, label (0/1)
    ~330 injection + ~330 benign — curated classic injection patterns.
    """
    print("  Loading deepset/prompt-injections ...")
    try:
        ds = load_dataset("deepset/prompt-injections")
        n = 0
        for split in ds:
            for row in ds[split]:
                text  = (row.get("text") or "").strip()
                label = int(row.get("label", 0))
                if text:
                    yield {"text": _truncate(text), "label": label,
                           "_src": "deepset",
                           "_cat": "injection" if label else "benign"}
                    n += 1
        print(f"    deepset: {n:,} records")
    except Exception as e:
        print(f"    WARNING deepset: {e}")


def iter_jackhhao() -> Iterator[dict]:
    """jackhhao/jailbreak-classification
    Schema: prompt, type ("jailbreak" | "normal")
    ~650 jailbreak + ~650 normal.
    """
    print("  Loading jackhhao/jailbreak-classification ...")
    try:
        ds = load_dataset("jackhhao/jailbreak-classification")
        n = 0
        for split in ds:
            for row in ds[split]:
                text  = (row.get("prompt") or row.get("text") or "").strip()
                ptype = (row.get("type") or "").strip().lower()
                label = 1 if "jailbreak" in ptype else 0
                if text:
                    yield {"text": _truncate(text), "label": label,
                           "_src": "jackhhao", "_cat": ptype}
                    n += 1
        print(f"    jackhhao: {n:,} records")
    except Exception as e:
        print(f"    WARNING jackhhao: {e}")


def iter_in_the_wild() -> Iterator[dict]:
    """TrustAIRLab/in-the-wild-jailbreak-prompts
    4 configs: jailbreak_2023_05_07, jailbreak_2023_12_25 (label=1)
               regular_2023_05_07,   regular_2023_12_25   (label=0)
    In-the-wild real jailbreaks collected from Reddit/Discord/forums.
    """
    print("  Loading TrustAIRLab/in-the-wild-jailbreak-prompts ...")
    config_labels = [
        ("jailbreak_2023_05_07", 1, "jailbreak"),
        ("jailbreak_2023_12_25", 1, "jailbreak"),
        ("regular_2023_05_07",   0, "regular"),
        ("regular_2023_12_25",   0, "regular"),
    ]
    n = 0
    for config, label, cat in config_labels:
        try:
            ds = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts", config, split="train"
            )
            for row in ds:
                text = (row.get("prompt") or row.get("text") or "").strip()
                if text:
                    yield {"text": _truncate(text), "label": label,
                           "_src": "in_the_wild", "_cat": cat}
                    n += 1
        except Exception as ec:
            print(f"    WARNING in-the-wild [{config}]: {ec}")
    print(f"    in-the-wild: {n:,} records")


def iter_toxic_chat() -> Iterator[dict]:
    """lmsys/toxic-chat — real user-AI conversations.
    Configs: toxicchat0124, toxicchat1123. Splits: train, test.
    Schema: human_turn, toxicity (0/1 stored as int/float/str), jailbreaking (0/1)
    Covers: toxic requests, jailbreak attempts, real-world adversarial chat.
    """
    print("  Loading lmsys/toxic-chat ...")
    n_inj = n_ben = 0
    for cfg in ("toxicchat0124", "toxicchat1123"):
        for split_name in ("train", "test"):
            try:
                ds = load_dataset("lmsys/toxic-chat", cfg, split=split_name)
                for row in ds:
                    text = (row.get("user_input") or row.get("human_turn") or "").strip()
                    if not text:
                        continue
                    # Handle int/float/str/bool variants
                    raw_tox  = row.get("toxicity",    0)
                    raw_jail = row.get("jailbreaking", 0)
                    try:
                        tox  = float(str(raw_tox).strip())  if raw_tox  is not None else 0.0
                        jail = float(str(raw_jail).strip()) if raw_jail is not None else 0.0
                    except (ValueError, TypeError):
                        tox = jail = 0.0
                    if tox >= 0.5 or jail >= 0.5:
                        yield {"text": _truncate(text), "label": 1,
                               "_src": "toxic_chat",
                               "_cat": "jailbreak" if jail >= 0.5 else "toxic"}
                        n_inj += 1
                    else:
                        yield {"text": _truncate(text), "label": 0,
                               "_src": "toxic_chat", "_cat": "benign"}
                        n_ben += 1
            except Exception:
                continue
    print(f"    toxic-chat: {n_inj:,} injection, {n_ben:,} benign")


def iter_cyberec() -> Iterator[dict]:
    """cyberec/llm-prompt-injection-attacks
    Schema: text, labels (list of strings)
    ~55K samples.
    """
    print("  Loading cyberec/llm-prompt-injection-attacks ...")
    n_inj = n_ben = 0
    for split_name in ("train", "validation"):
        try:
            ds = load_dataset("cyberec/llm-prompt-injection-attacks", split=split_name, streaming=True)
            for row in ds:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                labels = row.get("labels") or []
                is_benign = "BENIGN" in labels
                if is_benign:
                    yield {"text": _truncate(text), "label": 0,
                           "_src": "cyberec", "_cat": "benign"}
                    n_ben += 1
                else:
                    other_labels = [l for l in labels if l != "BENIGN"]
                    cat = other_labels[0] if other_labels else "injection"
                    yield {"text": _truncate(text), "label": 1,
                           "_src": "cyberec", "_cat": cat}
                    n_inj += 1
        except Exception as e:
            print(f"    WARNING cyberec [{split_name}]: {e}")
    print(f"    cyberec: {n_inj:,} injection, {n_ben:,} benign")


def iter_pku_saferlhf() -> Iterator[dict]:
    """PKU-Alignment/PKU-SafeRLHF
    ~330k QA pairs. We map:
      is_response_0_safe & is_response_1_safe.
      If both are True -> benign (0)
      If either is False -> injection/unsafe (1)
    """
    print("  Loading PKU-Alignment/PKU-SafeRLHF ...")
    n_inj = n_ben = 0
    for split_name in ("train", "test"):
        try:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split_name, streaming=True)
            for row in ds:
                text = (row.get("prompt") or "").strip()
                if not text:
                    continue
                safe0 = _safe_bool(row.get("is_response_0_safe"))
                safe1 = _safe_bool(row.get("is_response_1_safe"))
                if not safe0 or not safe1:
                    yield {"text": _truncate(text), "label": 1,
                           "_src": "pku_saferlhf", "_cat": "unsafe"}
                    n_inj += 1
                else:
                    yield {"text": _truncate(text), "label": 0,
                           "_src": "pku_saferlhf", "_cat": "safe"}
                    n_ben += 1
        except Exception as e:
            print(f"    WARNING pku_saferlhf [{split_name}]: {e}")
    print(f"    pku_saferlhf: {n_inj:,} injection, {n_ben:,} benign")


def iter_neuralchemy() -> Iterator[dict]:
    """neuralchemy/Prompt-injection-dataset
    Splits: train, validation, test
    Schema: text, label (0/1)
    """
    print("  Loading neuralchemy/Prompt-injection-dataset ...")
    n_inj = n_ben = 0
    for split_name in ("train", "validation", "test"):
        try:
            ds = load_dataset("neuralchemy/Prompt-injection-dataset", split=split_name, streaming=True)
            for row in ds:
                text = (row.get("text") or "").strip()
                label = row.get("label")
                if text and label is not None:
                    label = int(label)
                    yield {"text": _truncate(text), "label": label,
                           "_src": "neuralchemy", "_cat": "injection" if label == 1 else "benign"}
                    if label == 1:
                        n_inj += 1
                    else:
                        n_ben += 1
        except Exception as e:
            print(f"    WARNING neuralchemy [{split_name}]: {e}")
    print(f"    neuralchemy: {n_inj:,} injection, {n_ben:,} benign")


def iter_llmail() -> Iterator[dict]:
    """microsoft/llmail-inject-challenge
    Phase1: ~370K rows. Phase2: ~90k rows. All are injection/attack attempts (label=1).
    We extract 'body' or 'subject' if body is empty.
    """
    print("  Loading microsoft/llmail-inject-challenge ...")
    n = 0
    for split_name in ("Phase1", "Phase2"):
        try:
            ds = load_dataset("microsoft/llmail-inject-challenge", split=split_name, streaming=True)
            for row in ds:
                text = (row.get("body") or row.get("subject") or "").strip()
                if text:
                    yield {"text": _truncate(text), "label": 1,
                           "_src": "llmail", "_cat": "adaptive_attack"}
                    n += 1
        except Exception as e:
            print(f"    WARNING llmail [{split_name}]: {e}")
    print(f"    llmail: {n:,} records")



def iter_rubend18() -> Iterator[dict]:
    """rubend18/ChatGPT-Jailbreak-Prompts — curated ChatGPT jailbreaks.
    ~78 unique high-quality jailbreak prompts. DAN, AIM, STAN, etc.
    """
    print("  Loading rubend18/ChatGPT-Jailbreak-Prompts ...")
    for split_name in ("train", "test", "validation"):
        try:
            ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split=split_name)
            n = 0
            for row in ds:
                # Column may be 'Prompt', 'text', or 'prompt'
                text = (
                    row.get("Prompt") or row.get("text") or
                    row.get("prompt") or row.get("jailbreak") or ""
                ).strip()
                if text:
                    yield {"text": _truncate(text), "label": 1,
                           "_src": "rubend18", "_cat": "chatgpt_jailbreak"}
                    n += 1
            print(f"    rubend18: {n:,} records")
            return
        except Exception:
            continue
    print("    WARNING rubend18: all splits failed")



def iter_do_not_answer() -> Iterator[dict]:
    """LibrAI/do-not-answer — 939 instructions LLMs should refuse.
    Schema: id, risk_area, types_of_harm, specific_harms, question
    Covers: discrimination, hate speech, violence, privacy violations,
            malicious code, false info, harmful legal/financial/medical advice.
    """
    print("  Loading LibrAI/do-not-answer ...")
    try:
        ds = load_dataset("LibrAI/do-not-answer", split="train")
        n = 0
        for row in ds:
            text = (row.get("question") or row.get("instruction") or "").strip()
            if text:
                cat = row.get("types_of_harm") or row.get("risk_area") or "harmful"
                yield {"text": _truncate(text), "label": 1,
                       "_src": "do_not_answer", "_cat": str(cat)}
                n += 1
        print(f"    do-not-answer: {n:,} records")
    except Exception as e:
        print(f"    WARNING do-not-answer: {e}")


def iter_beavertails() -> Iterator[dict]:
    """PKU-Alignment/BeaverTails — 330K QA pairs, 14 harm categories.
    Uses 330k_train for broader coverage including harmful prompts.
    """
    print("  Loading PKU-Alignment/BeaverTails ...")
    n_inj = n_ben = 0
    for split_name in ("330k_train", "330k_test", "30k_train"):
        try:
            ds = load_dataset("PKU-Alignment/BeaverTails", split=split_name, streaming=True)
            per_split_cap = 350_000
            count = 0
            for row in ds:
                if count >= per_split_cap:
                    break
                text = (row.get("prompt") or "").strip()
                if not text:
                    continue
                s0 = row.get("is_safe") if "is_safe" in row else row.get("is_response_0_safe")
                s1 = row.get("is_safe") if "is_safe" in row else row.get("is_response_1_safe")
                # Convert to bool robustly
                safe0 = _safe_bool(s0)
                safe1 = _safe_bool(s1)
                cat = str(row.get("category") or "harm")
                if not safe0 or not safe1:
                    # At least one unsafe response → likely harmful prompt
                    yield {"text": _truncate(text), "label": 1,
                           "_src": "beavertails", "_cat": cat}
                    n_inj += 1
                else:
                    yield {"text": _truncate(text), "label": 0,
                           "_src": "beavertails", "_cat": "safe"}
                    n_ben += 1
                count += 1
            print(f"    beavertails [{split_name}]: {n_inj:,} inj, {n_ben:,} ben so far")
        except Exception as es:
            print(f"    WARNING beavertails [{split_name}]: {es}")
    print(f"    beavertails total: {n_inj:,} injection, {n_ben:,} benign")


# ─── GATED datasets (require huggingface-cli login) ──────────────────────────

def iter_hackaprompt(cap: int = 150_000) -> Iterator[dict]:
    """hackaprompt/hackaprompt-dataset  [GATED — needs HF login]
    ~600K competition prompt-injection attempts. All are injections (label=1).
    Schema: prompt, level, model, user_input, completion, injection_successful
    Attack levels 0-7 covering increasing sophistication.
    """
    print("  Loading hackaprompt/hackaprompt-dataset [GATED] ...")
    try:
        ds = load_dataset("hackaprompt/hackaprompt-dataset", split="train", streaming=True)
        n = 0
        seen: set[str] = set()
        for row in ds:
            if n >= cap:
                break
            text = (row.get("user_input") or row.get("prompt") or "").strip()
            if not text or len(text) < 8:
                continue
            key = text[:100]
            if key in seen:
                continue
            seen.add(key)
            level = row.get("level", 0)
            yield {"text": _truncate(text), "label": 1,
                   "_src": "hackaprompt", "_cat": f"level_{level}"}
            n += 1
        print(f"    hackaprompt: {n:,} records")
    except Exception as e:
        print(f"    WARNING hackaprompt (gated): {e}")
        print("    --> Run: huggingface-cli login")


def iter_wildjailbreak(inj_cap: int = 120_000, ben_cap: int = 120_000) -> Iterator[dict]:
    """allenai/wildjailbreak  [GATED — needs HF login]
    ~262K samples. adversarial column = jailbreak (label=1),
                   vanilla column = benign version (label=0).
    Tactics: persona, context manipulation, obfuscation, roleplay, etc.
    """
    print("  Loading allenai/wildjailbreak [GATED] ...")
    try:
        ds = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
        n_inj = n_ben = 0
        for row in ds:
            if n_inj >= inj_cap and n_ben >= ben_cap:
                break
            adv = (row.get("adversarial") or "").strip()
            van = (row.get("vanilla") or "").strip()
            tactics = row.get("tactics") or []
            cat = ",".join(str(t) for t in tactics[:2]) if tactics else "jailbreak"
            if adv and n_inj < inj_cap:
                yield {"text": _truncate(adv), "label": 1,
                       "_src": "wildjailbreak", "_cat": cat}
                n_inj += 1
            if van and n_ben < ben_cap:
                yield {"text": _truncate(van), "label": 0,
                       "_src": "wildjailbreak", "_cat": "vanilla"}
                n_ben += 1
        print(f"    wildjailbreak: {n_inj:,} injection, {n_ben:,} benign")
    except Exception as e:
        print(f"    WARNING wildjailbreak (gated): {e}")
        print("    --> Run: huggingface-cli login")


# ─── Benign-only datasets ─────────────────────────────────────────────────────

def iter_alpaca(cap: int = 100_000) -> Iterator[dict]:
    """tatsu-lab/alpaca — 52K instruction-following prompts. All benign."""
    print("  Loading tatsu-lab/alpaca ...")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        n = 0
        for row in ds:
            if n >= cap:
                break
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            text = f"{instruction} {inp}".strip() if inp else instruction
            if text:
                yield {"text": _truncate(text), "label": 0,
                       "_src": "alpaca", "_cat": "instruction"}
                n += 1
        print(f"    alpaca: {n:,} records")
    except Exception as e:
        print(f"    WARNING alpaca: {e}")


def iter_dolly() -> Iterator[dict]:
    """databricks/databricks-dolly-15k — 15K diverse benign human-authored instructions."""
    print("  Loading databricks/databricks-dolly-15k ...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        n = 0
        for row in ds:
            text = (row.get("instruction") or "").strip()
            ctx  = (row.get("context") or "").strip()
            if ctx:
                text = f"{text} {ctx}".strip()
            if text:
                yield {"text": _truncate(text), "label": 0,
                       "_src": "dolly", "_cat": str(row.get("category", "general"))}
                n += 1
        print(f"    dolly: {n:,} records")
    except Exception as e:
        print(f"    WARNING dolly: {e}")


def iter_no_robots() -> Iterator[dict]:
    """HuggingFaceH4/no_robots — 10K high-quality benign chat instructions."""
    print("  Loading HuggingFaceH4/no_robots ...")
    try:
        n = 0
        for split_name in ("train", "test"):
            try:
                ds = load_dataset("HuggingFaceH4/no_robots", split=split_name)
                for row in ds:
                    messages = row.get("messages") or []
                    user_msgs = [m.get("content", "") for m in messages
                                 if m.get("role") == "user"]
                    text = " ".join(user_msgs).strip()
                    if text:
                        yield {"text": _truncate(text), "label": 0,
                               "_src": "no_robots",
                               "_cat": str(row.get("category", "chat"))}
                        n += 1
            except Exception:
                continue
        print(f"    no_robots: {n:,} records")
    except Exception as e:
        print(f"    WARNING no_robots: {e}")


def iter_awesome() -> Iterator[dict]:
    """fka/awesome-chatgpt-prompts — 2K system-role prompts. All benign."""
    try:
        ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
        n = 0
        for row in ds:
            text = (row.get("prompt") or "").strip()
            if text:
                yield {"text": _truncate(text), "label": 0,
                       "_src": "awesome", "_cat": "system_prompt"}
                n += 1
        print(f"    awesome-chatgpt: {n:,} records")
    except Exception as e:
        print(f"    WARNING awesome: {e}")


# ─── Ordered source pipeline ──────────────────────────────────────────────────
# Listed in priority order. Gated sources gracefully fail if not logged in.

ALL_SOURCES = [
    # --- Mixed (both labels) ---
    iter_mirror,
    iter_deepset,
    iter_jackhhao,
    iter_in_the_wild,
    iter_toxic_chat,
    iter_cyberec,
    iter_pku_saferlhf,
    iter_neuralchemy,
    iter_beavertails,
    # --- Injection only ---
    iter_rubend18,
    iter_do_not_answer,
    iter_llmail,
    # --- GATED injection ---
    iter_hackaprompt,
    iter_wildjailbreak,
    # --- Benign only ---
    iter_alpaca,
    iter_dolly,
    iter_no_robots,
    iter_awesome,
]


# ─── Collect + deduplicate + balance ─────────────────────────────────────────

def collect(target: int) -> tuple[list[dict], list[dict]]:
    target_each = target // 2
    injection: list[dict] = []
    benign:    list[dict] = []
    seen:      set[str]   = set()

    def _add(rec: dict) -> bool:
        key = rec["text"].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        if rec["label"] == 1 and len(injection) < target_each:
            injection.append(rec)
            return True
        if rec["label"] == 0 and len(benign) < target_each:
            benign.append(rec)
            return True
        return False

    print("\n=== Downloading datasets ===")
    for src_fn in ALL_SOURCES:
        # Stop when both buckets are full
        if len(injection) >= target_each and len(benign) >= target_each:
            break
        # Skip benign-only sources when benign bucket is full
        benign_only = src_fn in (iter_alpaca, iter_dolly, iter_no_robots, iter_awesome)
        if benign_only and len(benign) >= target_each:
            continue
        # Skip injection-only sources when injection bucket is full
        injection_only = src_fn in (iter_rubend18, iter_do_not_answer, iter_llmail, iter_hackaprompt)
        if injection_only and len(injection) >= target_each:
            continue
        try:
            for rec in src_fn():
                _add(rec)
        except Exception as ex:
            print(f"  ERROR in {src_fn.__name__}: {ex}")

    return injection, benign


# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict], title: str) -> None:
    src = Counter(r["_src"] for r in records)
    lbl = Counter(r["label"] for r in records)
    cat = Counter(r["_cat"] for r in records)
    print(f"\n{'='*55}")
    print(f"  {title} -- {len(records):,} total")
    print(f"  Label  : benign={lbl[0]:,}  injection={lbl[1]:,}")
    print(f"  Sources:")
    for s, c in sorted(src.items(), key=lambda x: -x[1]):
        print(f"    {s:<35}: {c:,}")
    print(f"  Top attack categories:")
    inj_cats = Counter(r["_cat"] for r in records if r["label"] == 1)
    for c, n in inj_cats.most_common(10):
        print(f"    {c:<35}: {n:,}")


# ─── Split ────────────────────────────────────────────────────────────────────

def split_data(injection: list[dict], benign: list[dict],
               eval_ratio: float) -> tuple[list[dict], list[dict]]:
    random.shuffle(injection)
    random.shuffle(benign)

    def _cut(lst: list) -> tuple[list, list]:
        n = max(1, int(len(lst) * eval_ratio))
        return lst[n:], lst[:n]

    i_tr, i_ev = _cut(injection)
    b_tr, b_ev = _cut(benign)

    train = i_tr + b_tr
    eval_ = i_ev + b_ev
    random.shuffle(train)
    random.shuffle(eval_)
    return train, eval_


# ─── Writer ───────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            out = {"text": rec["text"], "label": rec["label"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    b = sum(1 for r in records if r["label"] == 0)
    i = sum(1 for r in records if r["label"] == 1)
    print(f"  Wrote {len(records):,} ({b:,} benign / {i:,} injection) -> {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def download(
    train_file: Path  = TRAIN_FILE,
    eval_file:  Path  = EVAL_FILE,
    target:     int   = 200_000,
    eval_ratio: float = 0.15,
) -> None:
    print("=" * 55)
    print(f"Prompt Injection Dataset Builder")
    print(f"Target: {target:,} samples ({target//2:,} per class)")
    print("=" * 55)
    print("\nNOTE: Gated datasets require: huggingface-cli login")
    print("      Free sources alone yield ~15-20K balanced samples.\n")

    injection, benign = collect(target)

    # If injection << benign, cap benign to injection count for balance
    min_count = min(len(injection), len(benign))
    if len(benign) > min_count * 1.5:
        random.shuffle(benign)
        benign = benign[:min_count]
        print(f"\nCapped benign to {min_count:,} to match injection count.")

    all_records = injection + benign
    print_stats(all_records, "Full dataset")

    train, eval_ = split_data(injection, benign, eval_ratio)
    print_stats(train, "Train split")

    b_ev = sum(1 for r in eval_ if r["label"] == 0)
    i_ev = sum(1 for r in eval_ if r["label"] == 1)
    print(f"\n  Eval split -- {len(eval_):,} total")
    print(f"  Label  : benign={b_ev:,}  injection={i_ev:,}")

    print()
    write_jsonl(train_file, train)
    write_jsonl(eval_file,  eval_)
    print("\nDone! Run: python -m prompt_injection.train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download prompt injection datasets from HuggingFace"
    )
    parser.add_argument("--target",     type=int,   default=200_000,
                        help="Total sample target (default 200000)")
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--train-file", type=Path,  default=TRAIN_FILE)
    parser.add_argument("--eval-file",  type=Path,  default=EVAL_FILE)
    args = parser.parse_args()
    download(args.train_file, args.eval_file, args.target, args.eval_ratio)
