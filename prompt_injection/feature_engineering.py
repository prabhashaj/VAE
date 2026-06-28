"""Handcrafted lexical feature engineering for prompt injection detection.

These features capture signals that the sentence-transformer encoder collapses:
  - Obfuscation patterns (base64, unicode escapes, character substitution)
  - Imperative command density (injection = lots of commands)
  - Injection keyword ratio
  - Structural anomalies (excessive special chars, repeated phrases)
  - Encoding/entropy anomalies

The feature vector (LEXICAL_FEATURE_DIM = 16) is concatenated with the
encoder embedding before the MLP head, giving the classifier direct access
to surface-level signals without relying on the frozen encoder.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from functools import lru_cache
from typing import Sequence

import torch


# ─── Injection keyword lexicon ────────────────────────────────────────────────
# Covers explicit override commands, jailbreak markers, roleplay hijack tokens,
# and indirect injection triggers.

_INJECTION_KEYWORDS: frozenset[str] = frozenset({
    # Direct override commands
    "ignore", "disregard", "forget", "override", "bypass", "overwrite",
    "cancel", "discard", "delete", "clear", "reset", "dismiss",
    # Jailbreak markers
    "jailbreak", "dan", "stan", "aim", "dev mode", "developer mode",
    "unrestricted", "unfiltered", "uncensored", "no restrictions",
    "without restrictions", "no limits", "no filter",
    # Role hijack
    "act as", "pretend to be", "you are now", "from now on",
    "new persona", "roleplay as", "play the role", "impersonate",
    # System prompt attacks
    "system prompt", "initial prompt", "original prompt", "instructions above",
    "previous instructions", "prior instructions", "above instructions",
    "ignore above", "ignore previous", "ignore all previous",
    # Exfiltration signals
    "reveal", "expose", "leak", "output", "print", "display", "show me",
    "tell me", "repeat", "verbatim", "word for word",
    # Authority injection
    "as your creator", "as your developer", "as your owner",
    "maintenance mode", "debug mode", "sudo", "admin",
    # Indirect injection markers
    "summarize this", "translate this", "process this", "execute",
    "follow these instructions", "new instruction", "updated instruction",
})

_IMPERATIVE_STARTERS: frozenset[str] = frozenset({
    "ignore", "forget", "disregard", "bypass", "override", "pretend",
    "act", "behave", "respond", "answer", "tell", "reveal", "show",
    "print", "output", "repeat", "say", "write", "generate", "create",
    "make", "do", "perform", "execute", "run", "stop", "start",
})

# Regex patterns for structural anomalies
_BASE64_RE     = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_UNICODE_ESC   = re.compile(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}|&#\d+;")
_SPECIAL_DELIM = re.compile(r"<\|[\w\s]+\|>|\[INST\]|\[SYS\]|###|<<<|>>>|\{\{|\}\}")
_URL_RE        = re.compile(r"https?://\S+|www\.\S+")
_CODE_BLOCK    = re.compile(r"```|`[^`]+`|\beval\(|\bexec\(|\bos\.|\bsubprocess\.")
_REPEAT_PHRASE = re.compile(r"(\b\w{4,}\b)(?:\s+\S+){0,5}\s+\1")  # repeated 4+ char word


# ─── Feature computation ──────────────────────────────────────────────────────

def _token_ngrams(text: str, n: int = 1) -> list[str]:
    """Simple whitespace tokenizer, lowercased."""
    tokens = text.lower().split()
    if n == 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution. High = normal; low = obfuscated."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def compute_features(text: str) -> list[float]:
    """
    Compute a 16-dimensional handcrafted feature vector for ``text``.

    Returns a plain Python list of floats (all values in [0, 1] range).
    """
    if not text:
        return [0.0] * 16

    # ── Basic token stats ─────────────────────────────────────────────────────
    tokens      = text.lower().split()
    n_tokens    = max(len(tokens), 1)
    n_chars     = max(len(text), 1)
    sentences   = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    n_sentences = max(len(sentences), 1)

    # ── Feature 1: Injection keyword density ─────────────────────────────────
    # Ratio of tokens matching our injection keyword set.
    # Also counts 2-gram matches for multi-word keywords.
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    inj_hits = sum(1 for t in tokens if t in _INJECTION_KEYWORDS)
    inj_hits += sum(1 for bg in bigrams if bg in _INJECTION_KEYWORDS)
    f1_inj_kw_density = min(inj_hits / n_tokens, 1.0)

    # ── Feature 2: Imperative verb at sentence start ──────────────────────────
    imperative_count = sum(
        1 for s in sentences
        if s.split() and s.split()[0].lower() in _IMPERATIVE_STARTERS
    )
    f2_imperative_ratio = min(imperative_count / n_sentences, 1.0)

    # ── Feature 3: Character entropy (inverted) ────────────────────────────────
    # Normal text has high entropy (~4–5 bits). Obfuscated has low entropy.
    entropy = _char_entropy(text)
    # Invert and normalize: low entropy (obfuscated) → high feature value.
    f3_entropy_inv = max(0.0, 1.0 - entropy / 6.0)

    # ── Feature 4: Base64-like pattern presence ────────────────────────────────
    b64_matches = _BASE64_RE.findall(text)
    b64_char_count = sum(len(m) for m in b64_matches)
    f4_base64 = min(b64_char_count / n_chars, 1.0)

    # ── Feature 5: Unicode / hex escape sequences ──────────────────────────────
    esc_count = len(_UNICODE_ESC.findall(text))
    f5_unicode_esc = min(esc_count / n_tokens, 1.0)

    # ── Feature 6: Special delimiter density ──────────────────────────────────
    delim_count = len(_SPECIAL_DELIM.findall(text))
    f6_special_delim = min(delim_count / n_tokens, 1.0)

    # ── Feature 7: Uppercase ratio ────────────────────────────────────────────
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        f7_upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    else:
        f7_upper_ratio = 0.0

    # ── Feature 8: Punctuation density ────────────────────────────────────────
    punct_chars = sum(1 for c in text if c in string.punctuation)
    f8_punct_density = min(punct_chars / n_chars, 1.0)

    # ── Feature 9: Question mark ratio ────────────────────────────────────────
    # Benign prompts tend to ask; injections tend to command.
    q_marks = text.count("?")
    f9_question_ratio = min(q_marks / n_sentences, 1.0)

    # ── Feature 10: URL presence ───────────────────────────────────────────────
    url_count = len(_URL_RE.findall(text))
    f10_url = min(url_count / n_sentences, 1.0)

    # ── Feature 11: Code/eval pattern ─────────────────────────────────────────
    code_hits = len(_CODE_BLOCK.findall(text))
    f11_code = min(code_hits / n_tokens, 1.0)

    # ── Feature 12: Repeated phrase detection (context overflow) ──────────────
    repeat_hits = len(_REPEAT_PHRASE.findall(text))
    f12_repeat = min(repeat_hits / n_sentences, 1.0)

    # ── Feature 13: Text length (log-normalized) ───────────────────────────────
    # Both very short (<5 tokens) and very long (>300 tokens) are suspicious.
    # Map to [0,1] with peaks at short and long extremes.
    f13_len_outlier = 1.0 - min(n_tokens, 300) / 300.0  # long → high

    # ── Feature 14: Lexical diversity (TTR) ───────────────────────────────────
    # Type-token ratio: low TTR = repetitive = possible padding attack.
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / n_tokens
    f14_low_diversity = 1.0 - min(ttr, 1.0)   # low diversity → high feature

    # ── Feature 15: "Ignore/forget + X + instructions" pattern ───────────────
    # Captures the canonical "Ignore all previous instructions" pattern family.
    ignore_pattern = re.compile(
        r"\b(ignore|forget|disregard|bypass|override)\b.{0,50}"
        r"\b(instructions?|prompt|previous|prior|above|all)\b",
        re.IGNORECASE | re.DOTALL,
    )
    f15_override_pattern = 1.0 if ignore_pattern.search(text) else 0.0

    # ── Feature 16: Role assignment trigger ───────────────────────────────────
    role_pattern = re.compile(
        r"\b(you are|you're|act as|pretend|from now on|your (new )?role|"
        r"you (will|must|should) (now )?be|I want you to)\b",
        re.IGNORECASE,
    )
    f16_role_assign = 1.0 if role_pattern.search(text) else 0.0

    return [
        f1_inj_kw_density,    # 0  injection keyword density
        f2_imperative_ratio,  # 1  imperative verb at sentence start
        f3_entropy_inv,       # 2  inverted char entropy (obfuscation signal)
        f4_base64,            # 3  base64-like pattern
        f5_unicode_esc,       # 4  unicode/hex escape density
        f6_special_delim,     # 5  special delimiter density
        f7_upper_ratio,       # 6  uppercase ratio
        f8_punct_density,     # 7  punctuation density
        f9_question_ratio,    # 8  question mark ratio (benign signal, inverted)
        f10_url,              # 9  URL presence
        f11_code,             # 10 code/eval pattern
        f12_repeat,           # 11 repeated phrase (context overflow)
        f13_len_outlier,      # 12 length outlier (very long)
        f14_low_diversity,    # 13 low lexical diversity (repetitive)
        f15_override_pattern, # 14 canonical override pattern
        f16_role_assign,      # 15 role assignment trigger
    ]


def compute_features_batch(texts: Sequence[str]) -> torch.Tensor:
    """
    Compute features for a batch of texts.

    Returns a float32 tensor of shape (len(texts), LEXICAL_FEATURE_DIM).
    """
    feats = [compute_features(t) for t in texts]
    return torch.tensor(feats, dtype=torch.float32)
