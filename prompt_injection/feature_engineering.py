"""Handcrafted lexical feature engineering for prompt injection detection.

These features capture signals that the sentence-transformer encoder collapses:
  - Obfuscation patterns (base64, unicode escapes, character substitution)
  - Imperative command density (injection = lots of commands)
  - Injection keyword ratio
  - Structural anomalies (excessive special chars, repeated phrases)
  - Encoding/entropy anomalies
  - Instruction hierarchy spoofing (SYSTEM:/USER:/ASSISTANT: delimiters)
  - Token smuggling (zero-width spaces, invisible Unicode, homoglyphs)
  - Gibberish detection (random character strings → benign prior)

The feature vector (LEXICAL_FEATURE_DIM = 25) is concatenated with the
encoder embedding before the MLP head, giving the classifier direct access
to surface-level signals without relying on the frozen encoder.

Exported functions:
  compute_features(text)       → list[float] of length 25
  compute_features_batch(texts) → torch.Tensor of shape (N, 25)
  keyword_override_check(text) → bool  (True if high-confidence injection keyword found)
"""
from __future__ import annotations

import math
import re
import string
import unicodedata
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

# ─── High-confidence keyword triggers for override guard ─────────────────────
# These phrases are unambiguous injection signals. Case-insensitive matching.
# Used by keyword_override_check() in predict.py.
_KEYWORD_OVERRIDE_PATTERNS: list[re.Pattern] = [
    # Direct override commands (multi-word, high precision)
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above|your)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:all\s+)?(?:previous|prior|above|your)?\s*instructions?\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:your|the|all)\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(?:your|the|all|previous)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|your)\b", re.IGNORECASE),
    # Jailbreak markers
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bno\s+restrictions\b", re.IGNORECASE),
    re.compile(r"\buncensored\b", re.IGNORECASE),
    re.compile(r"\bunfiltered\b", re.IGNORECASE),
    re.compile(r"\bunrestricted\b", re.IGNORECASE),
    # Role hijack
    re.compile(r"\bact\s+as\s+(?:an?\s+)?\w", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+to\s+be\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\s+you\b", re.IGNORECASE),
    re.compile(r"\broleplay\s+as\b", re.IGNORECASE),
    re.compile(r"\bimpersonate\b", re.IGNORECASE),
    # Authority injection
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\badmin\s+mode\b", re.IGNORECASE),
    re.compile(r"\bmaintenance\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdebug\s+mode\b", re.IGNORECASE),
    # System prompt exfiltration
    re.compile(r"\breveal\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bprevious\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\binitial\s+(?:system\s+)?prompt\b", re.IGNORECASE),
]


_BASE64_DECODED_KEYWORDS: frozenset[str] = frozenset({
    "instruction", "instructions", "prompt", "prompts", "system", "rule", "rules",
    "filter", "filters", "restriction", "restrictions", "security", "safety",
    "ignore", "disregard", "forget", "bypass", "override", "reveal", "expose",
    "leak", "output", "print", "display", "show", "tell", "repeat", "verbatim",
    "dan", "jailbreak", "sudo", "admin", "unrestricted", "unfiltered", "uncensored",
    "disable"
})


def keyword_override_check(text: str) -> bool:
    """Return True if the text contains a high-confidence injection keyword trigger.

    Used by predict.py to apply a hard injection_prob floor when a known
    attack pattern is detected regardless of the neural model's confidence.
    Matching is case-insensitive.

    Also automatically parses, pads, and base64-decodes any base64 looking blocks
    and checks if the decoded text contains prompt injection keywords or triggers.
    """
    # 1. Direct regex check
    for pattern in _KEYWORD_OVERRIDE_PATTERNS:
        if pattern.search(text):
            return True

    # 2. Base64 decoded check (token smuggling / encoding bypass)
    import base64
    # Find base64-looking sequences (alphanumeric, +, /, optional padding =, length >= 8)
    b64_blocks = re.findall(r"\b[A-Za-z0-9+/]{8,}={0,2}\b|[A-Za-z0-9+/]{12,}={0,2}", text)
    for block in b64_blocks:
        # Standardise padding to multiple of 4
        pad_needed = len(block) % 4
        padded = block
        if pad_needed:
            padded += "=" * (4 - pad_needed)
        try:
            # Validate=True ensures it's strict base64 encoding
            decoded_bytes = base64.b64decode(padded.encode("ascii", errors="ignore"), validate=True)
            decoded_str = decoded_bytes.decode("utf-8", errors="ignore").lower()
            # If the decoded string contains key injection terms or matches the override patterns
            for pattern in _KEYWORD_OVERRIDE_PATTERNS:
                if pattern.search(decoded_str):
                    return True
            # Also check against base64 specific broader keyword set
            decoded_words = set(re.findall(r"\b\w+\b", decoded_str))
            if any(w in _BASE64_DECODED_KEYWORDS for w in decoded_words):
                return True
            # Also check against the main keyword dictionary
            for kw in _INJECTION_KEYWORDS:
                if f" {kw} " in f" {decoded_str} " or decoded_str.startswith(kw) or decoded_str.endswith(kw):
                    return True
        except Exception:
            pass

    return False


# ─── Regex patterns for structural anomalies ─────────────────────────────────
_BASE64_RE     = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_UNICODE_ESC   = re.compile(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}|&#\d+;")
_SPECIAL_DELIM = re.compile(r"<\|[\w\s]+\|>|\[INST\]|\[SYS\]|###|<<<|>>>|\{\{|\}\}")
_URL_RE        = re.compile(r"https?://\S+|www\.\S+")
_CODE_BLOCK    = re.compile(r"```|`[^`]+`|\beval\(|\bexec\(|\bos\.|\bsubprocess\.")
_REPEAT_PHRASE = re.compile(r"(\b\w{4,}\b)(?:\s+\S+){0,5}\s+\1")  # repeated 4+ char word

# f17 — instruction hierarchy spoofing
_ROLE_DELIM_RE = re.compile(
    r"\b(?:SYSTEM|USER|ASSISTANT|HUMAN|AI|GPT|CLAUDE|GEMINI)\s*:",
    re.IGNORECASE,
)

# f18 — nested brackets (indirect injection staging)
_NESTED_BRACKET_RE = re.compile(r"\[\[.{1,200}\]\]|\{\{.{1,200}\}\}")

# f19 — token smuggling: zero-width and invisible Unicode characters
_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u00ad\ufeff\u2060-\u2064\u206a-\u206f]"
)
# Common homoglyph substitutions: Cyrillic а→a, е→e, о→o, etc.
_HOMOGLYPH_RE = re.compile(
    r"[\u0430\u0435\u043e\u0440\u0441\u0445\u0456\u04bb"   # Cyrillic lookalikes
    r"\u03b1\u03b5\u03bf\u03c1\u03c5\u03bd"                 # Greek lookalikes
    r"\u2160-\u2188]"                                        # Roman numeral lookalikes
)

# f23 — prompt framing phrases
_FRAMING_RE = re.compile(
    r"\b(answer\s+the\s+following|given\s+the\s+(?:above|following|context)|"
    r"based\s+on\s+(?:the\s+)?(?:above|following|context)|"
    r"using\s+the\s+(?:above|following|context)|"
    r"according\s+to\s+the\s+(?:above|following)|"
    r"in\s+light\s+of\s+the\s+(?:above|following)|"
    r"considering\s+the\s+(?:above|following))\b",
    re.IGNORECASE,
)

# f24 — ellipsis/separator patterns used to hide injected content
_SEPARATOR_RE = re.compile(r"\.{3,}|---+|===+|_{3,}|\*{3,}|#{3,}")


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


def _gibberish_score(text: str) -> float:
    """Estimate how 'gibberish-like' a text is.

    A pure random string like 'iuhdwciuhcdwiu' scores high.
    Normal English / injection text scores low.

    Heuristics:
      1. Vowel ratio: real words have ~35–45% vowels; random strings have irregular ratio.
      2. Consonant cluster ratio: real words rarely have >3 consecutive consonants.
      3. Token length uniformity: gibberish is often a single long token.
      4. Character n-gram unusualness: real words have common bigrams; gibberish doesn't.
    """
    if not text or len(text) < 3:
        return 0.0

    lowered = text.lower().strip()
    # Only consider alpha characters
    alpha = [c for c in lowered if c.isalpha()]
    if len(alpha) < 3:
        return 0.0

    vowels = set("aeiou")
    vowel_count = sum(1 for c in alpha if c in vowels)
    vowel_ratio = vowel_count / len(alpha)

    # Vowel ratio far from normal English (0.35–0.45) → more suspicious
    normal_low, normal_high = 0.25, 0.55
    vowel_anomaly = 0.0 if normal_low <= vowel_ratio <= normal_high else min(
        abs(vowel_ratio - 0.30), abs(vowel_ratio - 0.50)
    ) * 3.0

    # Consecutive consonant clusters (>3 in a row without vowel)
    max_cons_run = 0
    current_run = 0
    for c in alpha:
        if c not in vowels:
            current_run += 1
            max_cons_run = max(max_cons_run, current_run)
        else:
            current_run = 0
    cluster_score = min(max_cons_run / 8.0, 1.0)  # 8+ consecutive consonants → 1.0

    # Token structure: if the whole text is one long alpha token, suspicious
    tokens = lowered.split()
    if not tokens:
        return 0.0
    single_token_score = 1.0 if len(tokens) == 1 and len(tokens[0]) > 8 else 0.0

    # Character bigram entropy: real words have patterns; random strings are high-entropy
    bigrams = [lowered[i:i+2] for i in range(len(lowered) - 1) if lowered[i:i+2].isalpha()]
    if bigrams:
        bigram_entropy = _char_entropy("".join(bigrams))
        # Very high bigram entropy (>3.5 bits) with short text = suspicious
        bigram_score = min(max(0.0, bigram_entropy - 2.8) / 1.5, 1.0)
    else:
        bigram_score = 0.0

    # Weighted combination
    score = (
        vowel_anomaly * 0.30 +
        cluster_score * 0.30 +
        single_token_score * 0.20 +
        bigram_score * 0.20
    )
    return min(score, 1.0)


def compute_features(text: str) -> list[float]:
    """
    Compute a 25-dimensional handcrafted feature vector for ``text``.

    Features 1-16: original lexical signals (keyword density, entropy, etc.)
    Features 17-24: new structural/obfuscation signals (IMPROVEMENTS.md)
    Feature 25: gibberish score (new — detects random character strings)

    Returns a plain Python list of floats (all values in [0, 1] range).
    """
    if not text:
        return [0.0] * 25

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
    # Map to [0,1] with peaks at long extremes.
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

    # ── Feature 17: Instruction hierarchy spoofing ───────────────────────────
    # Detects SYSTEM:, USER:, ASSISTANT: role delimiters used in injection
    # to override multi-turn context.
    role_delim_count = len(_ROLE_DELIM_RE.findall(text))
    f17_hierarchy_spoof = min(role_delim_count / n_sentences, 1.0)

    # ── Feature 18: Nested brackets ───────────────────────────────────────────
    # [[...]] and {{...}} patterns used in indirect injection staging.
    nested_count = len(_NESTED_BRACKET_RE.findall(text))
    f18_nested_brackets = min(nested_count / n_sentences, 1.0)

    # ── Feature 19: Token smuggling ───────────────────────────────────────────
    # Zero-width spaces, invisible Unicode characters, and homoglyphs.
    zw_count = len(_ZERO_WIDTH_RE.findall(text))
    hg_count = len(_HOMOGLYPH_RE.findall(text))
    f19_token_smuggling = min((zw_count + hg_count) / max(n_chars * 0.01, 1), 1.0)

    # ── Feature 20: Multilingual switch ───────────────────────────────────────
    # Abrupt script change mid-text: count distinct Unicode script blocks.
    scripts: set[str] = set()
    for ch in text:
        try:
            script = unicodedata.name(ch, "").split()[0] if ch.strip() else None
            if script and script not in ("LATIN", "DIGIT", "SPACE", "PUNCTUATION"):
                scripts.add(script)
        except Exception:
            pass
    f20_multilingual = min(len(scripts) / 3.0, 1.0)

    # ── Feature 21: Sentence length variance ──────────────────────────────────
    # High variance = long instruction sentences hidden after short decoy sentences.
    if len(sentences) >= 2:
        sent_lens = [len(s.split()) for s in sentences]
        mean_len = sum(sent_lens) / len(sent_lens)
        variance = sum((l - mean_len) ** 2 for l in sent_lens) / len(sent_lens)
        f21_sent_variance = min(variance / 200.0, 1.0)
    else:
        f21_sent_variance = 0.0

    # ── Feature 22: Number density ────────────────────────────────────────────
    # Dense numeric content correlates with config/parameter overrides.
    num_tokens = sum(1 for t in tokens if any(c.isdigit() for c in t))
    f22_number_density = min(num_tokens / n_tokens, 1.0)

    # ── Feature 23: Prompt framing phrase ─────────────────────────────────────
    # "Answer the following", "Given the above", "Based on this context", etc.
    framing_count = len(_FRAMING_RE.findall(text))
    f23_framing = min(framing_count / n_sentences, 1.0)

    # ── Feature 24: Ellipsis/separator ────────────────────────────────────────
    # "...", "---", "===" separators used to hide injected content after
    # a decoy prefix.
    sep_count = len(_SEPARATOR_RE.findall(text))
    f24_separator = min(sep_count / n_sentences, 1.0)

    # ── Feature 25: Gibberish score ────────────────────────────────────────────
    # Detects random character strings like "iuhdwciuhcdwiu".
    # High score → benign prior (gibberish is almost never a real injection).
    f25_gibberish = _gibberish_score(text)

    return [
        f1_inj_kw_density,    #  0  injection keyword density
        f2_imperative_ratio,  #  1  imperative verb at sentence start
        f3_entropy_inv,       #  2  inverted char entropy (obfuscation signal)
        f4_base64,            #  3  base64-like pattern
        f5_unicode_esc,       #  4  unicode/hex escape density
        f6_special_delim,     #  5  special delimiter density
        f7_upper_ratio,       #  6  uppercase ratio
        f8_punct_density,     #  7  punctuation density
        f9_question_ratio,    #  8  question mark ratio (benign signal, inverted)
        f10_url,              #  9  URL presence
        f11_code,             # 10  code/eval pattern
        f12_repeat,           # 11  repeated phrase (context overflow)
        f13_len_outlier,      # 12  length outlier (very long)
        f14_low_diversity,    # 13  low lexical diversity (repetitive)
        f15_override_pattern, # 14  canonical override pattern
        f16_role_assign,      # 15  role assignment trigger
        f17_hierarchy_spoof,  # 16  instruction hierarchy spoofing
        f18_nested_brackets,  # 17  nested bracket patterns
        f19_token_smuggling,  # 18  zero-width / homoglyph smuggling
        f20_multilingual,     # 19  multilingual script switch
        f21_sent_variance,    # 20  sentence length variance
        f22_number_density,   # 21  number/digit density
        f23_framing,          # 22  prompt framing phrase
        f24_separator,        # 23  ellipsis/separator
        f25_gibberish,        # 24  gibberish score (random strings → benign)
    ]


def compute_features_batch(texts: Sequence[str]) -> torch.Tensor:
    """
    Compute features for a batch of texts.

    Returns a float32 tensor of shape (len(texts), LEXICAL_FEATURE_DIM).
    """
    feats = [compute_features(t) for t in texts]
    return torch.tensor(feats, dtype=torch.float32)
