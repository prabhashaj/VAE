"""Stage 1: Structural Filter — fast regex/heuristic checks (< 5ms).

Catches obvious injection patterns before they reach the VAE.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Compiled patterns — ordered roughly by severity
# ────────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # Role hijacking
    (re.compile(
        r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|your)\s+"
        r"(?:instructions?|prompts?|rules?|guidelines?|directives?)",
        re.IGNORECASE,
    ), "role_hijack", 0.9),
    (re.compile(
        r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|unrestricted|unfiltered)\s+(?:AI|assistant|bot)"
        r"|you\s+are\s+now\s+(?:an?\s+)?\w*\s*(?:AI|assistant|bot)",
        re.IGNORECASE,
    ), "role_hijack", 0.85),
    (re.compile(
        r"(?:act|behave|pretend|operate)\s+(?:as\s+)?(?:if|like)\s+.*?(?:no|without)\s+"
        r"(?:restrictions?|rules?|filters?|safety|guidelines?|constraints?)",
        re.IGNORECASE,
    ), "role_hijack", 0.85),

    # System prompt extraction
    (re.compile(
        r"(?:show|reveal|display|print|output|share|tell)\s+(?:me\s+)?(?:your\s+)?"
        r"(?:system\s+)?(?:prompt|instructions?|directives?|configuration)",
        re.IGNORECASE,
    ), "prompt_leak", 0.8),
    (re.compile(
        r"(?:what|repeat)\s+(?:is|are|were)\s+(?:your\s+)?"
        r"(?:system\s+)?(?:prompt|instructions?|initial\s+(?:instructions?|message))",
        re.IGNORECASE,
    ), "prompt_leak", 0.7),
    (re.compile(
        r"text\s+(?:above|before)\s+(?:this|the)\s+(?:line|message)",
        re.IGNORECASE,
    ), "prompt_leak", 0.75),

    # Delimiter injection
    (re.compile(
        r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]",
        re.IGNORECASE,
    ), "delimiter_injection", 0.9),
    (re.compile(
        r"\[SYSTEM\].*?\[/SYSTEM\]",
        re.IGNORECASE | re.DOTALL,
    ), "delimiter_injection", 0.85),

    # Developer/override mode
    (re.compile(
        r"(?:developer|admin|debug|maintenance|testing)\s+mode",
        re.IGNORECASE,
    ), "override_attempt", 0.7),
    (re.compile(
        r"(?:override|bypass|disable|turn\s+off)\s*[:=]?\s*(?:safety|filter|content\s+policy)",
        re.IGNORECASE,
    ), "override_attempt", 0.85),

    # Code/payload injection
    (re.compile(
        r"<script[^>]*>.*?</script>",
        re.IGNORECASE | re.DOTALL,
    ), "payload_injection", 0.9),
    (re.compile(
        r"__import__\s*\(",
        re.IGNORECASE,
    ), "payload_injection", 0.9),
    (re.compile(
        r"(?:DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO)\s+",
        re.IGNORECASE,
    ), "payload_injection", 0.85),

    # Encoding tricks
    (re.compile(
        r"(?:base64|rot13|hex)\s*(?:decode|encode|:)",
        re.IGNORECASE,
    ), "encoding_trick", 0.6),
]


@dataclass
class FilterMatch:
    """A single pattern match."""

    pattern_name: str
    severity: float
    matched_text: str


@dataclass
class StructuralResult:
    """Result from the structural filter stage."""

    passed: bool
    score: float  # 0.0 = clean, 1.0 = definitely malicious
    matches: list[FilterMatch] = field(default_factory=list)
    reason: str = ""


class StructuralFilter:
    """Fast regex-based structural analysis of prompt text.

    Parameters
    ----------
    max_length : int
        Maximum allowed prompt length in characters.
    block_threshold : float
        Score at or above which the prompt is blocked (0.0–1.0).
    """

    def __init__(self, max_length: int = 2048, block_threshold: float = 0.8) -> None:
        self.max_length = max_length
        self.block_threshold = block_threshold

    def check(self, text: str) -> StructuralResult:
        """Run all structural checks on the given text."""
        matches: list[FilterMatch] = []

        # ── Length check ────────────────────────────────────────────
        if len(text) > self.max_length:
            matches.append(FilterMatch(
                pattern_name="excessive_length",
                severity=0.5,
                matched_text=f"length={len(text)} > max={self.max_length}",
            ))

        # ── Empty / whitespace ──────────────────────────────────────
        if not text.strip():
            return StructuralResult(passed=True, score=0.0, reason="empty input")

        # ── Token density (ratio of special chars) ──────────────────
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        density = special_count / max(len(text), 1)
        if density > 0.4:
            matches.append(FilterMatch(
                pattern_name="high_special_char_density",
                severity=0.5,
                matched_text=f"density={density:.2f}",
            ))

        # ── Excessive newlines (delimiter stuffing) ─────────────────
        newline_ratio = text.count("\n") / max(len(text), 1)
        if newline_ratio > 0.15:
            matches.append(FilterMatch(
                pattern_name="excessive_newlines",
                severity=0.4,
                matched_text=f"newline_ratio={newline_ratio:.2f}",
            ))

        # ── Regex patterns ──────────────────────────────────────────
        for pattern, name, severity in _PATTERNS:
            found = pattern.search(text)
            if found:
                matches.append(FilterMatch(
                    pattern_name=name,
                    severity=severity,
                    matched_text=found.group()[:100],  # truncate for safety
                ))

        # ── Compute aggregate score ─────────────────────────────────
        if not matches:
            return StructuralResult(passed=True, score=0.0, reason="no matches")

        # Take the maximum severity as the aggregate score
        score = max(m.severity for m in matches)

        passed = score < self.block_threshold
        reason = f"highest_severity={score:.2f}, patterns={[m.pattern_name for m in matches]}"

        if not passed:
            logger.info("Structural filter BLOCKED: %s", reason)

        return StructuralResult(
            passed=passed,
            score=score,
            matches=matches,
            reason=reason,
        )
