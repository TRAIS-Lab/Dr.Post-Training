"""Benchmark decontamination used by SFT/data/prepare_datasets.py.

Rule: a candidate user prompt is *blocked*
if its normalised text (NFKC, casefold, collapsed whitespace) equals a reference
prompt exactly, or if at least ``DECONTAM_THRESHOLD`` of its unique word
``DECONTAM_NGRAM``-grams are shared with a single reference prompt. Shorter
prompts (fewer than ``DECONTAM_NGRAM`` words) are matched exactly only.

References are the user prompts of the benchmark files (and, for the pools,
the target splits) that ``prepare_datasets.py`` has already written.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DECONTAM_NGRAM = 8
DECONTAM_THRESHOLD = 0.8   # shared unique n-grams / min(candidate, reference)
_WORD_RE = re.compile(r"\w+")


def norm_prompt(text: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def word_ngrams(normalized: str, n: int = DECONTAM_NGRAM) -> frozenset:
    tokens = _WORD_RE.findall(normalized)
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def user_prompts(messages: Optional[Sequence[Mapping[str, Any]]]) -> List[str]:
    return [
        m.get("content", "") for m in (messages or [])
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]


class PromptDecontaminator:
    """Exact normalised-prompt and word-n-gram near-duplicate blocker."""

    def __init__(self, references: Iterable[Tuple[str, str]] = (),
                 ngram: int = DECONTAM_NGRAM, threshold: float = DECONTAM_THRESHOLD):
        self.ngram, self.threshold = ngram, threshold
        self._refs: List[Tuple[str, frozenset]] = []
        self._exact: Dict[str, int] = {}
        self._inverted: Dict[tuple, set] = {}
        for ref_id, prompt in references:
            self.add(ref_id, prompt)

    def add(self, ref_id: str, prompt: Any) -> None:
        normalized = norm_prompt(prompt)
        if not normalized or normalized in self._exact:
            return
        grams = word_ngrams(normalized, self.ngram)
        index = len(self._refs)
        self._refs.append((ref_id, grams))
        self._exact[normalized] = index
        for g in grams:
            self._inverted.setdefault(g, set()).add(index)

    def __len__(self) -> int:
        return len(self._refs)

    def match(self, prompt: Any) -> Optional[Tuple[str, str, float]]:
        """Return (kind, reference_id, score) or None."""
        normalized = norm_prompt(prompt)
        if not normalized:
            return ("empty", "", 1.0)
        if normalized in self._exact:
            return ("exact", self._refs[self._exact[normalized]][0], 1.0)
        grams = word_ngrams(normalized, self.ngram)
        if not grams:
            return None
        hits: Dict[int, int] = {}
        for g in grams:
            for idx in self._inverted.get(g, ()):
                hits[idx] = hits.get(idx, 0) + 1
        for idx, shared in sorted(hits.items()):
            ref_id, ref_grams = self._refs[idx]
            denom = min(len(grams), len(ref_grams))
            if denom and shared / denom >= self.threshold:
                return (f"near_{self.ngram}gram", ref_id, shared / denom)
        return None

    def match_messages(self, messages) -> Optional[Tuple[str, str, float]]:
        for p in user_prompts(messages):
            m = self.match(p)
            if m is not None:
                return m
        return None

    def blocked_messages(self, messages) -> bool:
        return self.match_messages(messages) is not None
