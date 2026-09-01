"""Shared helpers for the generative downstream benchmarks (IFEval, IFBench, MATH500, MBPP+).

These benchmarks differ from the legacy tasks (SamSum, TyDiQA, ...) in two ways:

* they are read from ``eval/<task>/<task>_bench_data.jsonl`` (prompt + scoring
  metadata, no reference used for loss), see ``bench_data.py``;
* prompts are rendered through the tokenizer's chat template so the same
  code path serves Qwen3 (native template) and Llama (tulu fallback).
"""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any, Dict, Iterable, Mapping, Optional

from SFT.data.chat_format import render_generation_prompt

RESULT_SCHEMA_VERSION = "drpt.eval.result.v1"

_CHAT_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|eot_id|>",
    "<|end|>",
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def clean_model_response(text: str, *, strip_thinking: bool = True) -> str:
    """Remove reasoning blocks and chat-template residue from a completion.

    Prompts are rendered with thinking disabled, but a checkpoint can still emit
    a ``<think>...</think>`` block or a stray ``</think>``. Only model-control
    markup is removed; punctuation and markdown are preserved because IFEval
    constraints depend on them.
    """
    out = text or ""
    if strip_thinking:
        out = _THINK_BLOCK_RE.sub("", out)
        if "</think>" in out.lower():
            out = re.split(r"</think>", out, flags=re.IGNORECASE)[-1]
        elif re.match(r"^\s*<think>", out, flags=re.IGNORECASE):
            return ""
    for marker in _CHAT_MARKERS:
        index = out.find(marker)
        if index != -1:
            out = out[:index]
    return out.strip()


def render_generation_chat(tokenizer: Any, user_content: str, *, enable_thinking: bool = False) -> str:
    """Render one benchmark prompt with the active chat template."""
    return render_generation_prompt(tokenizer, user_content, enable_thinking=enable_thinking)


def require_distribution_version(distribution: str, expected: Optional[str] = None) -> str:
    """Return an installed dependency version; fail clearly if missing or pinned-and-drifted."""
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        hint = f"=={expected}" if expected else ""
        raise RuntimeError(
            f"{distribution}{hint} is required for this evaluator; "
            f"install it in the active environment (pip install {distribution}{hint})"
        ) from exc
    if expected and actual != expected:
        raise RuntimeError(
            f"{distribution}=={expected} is required, but version {actual} is installed"
        )
    return actual


def single_source_revision(records: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """Extract one dataset revision from benchmark record metadata (None if absent)."""
    revisions = {
        str(record.get("metadata", {}).get("source_revision"))
        for record in records
        if record.get("metadata", {}).get("source_revision")
    }
    if len(revisions) > 1:
        raise ValueError(f"Benchmark records contain multiple source revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def result_provenance(
    *,
    dataset_repository: str,
    dataset_revision: Optional[str],
    dataset_split: str,
    evaluator: Mapping[str, Any],
    n_tasks: int,
    max_new_tokens: int,
    thinking: bool = False,
) -> Dict[str, Any]:
    """Provenance block stored in every benchmark result JSON."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": {
            "repository": dataset_repository,
            "revision": dataset_revision,
            "split": dataset_split,
            "n_tasks": int(n_tasks),
        },
        "evaluator": dict(evaluator),
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": int(max_new_tokens),
            "thinking": bool(thinking),
            "thinking_output_stripped": True,
        },
    }
