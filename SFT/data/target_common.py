"""Small dependency-free helpers shared by the rewrite-target pipeline.

Kept free of torch / transformers / eval imports so that stage 1
(``gen_target_candidates.py``, which may run in a vLLM environment) and stage 2
(``build_rewrite_target.py``, which runs in the training environment) can both
import it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, List, Mapping

# ``<base>_gen<tag>`` = answers re-solved from scratch by a generator model and verified,
# ``<base>_rw<tag>``  = reference answers rewritten by a generator model and verified.
# The base target supplies the prompts, the reference answers used for verification
# and (via SFT/eval/eval.py) the benchmarks.
TARGET_VARIANT_RE = re.compile(r"_(gen|rw)[a-z0-9]*$")

MODE_ALIASES = {"gen": "solve", "solve": "solve", "rw": "rewrite", "rewrite": "rewrite"}
MODE_TAGS = {"solve": "gen", "rewrite": "rw"}


def base_target_name(target: str) -> str:
    """``precise_if_gen32b`` -> ``precise_if``; names without a variant suffix are unchanged."""
    return TARGET_VARIANT_RE.sub("", target)


def infer_domain(target: str) -> str:
    """Map a target name to its verifier domain: ``if`` | ``math`` | ``code``."""
    base = base_target_name(target)
    if base.startswith("precise_if") or base.startswith("ifeval"):
        return "if"
    if base.startswith("mbpp") or "code" in base:
        return "code"
    if base.startswith("math") or base.startswith("gsm"):
        return "math"
    raise ValueError(f"cannot infer the verifier domain of target {target!r}")


def canonical_mode(mode: str) -> str:
    try:
        return MODE_ALIASES[mode.lower()]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODE_ALIASES)}") from None


def user_prompt(row: Mapping[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError(f"target row {row.get('id')!r} has no user message")


def assistant_reference(row: Mapping[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError(f"target row {row.get('id')!r} has no assistant message")


def target_split_path(data_dir: str, target: str, split: str) -> str:
    return os.path.join(data_dir, "eval", target, f"{target}_{split}_data.jsonl")


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write atomically (tmp file + rename) so an interrupted job never leaves a truncated file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
