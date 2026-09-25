"""Thin wrapper around the vendored official IFEval verifier.

Exposes one call used by both sides of the pipeline:

* data preparation, to re-verify support-pool responses before they become
  target-gradient examples, and
* downstream evaluation, to score generated responses.

Using the same code path for both is deliberate — a target set "verified" by a
different scorer than the one reporting the metric is not verified at all.

The strict/loose response variants replicate
``instruction_following_eval.evaluation_lib.test_instruction_following_{strict,loose}``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence

_NLTK_LOCK = threading.Lock()
_NLTK_READY = False

# Sentence counting in instructions_util needs the Punkt models. nltk >= 3.9
# reads punkt_tab; older releases read the punkt pickle. Fetch both once.
_NLTK_PACKAGES = ("punkt", "punkt_tab")


def ensure_nltk_data() -> None:
    """Download the Punkt tokenizer models once per process, if missing."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    with _NLTK_LOCK:
        if _NLTK_READY:
            return
        import nltk

        for package in _NLTK_PACKAGES:
            try:
                nltk.data.find(f"tokenizers/{package}")
            except LookupError:
                nltk.download(package, quiet=True)
        _NLTK_READY = True


def loose_response_variants(response: str) -> List[str]:
    """The eight response forms the official loose metric accepts."""
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def evaluate_instruction_following(
    *,
    prompt: str,
    response: str,
    instruction_id_list: Sequence[str],
    kwargs_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Score one (prompt, response) pair against its instruction list.

    Returns ``follow_instruction_list`` (one bool per instruction) and
    ``follow_all_instructions``. An unknown instruction id counts as not
    followed rather than raising, so one unsupported constraint cannot abort a
    541-prompt evaluation.
    """
    ensure_nltk_data()
    from SFT.eval.tasks.ifeval_lib import instructions_registry

    kwargs_list = list(kwargs_list or [])
    candidates = [response] if strict else loose_response_variants(response)

    follow_instruction_list: List[bool] = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            follow_instruction_list.append(False)
            continue

        instruction = instruction_cls(instruction_id)
        instruction_kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        try:
            instruction.build_description(**(instruction_kwargs or {}))
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
        except Exception:  # noqa: BLE001 - a malformed constraint is "not followed"
            follow_instruction_list.append(False)
            continue

        is_following = False
        for candidate in candidates:
            if not candidate.strip():
                continue
            try:
                if instruction.check_following(candidate):
                    is_following = True
                    break
            except Exception:  # noqa: BLE001 - checker crash is "not followed"
                continue
        follow_instruction_list.append(is_following)

    return {
        "instruction_id_list": list(instruction_id_list),
        "follow_instruction_list": follow_instruction_list,
        "follow_all_instructions": bool(follow_instruction_list)
        and all(follow_instruction_list),
    }


def aggregate(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Prompt-level and instruction-level accuracy over scored examples."""
    prompt_total = len(results)
    prompt_correct = sum(1 for r in results if r["follow_all_instructions"])
    instruction_flags = [flag for r in results for flag in r["follow_instruction_list"]]
    instruction_total = len(instruction_flags)
    instruction_correct = sum(1 for flag in instruction_flags if flag)
    return {
        "prompt_level_acc": (prompt_correct / prompt_total * 100.0) if prompt_total else 0.0,
        "inst_level_acc": (
            instruction_correct / instruction_total * 100.0
        ) if instruction_total else 0.0,
        "n_prompts": prompt_total,
        "n_instructions": instruction_total,
    }
