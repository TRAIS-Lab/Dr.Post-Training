"""MATH-500 greedy pass@1 evaluation.

Records come from ``eval/math500/math500_bench_data.jsonl`` (build with
``prepare_datasets.py --datasets math500``); the legacy
``math500_test_data.jsonl`` split is accepted as a fallback. Prompts are rendered
through the tokenizer's chat template.

Scoring uses `math-verify <https://github.com/huggingface/Math-Verify>`_ when it
is installed (the Next repo pins 0.9.0); otherwise it falls back to the original
regex ``\\boxed{}`` extraction with light LaTeX normalisation, and says so in the
result's provenance. Accuracy is reported in percent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from SFT.eval.tasks.bench_data import bench_data_path, first_user_content, load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    result_provenance,
    single_source_revision,
)
from ..utils import generate_completions, get_eos_token_ids

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 4096
DATASET_REPOSITORY = "HuggingFaceH4/MATH-500"
PREFERRED_MATH_VERIFY_VERSION = "0.9.0"

_PROMPT_TEMPLATE = (
    "Solve the following mathematics problem. Show your reasoning, then put only "
    "the final answer inside \\boxed{{}}.\n\n{problem}"
)


# ---------------------------------------------------------------------------
# Legacy regex scorer (kept as the fallback when math-verify is unavailable)
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the *last* ``\\boxed{...}``, handling nested braces."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    answer = answer.strip()
    for token in ("\\,", "\\;", "\\!", "\\ ", "\\text{", "\\mathrm{", "\\textbf{", "\\left", "\\right", "$"):
        answer = answer.replace(token, "")
    # Normalise fractions before stripping braces (the legacy code did it after,
    # so the pattern could never match).
    answer = re.sub(r"\\[dt]?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", answer)
    answer = answer.replace("{", "").replace("}", "")
    answer = answer.replace(" ", "")
    return answer.lower()


def answers_match(pred: Optional[str], ref: str) -> bool:
    pred_norm, ref_norm = normalize_answer(pred), normalize_answer(ref)
    if pred_norm == ref_norm:
        return True
    try:
        return abs(float(pred_norm) - float(ref_norm)) < 1e-6
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Scorer selection
# ---------------------------------------------------------------------------

def _load_scorer() -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Return ``score(gold, prediction) -> {correct, status, error}`` and its provenance."""
    try:
        import importlib.metadata
        from math_verify import parse, verify  # type: ignore

        version = importlib.metadata.version("math-verify")
        if version != PREFERRED_MATH_VERIFY_VERSION:
            logger.warning("math-verify %s installed; Next campaigns used %s", version, PREFERRED_MATH_VERIFY_VERSION)

        def score(gold: str, prediction: str) -> Dict[str, Any]:
            try:
                parsed_gold = parse(f"${gold}$")
            except Exception as exc:  # noqa: BLE001
                return {"correct": False, "status": "gold_parse_error", "error": f"{type(exc).__name__}: {exc}"}
            if not parsed_gold:
                return {"correct": False, "status": "gold_parse_error", "error": "empty parsed gold"}
            try:
                parsed_pred = parse(prediction)
            except Exception as exc:  # noqa: BLE001
                return {"correct": False, "status": "prediction_parse_error", "error": f"{type(exc).__name__}: {exc}"}
            if not parsed_pred:
                return {"correct": False, "status": "prediction_parse_error", "error": "empty parsed prediction"}
            try:
                correct = bool(verify(parsed_gold, parsed_pred))
            except Exception as exc:  # noqa: BLE001
                return {"correct": False, "status": "verification_error", "error": f"{type(exc).__name__}: {exc}"}
            return {"correct": correct, "status": "correct" if correct else "incorrect", "error": ""}

        return score, {"package": "math-verify", "version": version}
    except ImportError:
        logger.warning(
            "math-verify is not installed; falling back to regex \\boxed{} matching. "
            "Install math-verify==%s for the official-style scorer.", PREFERRED_MATH_VERIFY_VERSION,
        )

        def score(gold: str, prediction: str) -> Dict[str, Any]:
            boxed = extract_boxed_answer(prediction)
            if boxed is None:
                return {"correct": False, "status": "prediction_parse_error", "error": "no \\boxed{} found"}
            correct = answers_match(boxed, gold)
            return {"correct": correct, "status": "correct" if correct else "incorrect", "error": ""}

        return score, {"package": "regex_boxed_fallback", "version": "legacy"}


def _load_records(data_dir: str, k: int) -> Tuple[List[Dict[str, Any]], str]:
    """Benchmark file first; legacy ``math500_test_data.jsonl`` as fallback."""
    if os.path.exists(bench_data_path(data_dir, "math500")):
        return load_bench_records(data_dir, "math500", k=k), "bench"
    legacy = os.path.join(data_dir, "eval", "math500", "math500_test_data.jsonl")
    if not os.path.exists(legacy):
        raise FileNotFoundError(
            f"MATH500 data not found: {bench_data_path(data_dir, 'math500')} or {legacy}\n"
            "Build it with: python SFT/data/prepare_datasets.py --datasets math500"
        )
    logger.warning("Using legacy split %s (a subset of MATH-500); build the bench file for the full set", legacy)
    records: List[Dict[str, Any]] = []
    with open(legacy, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row.get("messages", [])
            if len(messages) < 2:
                continue
            metadata = dict(row.get("metadata", {}))
            metadata.setdefault("problem", messages[0]["content"])
            if not metadata.get("answer"):
                metadata["answer"] = extract_boxed_answer(messages[1]["content"])
            if metadata["answer"] is None:
                continue
            records.append({"id": row.get("id"), "messages": [messages[0]], "metadata": metadata})
            if k > 0 and len(records) >= k:
                break
    return records, "legacy_test_split"


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    data_dir = getattr(args, "data_dir", "./data")
    requested_n = getattr(args, "n_test", -1)
    score, scorer_info = _load_scorer()
    records, source = _load_records(data_dir, requested_n)
    print(f"Loaded {len(records)} MATH500 problems ({source}); scorer={scorer_info['package']}")

    prompts = []
    for record in records:
        metadata = record.get("metadata", {})
        problem = metadata.get("problem") or first_user_content(record)
        prompts.append(render_generation_chat(
            tokenizer, _PROMPT_TEMPLATE.format(problem=problem), enable_thinking=False,
        ))
    generations = generate_completions(
        model, tokenizer, prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False, disable_tqdm=False,
    )

    per_example: List[Dict[str, Any]] = []
    level_correct: Dict[Any, int] = {}
    level_total: Dict[Any, int] = {}
    for record, generation in zip(records, generations):
        metadata = record.get("metadata", {})
        gold = metadata.get("answer")
        if gold is None:
            raise ValueError(f"MATH500 record has no metadata.answer: {record.get('id')}")
        response = clean_model_response(generation)
        scored = score(str(gold), response)
        level = metadata.get("level", "unknown")
        level_total[level] = level_total.get(level, 0) + 1
        if scored["correct"]:
            level_correct[level] = level_correct.get(level, 0) + 1
        per_example.append({
            "id": record.get("id"), "unique_id": metadata.get("unique_id"),
            "gold_answer": gold, "raw_generation": generation, "response": response, **scored,
        })

    n_correct = sum(1 for row in per_example if row["correct"])
    n_parse_errors = sum(1 for row in per_example if "parse_error" in row["status"])
    accuracy = n_correct / len(records) * 100.0 if records else 0.0

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        with open(os.path.join(output_dir, "math500_generations.jsonl"), "w", encoding="utf-8") as handle:
            for row in per_example:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nMATH500 Results:")
    print(f"  accuracy: {accuracy:.2f}%  ({n_correct}/{len(records)}), parse errors: {n_parse_errors}")
    if len(level_total) > 1:
        for level in sorted(level_total, key=str):
            print(f"    level {level}: {level_correct.get(level, 0)}/{level_total[level]}")

    return {
        "evaluation_scope": "full" if requested_n <= 0 else "limited",
        "accuracy": accuracy,
        "n_test": len(records),
        "n_correct": n_correct,
        "n_parse_errors": n_parse_errors,
        "level_breakdown": {
            str(level): level_correct.get(level, 0) / level_total[level] * 100.0 for level in level_total
        },
        "provenance": result_provenance(
            dataset_repository=DATASET_REPOSITORY,
            dataset_revision=single_source_revision(records),
            dataset_split="test",
            evaluator={**scorer_info, "primary_metric": "accuracy", "records_source": source},
            n_tasks=len(records), max_new_tokens=max_new_tokens, thinking=False,
        ),
    }
