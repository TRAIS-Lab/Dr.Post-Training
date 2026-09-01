"""IFBench evaluation through the official AllenAI verifier (external checkout).

Generation happens here; scoring is delegated to a local checkout of
``allenai/IFBench`` (``python -m run_eval``) so its 58 out-of-distribution
constraint implementations are not approximated with the older IFEval registry.

Point ``--ifbench_repo`` (or ``DRPT_IFBENCH_REPO``) at the checkout. The commit
is compared against ``DEFAULT_IFBENCH_REVISION`` and a mismatch is logged as a
warning; pass ``--ifbench_revision`` to pin a different commit. Primary metric:
prompt-level loose accuracy (the IFBench reporting convention).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from SFT.eval.tasks.bench_data import first_user_content, load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    result_provenance,
    single_source_revision,
)
from SFT.eval.tasks.ifeval_scoring import aggregate
from ..utils import generate_completions, get_eos_token_ids

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 2048
DATASET_REPOSITORY = "allenai/IFBench_test"
EVALUATOR_REPOSITORY = "https://github.com/allenai/IFBench"
DEFAULT_IFBENCH_REVISION = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
REQUIRED_FILES = ("run_eval.py", "evaluation_lib.py", "instructions_registry.py", "instructions.py")


def registry_instruction_ids(registry_path: str) -> Set[str]:
    """Read literal keys of INSTRUCTION_DICT from the official registry without importing it."""
    path = Path(registry_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "INSTRUCTION_DICT" for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if keys:
                return keys
        break
    raise RuntimeError(f"Could not find a literal INSTRUCTION_DICT in {path}")


def validate_instruction_coverage(records: Iterable[Mapping[str, Any]], registry_ids: Set[str]) -> None:
    required = {
        str(instruction_id)
        for record in records
        for instruction_id in record.get("metadata", {}).get("instruction_id_list", [])
    }
    missing = sorted(required - registry_ids)
    if missing:
        raise RuntimeError(
            "IFBench checkout does not implement all constraints in the benchmark file: "
            + ", ".join(missing)
        )


def _git_revision(repo_path: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def resolve_checkout(repo_path: str, expected_revision: str) -> str:
    if not repo_path:
        raise RuntimeError(
            "IFBench requires --ifbench_repo (or DRPT_IFBENCH_REPO) pointing to a checkout of "
            f"{EVALUATOR_REPOSITORY} (git clone {EVALUATOR_REPOSITORY}; git checkout {DEFAULT_IFBENCH_REVISION})"
        )
    for filename in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(repo_path, filename)):
            raise RuntimeError(f"Invalid IFBench checkout; missing {filename}: {repo_path}")
    actual = _git_revision(repo_path)
    if expected_revision and actual != expected_revision:
        logger.warning(
            "IFBench checkout at %s is at %s, expected %s; numbers may not match other runs",
            repo_path, actual, expected_revision,
        )
    return actual


def _write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def official_output_paths(output_dir: str, response_path: str) -> Tuple[str, str]:
    """``run_eval`` names its outputs after the response file's basename."""
    basename = os.path.basename(response_path)
    model_name = basename.replace("-responses.jsonl", "") if "-responses.jsonl" in basename else os.path.splitext(basename)[0]
    return (
        os.path.join(output_dir, f"{model_name}-eval_results_strict.jsonl"),
        os.path.join(output_dir, f"{model_name}-eval_results_loose.jsonl"),
    )


def aggregate_official_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return aggregate([
        {
            "follow_all_instructions": bool(row.get("follow_all_instructions", False)),
            "follow_instruction_list": [bool(v) for v in row.get("follow_instruction_list", [])],
        }
        for row in rows
    ])


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    data_dir = getattr(args, "data_dir", "./data")
    n_test = getattr(args, "n_test", -1)
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        raise ValueError("IFBench requires args.output_dir for the official verifier's files")

    repo_path = getattr(args, "ifbench_repo", None) or os.environ.get("DRPT_IFBENCH_REPO", "")
    expected_revision = getattr(args, "ifbench_revision", None) or os.environ.get(
        "DRPT_IFBENCH_REVISION", DEFAULT_IFBENCH_REVISION
    )
    evaluator_revision = resolve_checkout(repo_path, expected_revision)

    records = load_bench_records(data_dir, "ifbench", k=n_test)
    validate_instruction_coverage(
        records, registry_instruction_ids(os.path.join(repo_path, "instructions_registry.py"))
    )
    print(f"Loaded {len(records)} IFBench prompts")
    prompts = [
        render_generation_chat(tokenizer, first_user_content(record), enable_thinking=False)
        for record in records
    ]
    generations = generate_completions(
        model, tokenizer, prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False, disable_tqdm=False,
    )
    responses = [clean_model_response(generation) for generation in generations]

    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.abspath(os.path.join(output_dir, "ifbench_official_input.jsonl"))
    response_path = os.path.abspath(os.path.join(output_dir, "ifbench-responses.jsonl"))
    official_inputs, official_responses, audit_rows = [], [], []
    for record, generation, response in zip(records, generations, responses):
        metadata = record.get("metadata", {})
        prompt = metadata.get("prompt") or first_user_content(record)
        official_inputs.append({
            "key": metadata.get("key", record.get("id")),
            "prompt": prompt,
            "instruction_id_list": metadata.get("instruction_id_list", []),
            "kwargs": metadata.get("kwargs", []),
        })
        official_responses.append({"prompt": prompt, "response": response})
        audit_rows.append({
            "id": record.get("id"), "key": metadata.get("key", record.get("id")),
            "prompt": prompt, "raw_generation": generation, "response": response,
        })
    _write_jsonl(input_path, official_inputs)
    _write_jsonl(response_path, official_responses)
    _write_jsonl(os.path.join(output_dir, "ifbench_generations.jsonl"), audit_rows)

    command = [
        getattr(args, "ifbench_python", None) or sys.executable, "-m", "run_eval",
        f"--input_data={input_path}", f"--input_response_data={response_path}",
        f"--output_dir={os.path.abspath(output_dir)}",
    ]
    try:
        subprocess.run(command, cwd=repo_path, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Official IFBench evaluator failed: {' '.join(command)}") from exc

    strict_path, loose_path = official_output_paths(output_dir, response_path)
    strict_rows, loose_rows = _read_jsonl(strict_path), _read_jsonl(loose_path)
    if len(strict_rows) != len(records) or len(loose_rows) != len(records):
        raise RuntimeError(
            f"IFBench evaluator returned incomplete output: expected {len(records)}, "
            f"strict={len(strict_rows)}, loose={len(loose_rows)}"
        )
    strict_scores = aggregate_official_rows(strict_rows)
    loose_scores = aggregate_official_rows(loose_rows)

    print("\nIFBench Results:")
    print(f"  strict prompt-level: {strict_scores['prompt_level_acc']:.2f}  "
          f"loose prompt-level: {loose_scores['prompt_level_acc']:.2f}  n={len(records)}")

    return {
        "evaluation_scope": "full" if n_test <= 0 else "limited",
        "prompt_level_strict_acc": strict_scores["prompt_level_acc"],
        "inst_level_strict_acc": strict_scores["inst_level_acc"],
        "prompt_level_loose_acc": loose_scores["prompt_level_acc"],
        "inst_level_loose_acc": loose_scores["inst_level_acc"],
        # IFBench reporting convention: prompt-level loose (percent).
        "accuracy": loose_scores["prompt_level_acc"],
        "n_test": len(records),
        "n_instructions": int(loose_scores["n_instructions"]),
        "provenance": result_provenance(
            dataset_repository=DATASET_REPOSITORY,
            dataset_revision=single_source_revision(records),
            dataset_split="train",
            evaluator={
                "repository": EVALUATOR_REPOSITORY, "revision": evaluator_revision,
                "entrypoint": "python -m run_eval", "primary_metric": "prompt_level_loose_acc",
            },
            n_tasks=len(records), max_new_tokens=max_new_tokens, thinking=False,
        ),
    }
