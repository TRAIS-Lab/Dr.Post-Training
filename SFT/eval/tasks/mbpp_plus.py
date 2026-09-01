"""MBPP+ generation and official EvalPlus pass@1 evaluation.

Records come from ``eval/mbpp_plus/mbpp_plus_bench_data.jsonl`` (built with
``prepare_datasets.py --datasets mbpp_plus``, which needs the ``evalplus``
package on the host). Generation happens in-process; scoring runs
``evalplus.evaluate`` in one of two ways:

* ``apptainer`` (default when available): the pinned official EvalPlus image,
  ``--containall`` with the run directory bound at ``/workspace``. Generated
  code never executes on the host. If ``--evalplus_dataset_path`` (or
  ``DRPT_EVALPLUS_DATASET_PATH``) points at a local MbppPlus JSONL it is bound
  read-only and networking is disabled; otherwise the container fetches it.
* ``host``: ``python -m evalplus.evaluate`` in the current environment.
  This executes model-generated code on the host; use only in a throwaway env.

Primary metric: ``base_plus_extra_pass_at_1`` (percent), i.e. pass on both the
original MBPP tests and the EvalPlus extra tests.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from SFT.eval.tasks.bench_data import load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    result_provenance,
    single_source_revision,
)
from ..utils import generate_completions, get_eos_token_ids

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 2048
DATASET_REPOSITORY = "evalplus/mbppplus"
MBPP_PLUS_DATASET_VERSION = "v0.2.0"
# Official EvalPlus v0.3.1 image, digest-pinned (same pin as the Next campaigns).
DEFAULT_EVALPLUS_IMAGE = (
    "docker://ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
)
SAMPLES_FILENAME = "mbpp_plus_samples.jsonl"

_PROMPT_TEMPLATE = (
    "Complete this MBPP+ task. Return one complete, self-contained Python solution "
    "including the required function signature. Do not include prose outside the code.\n\n{prompt}"
)
_FENCED_PYTHON_RE = re.compile(r"```(?:python|py)\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)
_FENCED_ANY_RE = re.compile(r"```\s*\n(.*?)(?:```|\Z)", re.DOTALL)
_BEGIN_DONE_RE = re.compile(r"\[BEGIN\](.*?)(?:\[DONE\]|\Z)", re.DOTALL)
_FUNCTION_RE = re.compile(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(")


_CHAT_MARKERS = ("<|im_end|>", "<|im_start|>", "<|user|>", "<|assistant|>", "<|system|>", "<|eot_id|>", "<|end|>")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_markup_keep_indent(text: str) -> str:
    """Drop think blocks and chat residue but keep leading indentation (Python bodies need it)."""
    out = _THINK_BLOCK_RE.sub("", text or "")
    if "</think>" in out.lower():
        out = re.split(r"</think>", out, flags=re.IGNORECASE)[-1]
    for marker in _CHAT_MARKERS:
        index = out.find(marker)
        if index != -1:
            out = out[:index]
    return out.lstrip("\r\n").rstrip()


def extract_code(text: str) -> str:
    """Recover a Python program from a free-form generation (fenced block, [BEGIN]/[DONE], or raw).

    Leading indentation of the first line is preserved so a body-only completion
    can be appended to the prompt's ``def`` line. A complete function is dedented.
    """
    if not text:
        return ""
    code = None
    for pattern in (_FENCED_PYTHON_RE, _FENCED_ANY_RE, _BEGIN_DONE_RE):
        match = pattern.search(text)
        if match and match.group(1).strip():
            code = match.group(1)
            break
    if code is None:
        code = text
    code = code.lstrip("\r\n").rstrip()
    if _FUNCTION_RE.search(code):
        code = textwrap.dedent(code)
    return code


def assemble_solution(problem_prompt: str, generation: str) -> str:
    """EvalPlus's self-contained ``solution`` field."""
    code = extract_code(_strip_markup_keep_indent(generation))
    if _FUNCTION_RE.search(code):
        return code
    # The model emitted only a body; append it to the prompt's signature.
    return problem_prompt.rstrip() + "\n" + code


def _write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def resolve_runner(explicit: Optional[str]) -> str:
    choice = (explicit or os.environ.get("DRPT_EVALPLUS_RUNNER") or "auto").lower()
    if choice == "auto":
        for candidate in ("apptainer", "singularity"):
            if shutil.which(candidate):
                return candidate
        return "host"
    if choice in ("apptainer", "singularity"):
        if not shutil.which(choice):
            raise RuntimeError(f"--evalplus_runner {choice} requested but it is not on PATH")
        return choice
    if choice == "host":
        return "host"
    raise ValueError(f"Unknown evalplus runner {choice!r}; expected auto|apptainer|singularity|host")


def build_evaluate_command(
    *,
    runner: str,
    work_dir: str,
    image: str,
    dataset_path: Optional[str],
) -> List[str]:
    """Command that runs ``evalplus.evaluate`` on ``work_dir/SAMPLES_FILENAME``."""
    if runner == "host":
        return [
            sys.executable, "-m", "evalplus.evaluate", "--dataset", "mbpp",
            "--samples", os.path.join(work_dir, SAMPLES_FILENAME), "--version", MBPP_PLUS_DATASET_VERSION,
        ]
    command = [runner, "exec", "--containall", "--cleanenv", "--bind", f"{work_dir}:/workspace"]
    if dataset_path:
        container_dataset = "/opt/drpt/evalplus/MbppPlus.jsonl"
        command += [
            "--net", "--network", "none",
            "--bind", f"{dataset_path}:{container_dataset}:ro",
            "--env", f"MBPP_OVERRIDE_PATH={container_dataset}",
        ]
    command += ["--env", "XDG_CACHE_HOME=/workspace/.cache", image,
                "evalplus.evaluate", "--dataset", "mbpp",
                "--samples", f"/workspace/{SAMPLES_FILENAME}", "--version", MBPP_PLUS_DATASET_VERSION]
    return command


def parse_evalplus_results(payload: Mapping[str, Any], scored_task_ids: Sequence[str]) -> Dict[str, Any]:
    """EvalPlus persists per-completion statuses; with one greedy sample per task, pass@1 is their mean."""
    evaluated = payload.get("eval", {})
    missing = [task_id for task_id in scored_task_ids if task_id not in evaluated]
    if missing:
        raise RuntimeError(f"EvalPlus result is missing {len(missing)} tasks, e.g. {missing[:5]}")
    rows = []
    for task_id in scored_task_ids:
        task_rows = evaluated[task_id]
        if not isinstance(task_rows, list) or len(task_rows) != 1:
            raise RuntimeError(f"EvalPlus expected exactly one completion for {task_id}, got {task_rows!r}")
        rows.append(task_rows[0])
    n = len(rows)
    base = sum(row.get("base_status") == "pass" for row in rows) / n
    plus = sum(row.get("base_status") == "pass" and row.get("plus_status") == "pass" for row in rows) / n
    return {
        "base_pass_at_1": base * 100.0,
        "plus_pass_at_1": plus * 100.0,
        "base_plus_extra_pass_at_1": plus * 100.0,
        "n_test": n,
        "dataset_hash": payload.get("hash"),
    }


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    data_dir = getattr(args, "data_dir", "./data")
    requested_n = getattr(args, "n_test", -1)
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        raise ValueError("MBPP+ requires args.output_dir for samples and EvalPlus results")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_records = load_bench_records(data_dir, "mbpp_plus", k=-1)
    scored_records = all_records[:requested_n] if 0 < requested_n < len(all_records) else all_records
    scope = "limited" if len(scored_records) < len(all_records) else "full"

    def _task(record: Mapping[str, Any]) -> Tuple[str, str]:
        metadata = record.get("metadata", {})
        task_id = metadata.get("task_id") or record.get("id")
        prompt = metadata.get("prompt")
        if not task_id or not prompt:
            raise ValueError(f"Malformed MBPP+ record: {record.get('id')}")
        return str(task_id), str(prompt)

    task_rows = [_task(record) for record in scored_records]
    print(f"Loaded {len(all_records)} MBPP+ tasks; scoring {len(task_rows)} ({scope})")

    prompts = [
        render_generation_chat(tokenizer, _PROMPT_TEMPLATE.format(prompt=problem_prompt), enable_thinking=False)
        for _, problem_prompt in task_rows
    ]
    generations = generate_completions(
        model, tokenizer, prompts,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False, disable_tqdm=False,
    )

    samples, generation_rows = [], []
    for (task_id, problem_prompt), generation in zip(task_rows, generations):
        solution = assemble_solution(problem_prompt, generation)
        samples.append({"task_id": task_id, "solution": solution})
        generation_rows.append({
            "task_id": task_id, "raw_generation": generation,
            "response": clean_model_response(generation), "solution": solution,
        })
    _write_jsonl(os.path.join(output_dir, "mbpp_plus_generations.jsonl"), generation_rows)

    # EvalPlus requires one completion for every task in the registry. For a
    # limited smoke run, fill the unscored tasks with their canonical solutions
    # (stored in the bench file by prepare_datasets.py); they never enter the metric.
    if scope == "limited":
        scored_ids = {task_id for task_id, _ in task_rows}
        for record in all_records:
            task_id, problem_prompt = _task(record)
            if task_id in scored_ids:
                continue
            canonical = record.get("metadata", {}).get("canonical_solution")
            if canonical is None:
                raise RuntimeError(
                    "Limited MBPP+ runs need metadata.canonical_solution in the bench file; "
                    "rebuild it with prepare_datasets.py or use --n_test -1"
                )
            samples.append({"task_id": task_id, "solution": problem_prompt + canonical})

    work_dir = tempfile.mkdtemp(prefix=".mbpp_plus_eval-", dir=output_dir)
    os.chmod(work_dir, 0o700)
    _write_jsonl(os.path.join(work_dir, SAMPLES_FILENAME), samples)

    runner = resolve_runner(getattr(args, "evalplus_runner", None))
    image = getattr(args, "evalplus_image", None) or os.environ.get("DRPT_EVALPLUS_IMAGE") or DEFAULT_EVALPLUS_IMAGE
    dataset_path = getattr(args, "evalplus_dataset_path", None) or os.environ.get("DRPT_EVALPLUS_DATASET_PATH")
    if dataset_path:
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        if not os.path.isfile(dataset_path):
            raise RuntimeError(f"--evalplus_dataset_path does not exist: {dataset_path}")
    if runner == "host":
        logger.warning("Running EvalPlus on the host: model-generated code will execute in this environment")
    command = build_evaluate_command(runner=runner, work_dir=work_dir, image=image, dataset_path=dataset_path)
    print(f"Running EvalPlus via {runner}: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"EvalPlus evaluation failed: {' '.join(command)}") from exc

    results_path = os.path.join(work_dir, SAMPLES_FILENAME.removesuffix(".jsonl") + "_eval_results.json")
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"EvalPlus did not write a readable result: {results_path}") from exc
    shutil.copyfile(results_path, os.path.join(output_dir, "mbpp_plus_official_eval_results.json"))
    shutil.rmtree(work_dir, ignore_errors=True)

    scores = parse_evalplus_results(payload, [task_id for task_id, _ in task_rows])
    print("\nMBPP+ Results:")
    print(f"  base pass@1: {scores['base_pass_at_1']:.2f}%  base+extra pass@1: {scores['base_plus_extra_pass_at_1']:.2f}%  n={scores['n_test']}")
    scores["evaluation_scope"] = scope
    scores["accuracy"] = scores["base_plus_extra_pass_at_1"]
    scores["provenance"] = result_provenance(
        dataset_repository=DATASET_REPOSITORY,
        dataset_revision=single_source_revision(scored_records),
        dataset_split="test",
        evaluator={
            "package": "evalplus", "runner": runner, "image": image if runner != "host" else None,
            "dataset_version": MBPP_PLUS_DATASET_VERSION, "dataset_hash": scores.get("dataset_hash"),
            "primary_metric": "base_plus_extra_pass_at_1",
        },
        n_tasks=len(scored_records), max_new_tokens=max_new_tokens, thinking=False,
    )
    return scores
