"""Loader for downstream **benchmark** files.

Terminology (see SFT/README.md): a *benchmark* is the official evaluation set of
a downstream task (IFEval, IFBench, MATH500, MBPP+). It is scored by generating
a response and running the task's verifier. It is a different object from the
*target* splits (``eval/<target>/<target>_{validation,test}_data.jsonl``), which
carry reference responses and are used for the curation gradient and for the
loss curve during training.

Benchmark files live at ``{data_dir}/eval/{task}/{task}_bench_data.jsonl``.
Each row is::

    {"dataset": "<task>", "id": "...",
     "messages": [{"role": "user", "content": "<prompt>"}],
     "metadata": {...task-specific scoring fields...}}

Build them with ``python SFT/data/prepare_datasets.py --datasets ifeval ifbench math500 mbpp_plus``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def bench_data_path(data_dir: str, task: str) -> str:
    return os.path.join(data_dir, "eval", task, f"{task}_bench_data.jsonl")


def load_bench_records(data_dir: str, task: str, k: int = -1) -> List[Dict[str, Any]]:
    """Load the official benchmark records for one task.

    ``k <= 0`` loads the whole benchmark (the default for reporting); a positive
    ``k`` truncates to a deterministic prefix for smoke tests.
    """
    path = bench_data_path(data_dir, task)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{task} benchmark data not found: {path}\n"
            f"Build it with: python SFT/data/prepare_datasets.py --datasets {task}"
        )
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if k > 0 and len(records) >= k:
                break
    if not records:
        raise ValueError(f"{task} benchmark data is empty: {path}")
    return records


def first_user_content(record: Dict[str, Any]) -> str:
    for message in record.get("messages", []) or []:
        if message.get("role") == "user":
            return message.get("content", "") or ""
    return ""
