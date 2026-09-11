#!/usr/bin/env python3
"""Stage 2 of the rewrite-target pipeline: verify candidates and assemble a new target.

Reads the candidate file(s) written by ``gen_target_candidates.py``, scores every
candidate with the source target's verifier (``SFT/data/target_verifiers.py``:
Math-Verify / recovered IFEval constraints / MBPP tests) and writes

    eval/<name>/<name>_{validation,test}_data.jsonl   (same schema as the source target;
                                                       only the assistant turn changes)
    eval/<name>/manifest.json                          (counts, per-row verification summary)

Per row the picked candidate is the first (rewrite mode) or shortest (solve mode)
verified-correct one; ``--pick`` overrides. Rows without a correct candidate follow
``--on_fail_validation`` (default ``drop``: D* only contains verified rows, as the
2026-09-02 ``math_gen`` target did) and ``--on_fail_test`` (default ``reference``:
the held-out loss split keeps its row count). Rows the verifier cannot check at all
(no recoverable IFEval constraint, no final answer in the reference) follow
``--unverifiable`` (default ``accept``: keep the picked candidate unverified, so the
new D* has one consistent style; ``reference`` / ``drop`` are the safer choices).

The new target is registered automatically: ``SFT/eval/eval.py`` maps
``<base>_gen*`` / ``<base>_rw*`` to the base target's benchmarks and the training
loader accepts any ``eval/<task>/<task>_<split>_data.jsonl``.

Example
-------
  python SFT/data/build_rewrite_target.py --data_dir $DATA --source_target precise_if \\
      --name precise_if_gen32b --candidates $DATA/candidates/precise_if_gen32b.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from SFT.data.target_common import (  # noqa: E402
    assistant_reference, base_target_name, infer_domain, read_jsonl, target_split_path, write_jsonl,
)
from SFT.data.target_verifiers import build_verifier  # noqa: E402
from SFT.eval.tasks.common import clean_model_response  # noqa: E402

PICK_DEFAULT = {"solve": "shortest", "rewrite": "first"}


def pick_candidate(correct: List[str], how: str) -> str:
    if how == "first":
        return correct[0]
    if how == "shortest":
        return min(correct, key=len)
    if how == "longest":
        return max(correct, key=len)
    if how == "median":
        return sorted(correct, key=len)[len(correct) // 2]
    raise ValueError(how)


def build_split(
    *, split: str, source_rows: List[dict], cand_rows: Dict[str, dict], verifier, name: str, source_target: str,
    pick: Optional[str], on_fail: str, unverifiable: str, min_chars: int,
) -> tuple[List[dict], dict, List[dict]]:
    counts: collections.Counter = collections.Counter()
    statuses: collections.Counter = collections.Counter()
    methods: collections.Counter = collections.Counter()
    out_rows: List[dict] = []
    per_row: List[dict] = []
    ref_lengths: List[int] = []
    gen_lengths: List[int] = []

    for row in source_rows:
        counts["rows_in"] += 1
        cand = cand_rows.get(row["id"])
        reference = assistant_reference(row)
        mode = (cand or {}).get("mode", "solve")
        how = pick or PICK_DEFAULT.get(mode, "shortest")
        candidates = []
        seen = set()
        for text in (cand or {}).get("candidates", []):
            cleaned = clean_model_response(text).strip()
            if len(cleaned) < min_chars or cleaned in seen:
                counts["candidates_dropped_empty_or_duplicate"] += 1
                continue
            seen.add(cleaned)
            candidates.append(cleaned)
        counts["candidates_total"] += len(candidates)

        context = verifier.prepare(row)
        chosen: Optional[str] = None
        source = "reference"
        verification = "no_candidates" if not candidates else None
        n_correct = 0
        if candidates:
            if context is None:
                counts["rows_unverifiable"] += 1
                verification = "unverifiable"
                if unverifiable == "accept":
                    chosen, source = pick_candidate(candidates, how), "generated"
                elif unverifiable == "drop":
                    counts["rows_dropped"] += 1
                    per_row.append({"id": row["id"], "verification": verification, "decision": "drop"})
                    continue
            else:
                results = [verifier.verify(context, text) for text in candidates]
                for result in results:
                    statuses[result.status] += 1
                    if result.correct:
                        methods[result.detail.get("method", verifier.name)] += 1
                correct = [text for text, result in zip(candidates, results) if result.correct]
                n_correct = len(correct)
                counts["candidates_correct"] += n_correct
                if correct:
                    counts["rows_with_correct"] += 1
                    chosen, source, verification = pick_candidate(correct, how), "generated", "correct"
                else:
                    counts["rows_all_incorrect"] += 1
                    verification = "all_incorrect"
                    if on_fail == "drop":
                        counts["rows_dropped"] += 1
                        per_row.append({"id": row["id"], "verification": verification, "n_candidates": len(candidates),
                                        "decision": "drop", "gold": str(context.get("gold_answer", ""))[:120]})
                        continue
        elif on_fail == "drop":
            counts["rows_dropped"] += 1
            per_row.append({"id": row["id"], "verification": verification, "decision": "drop"})
            continue

        new = json.loads(json.dumps(row))
        new["dataset"] = name
        new["id"] = row["id"].replace(f"{source_target}::", f"{name}::", 1) if row["id"].startswith(f"{source_target}::") else f"{name}::{row['id']}"
        if chosen is not None:
            new["messages"] = [m if m.get("role") != "assistant" else {"role": "assistant", "content": chosen} for m in row["messages"]]
            gen_lengths.append(len(chosen))
        ref_lengths.append(len(reference))
        new.setdefault("metadata", {}).update({
            "source_target": source_target, "answer_source": source, "verification": verification,
            "generator": (cand or {}).get("generator") if source == "generated" else None,
            "generation_mode": mode if source == "generated" else None,
            "n_candidates": len(candidates), "n_candidates_correct": n_correct, "reference_solution": reference,
        })
        counts["rows_out"] += 1
        counts[f"rows_out_{source}"] += 1
        out_rows.append(new)
        per_row.append({"id": row["id"], "verification": verification, "n_candidates": len(candidates),
                        "n_correct": n_correct, "decision": source, "chars_ref": len(reference),
                        "chars_out": len(chosen) if chosen is not None else len(reference)})

    summary = {
        "split": split, **dict(counts), "status_histogram": dict(statuses), "correct_by_method": dict(methods),
        "candidate_accuracy": round(counts["candidates_correct"] / counts["candidates_total"], 4) if counts["candidates_total"] else None,
        "median_chars_reference": statistics.median(ref_lengths) if ref_lengths else None,
        "median_chars_generated": statistics.median(gen_lengths) if gen_lengths else None,
    }
    return out_rows, summary, per_row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--source_target", required=True, help="target whose prompts/references are reused, e.g. precise_if")
    parser.add_argument("--name", required=True, help="new target name, e.g. precise_if_gen32b (must be <source>_gen*/_rw*)")
    parser.add_argument("--candidates", nargs="+", required=True, help="candidate jsonl file(s) from gen_target_candidates.py")
    parser.add_argument("--pick", choices=["first", "shortest", "longest", "median"], default=None,
                        help="which verified candidate to keep (default: first for rewrite, shortest for solve)")
    parser.add_argument("--on_fail_validation", choices=["drop", "reference"], default="drop")
    parser.add_argument("--on_fail_test", choices=["drop", "reference"], default="reference")
    parser.add_argument("--unverifiable", choices=["accept", "reference", "drop"], default="accept")
    parser.add_argument("--loose_if", action="store_true", help="verify IF constraints with the loose rule (default strict)")
    parser.add_argument("--min_chars", type=int, default=8, help="drop candidates shorter than this")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if base_target_name(args.name) != args.source_target:
        raise SystemExit(f"--name {args.name!r} must be {args.source_target}_gen<tag> or {args.source_target}_rw<tag> "
                         "so eval.py can map it to the source target's benchmarks")
    out_dir = os.path.join(args.data_dir, "eval", args.name)
    if os.path.exists(os.path.join(out_dir, "manifest.json")) and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists (--overwrite to rebuild)")

    cand_by_split: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)
    generators = set()
    modes = set()
    for path in args.candidates:
        for row in read_jsonl(path):
            if row.get("source_target") != args.source_target:
                raise SystemExit(f"{path}: candidate row {row.get('id')} belongs to {row.get('source_target')!r}, not {args.source_target!r}")
            cand_by_split[row["split"]][row["id"]] = row
            generators.add(row.get("generator"))
            modes.add(row.get("mode"))
    if not cand_by_split:
        raise SystemExit("no candidate rows found")

    domain = infer_domain(args.source_target)
    verifier = build_verifier(args.source_target, **({"strict": not args.loose_if} if domain == "if" else {}))
    manifest: Dict[str, Any] = {
        "name": args.name, "source_target": args.source_target, "domain": domain, "verifier": verifier.name,
        "generators": sorted(g for g in generators if g), "modes": sorted(m for m in modes if m),
        "candidate_files": [os.path.abspath(p) for p in args.candidates],
        "policy": {"pick": args.pick or PICK_DEFAULT, "on_fail_validation": args.on_fail_validation,
                   "on_fail_test": args.on_fail_test, "unverifiable": args.unverifiable, "if_strict": not args.loose_if},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "splits": {}, "rows": {},
    }
    if hasattr(verifier, "scorer_info"):
        manifest["scorer"] = verifier.scorer_info

    for split in ("validation", "test"):
        if split not in cand_by_split:
            print(f"{split}: no candidates, split not written")
            continue
        source_rows = read_jsonl(target_split_path(args.data_dir, args.source_target, split))
        # --limit smoke runs only cover a prefix of the rows: restrict to rows that have candidates.
        source_rows = [r for r in source_rows if r["id"] in cand_by_split[split]]
        on_fail = args.on_fail_validation if split == "validation" else args.on_fail_test
        started = time.time()
        rows, summary, per_row = build_split(
            split=split, source_rows=source_rows, cand_rows=cand_by_split[split], verifier=verifier, name=args.name,
            source_target=args.source_target, pick=args.pick, on_fail=on_fail, unverifiable=args.unverifiable,
            min_chars=args.min_chars,
        )
        write_jsonl(target_split_path(args.data_dir, args.name, split), rows)
        summary["elapsed_sec"] = round(time.time() - started, 1)
        manifest["splits"][split] = summary
        manifest["rows"][split] = per_row
        print(f"{split}: {summary['rows_in']} rows in -> {summary['rows_out']} out "
              f"({summary.get('rows_out_generated', 0)} generated, {summary.get('rows_out_reference', 0)} reference kept, "
              f"{summary.get('rows_dropped', 0)} dropped, {summary.get('rows_unverifiable', 0)} unverifiable); "
              f"candidate accuracy {summary['candidate_accuracy']}; median chars ref {summary['median_chars_reference']} "
              f"-> gen {summary['median_chars_generated']}; statuses {summary['status_histogram']}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, ensure_ascii=False)
    print(f"manifest -> {os.path.join(out_dir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
