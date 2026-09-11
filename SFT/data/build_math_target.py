#!/usr/bin/env python3
"""Build the `math` target (and its subset variants): the benchmarks' held-out sets AND the pools' math sources.

D* (128) / held-out (500):
  benchmark side : MATH train  32 / 125  (rows of eval/math_ref, official reference solutions)
                   GSM8K train 32 / 125  (openai/gsm8k train, pinned; solutions exactly as published)
  pool side      : 16 / 62-63 per source from eval/math_pool (Persona MATH / GSM / Algebra,
                   OpenMathInstruct 2 -- held-out Dolci rows that are in no training pool)
Variants (--name) reuse the same rows and only change which sources are included:
  math       : MATH 32, GSM8K 32, four pool sources 16 each                (held-out 125/125/62-63)
  math_v2    : GSM8K 32, Persona MATH/GSM/Algebra 32 each                  (drops MATH train + OpenMathInstruct 2,
               the two D* sources whose steps select the short OpenMathInstruct-2 pool rows)
  math_bench : MATH 64, GSM8K 64                                            (benchmark held-out sets only)
Rows are copied verbatim; the only curation is which rows are included. GSM8K train problems
that match any pool row (exact or >= 80% shared word 8-grams; ~400 of 7,473 are in the pools
via OpenMathInstruct 2 / FLAN) are excluded, and all rows are checked against the benchmarks.
"""
import argparse, json, os, random, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import load_dataset
from SFT.data.decontam import PromptDecontaminator, user_prompts
from SFT.data.prepare_datasets import _reference_prompts, BENCHMARK_PINS, DOLCI_POOL_DOMAINS, DOLCI_SAMPLE_SEED

POOL_SOURCES = ("Tulu 3 Persona MATH", "Tulu 3 Persona GSM", "Tulu 3 Persona Algebra", "OpenMathInstruct 2")
QUOTAS = {  # name -> source -> (validation = D*, test = held-out)
    "math": {"MATH train": (32, 125), "GSM8K train": (32, 125),
             "Tulu 3 Persona MATH": (16, 63), "Tulu 3 Persona GSM": (16, 62),
             "Tulu 3 Persona Algebra": (16, 63), "OpenMathInstruct 2": (16, 62)},
    "math_v2": {"GSM8K train": (32, 125), "Tulu 3 Persona MATH": (32, 125),
                "Tulu 3 Persona GSM": (32, 125), "Tulu 3 Persona Algebra": (32, 125)},
    "math_bench": {"MATH train": (64, 250), "GSM8K train": (64, 250)},
}


def read(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--name", default="math", choices=sorted(QUOTAS)); a = ap.parse_args()
    QUOTA = QUOTAS[a.name]; name = a.name
    ev = os.path.join(a.data_dir, "eval")
    bench = PromptDecontaminator(_reference_prompts(a.data_dir))

    # --- MATH train: reuse math_ref rows (the pools were decontaminated against exactly these)
    math_val, math_test = read(f"{ev}/math_ref/math_ref_validation_data.jsonl"), read(f"{ev}/math_ref/math_ref_test_data.jsonl")
    for r in math_val + math_test:
        r["metadata"]["source_dataset"] = "MATH train"

    # --- GSM8K train: exclude problems present in any pool, then benchmark-check
    pool_prompts = []
    for pool in DOLCI_POOL_DOMAINS:
        for r in read(os.path.join(a.data_dir, "train", pool, f"{pool}_data.jsonl")):
            pool_prompts += [(f"{pool}:{r['id']}:{i}", p) for i, p in enumerate(user_prompts(r["messages"]))]
    in_pool = PromptDecontaminator(pool_prompts)
    pin = BENCHMARK_PINS["gsm8k"]
    gsm = load_dataset(pin["repo"], pin["config"], split="train", revision=pin["revision"])
    idx = list(range(len(gsm))); random.Random(DOLCI_SAMPLE_SEED).shuffle(idx)
    gsm_rows, n_pool, n_bench = [], 0, 0
    for i in idx:
        q, sol = gsm[i]["question"], gsm[i]["answer"]
        if in_pool.match(q) is not None:
            n_pool += 1; continue
        if bench.match(q) is not None:
            n_bench += 1; continue
        gsm_rows.append({"id": f"gsm8k_train::{i}", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": sol}],
                         "metadata": {"source_dataset": "GSM8K train", "source_repo": pin["repo"], "source_revision": pin["revision"],
                                      "source_split": "train", "source_index": i}})
        if len(gsm_rows) >= sum(QUOTA.get("GSM8K train", (0, 0))):
            break
    print(f"GSM8K train: skipped {n_pool} problems present in a pool and {n_bench} benchmark matches; kept {len(gsm_rows)}")

    # --- pool side: first rows per source of math_pool (already outside every pool, benchmark-clean)
    pool_val, pool_test = read(f"{ev}/math_pool/math_pool_validation_data.jsonl"), read(f"{ev}/math_pool/math_pool_test_data.jsonl")

    def take(rows, source, n):
        out = [r for r in rows if r["metadata"]["source_dataset"] == source][:n]
        assert len(out) == n, f"{source}: wanted {n}, got {len(out)}"
        return out

    qm, qg = QUOTA.get("MATH train", (0, 0)), QUOTA.get("GSM8K train", (0, 0))
    val = take(math_val, "MATH train", qm[0]) + gsm_rows[:qg[0]]
    test = take(math_test, "MATH train", qm[1]) + gsm_rows[qg[0]:sum(qg)]
    for s in POOL_SOURCES:
        if s in QUOTA:
            val += take(pool_val, s, QUOTA[s][0]); test += take(pool_test, s, QUOTA[s][1])

    # every row must be benchmark-clean; no row may appear in both splits
    assert not any(bench.blocked_messages(r["messages"]) for r in val + test), "benchmark match in target"
    assert len({r["id"] for r in val} & {r["id"] for r in test}) == 0
    out_dir = f"{ev}/{name}"; os.makedirs(out_dir, exist_ok=True)
    for split, part in (("validation", val), ("test", test)):
        path = f"{out_dir}/{name}_{split}_data.jsonl"
        with open(path, "w") as f:
            for r in part:
                rid = re.sub(r"^(math_ref|math_pool)::", "", r["id"])
                f.write(json.dumps({"dataset": name, "id": f"{name}::{rid}", "messages": r["messages"], "metadata": r["metadata"]}, ensure_ascii=False) + "\n")
        print(f"{name}/{split}: {len(part)} rows {dict(Counter(r['metadata']['source_dataset'] for r in part))} -> {path}")


if __name__ == "__main__":
    main()
