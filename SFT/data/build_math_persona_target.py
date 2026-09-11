#!/usr/bin/env python3
"""Build the `math_persona` target: held-out Dolci rows from the pool's own math sources
(Tulu 3 Persona MATH / Algebra / GSM) that are NOT in any training pool, decontaminated
against the benchmarks. D* then has exactly the pool's math style; the target steers
toward the pool's math distribution rather than toward MATH-style problems.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SFT.data.prepare_datasets import (_load_dolci, _dolci_clean_messages, PromptDecontaminator,
                                       _reference_prompts, DOLCI_PIN, DOLCI_SAMPLE_SEED, DOLCI_POOL_DOMAINS)

SOURCES = ("Tulu 3 Persona MATH", "Tulu 3 Persona Algebra", "Tulu 3 Persona GSM")
N_VAL, N_TEST = 128, 500

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True); ap.add_argument("--num_proc", type=int, default=16)
    a = ap.parse_args()
    in_pool = set()
    for pool in DOLCI_POOL_DOMAINS:
        p = os.path.join(a.data_dir, "train", pool, f"{pool}_data.jsonl")
        if os.path.isfile(p):
            in_pool |= {json.loads(l)["metadata"]["source_id"] for l in open(p)}
    print(f"rows already in pools: {len(in_pool):,}")
    blocker = PromptDecontaminator(_reference_prompts(a.data_dir))
    ds = _load_dolci(a.num_proc)
    cand = ds.filter(lambda ex, ids=in_pool: ex["eligible"] and ex["source_dataset"] in SOURCES and ex["n_turns"] == 2 and ex["id"] not in ids,
                     num_proc=a.num_proc, desc="persona candidates")
    print(f"held-out single-turn Persona math rows not in any pool: {len(cand):,}")
    cand = cand.shuffle(seed=DOLCI_SAMPLE_SEED)
    rows, blocked = [], 0
    for ex in cand:
        if len(rows) >= N_VAL + N_TEST: break
        msgs = _dolci_clean_messages(ex["messages"])
        if blocker.blocked_messages(msgs): blocked += 1; continue
        rows.append({"dataset": "math_persona", "id": f"math_persona::{ex['id']}", "messages": msgs,
                     "metadata": {"source_id": ex["id"], "source_dataset": ex["source_dataset"], "domain": ex["domain"],
                                  "source_repo": DOLCI_PIN["repo"], "source_revision": DOLCI_PIN["revision"]}})
    print(f"dropped {blocked} benchmark-overlapping rows; selected {len(rows)}")
    out_dir = os.path.join(a.data_dir, "eval", "math_persona"); os.makedirs(out_dir, exist_ok=True)
    for split, part in (("validation", rows[:N_VAL]), ("test", rows[N_VAL:N_VAL + N_TEST])):
        path = os.path.join(out_dir, f"math_persona_{split}_data.jsonl")
        with open(path, "w") as f:
            for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter
        print(f"{split}: {len(part)} rows {dict(Counter(r['metadata']['source_dataset'] for r in part))} -> {path}")

if __name__ == "__main__":
    main()
