#!/usr/bin/env python3
"""Build `math_pool`: held-out Dolci rows from all four pool math sources (Tulu 3 Persona MATH /
GSM / Algebra + the Dolci slice of OpenMathInstruct 2), equal weight (D* 32 x 4 = 128, held-out
125 x 4 = 500), not in any training pool, decontaminated against the benchmarks. The pool side of
the `math` target (build_math_target.py) is drawn from these files.
"""
import argparse, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SFT.data.prepare_datasets import (_load_dolci, _dolci_clean_messages, PromptDecontaminator,
                                       _reference_prompts, DOLCI_PIN, DOLCI_SAMPLE_SEED, DOLCI_POOL_DOMAINS)

SOURCES = ("Tulu 3 Persona MATH", "Tulu 3 Persona GSM", "Tulu 3 Persona Algebra", "OpenMathInstruct 2")
N_VAL, N_TEST = 128, 500


def largest_remainder(weights, total):
    raw = [w / sum(weights) * total for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True); ap.add_argument("--num_proc", type=int, default=16)
    a = ap.parse_args()
    in_pool = set()
    for pool in DOLCI_POOL_DOMAINS:
        p = os.path.join(a.data_dir, "train", pool, f"{pool}_data.jsonl")
        if not os.path.isfile(p):
            continue
        in_pool |= {json.loads(l)["metadata"]["source_id"] for l in open(p)}
    print(f"rows already in pools: {len(in_pool):,}")

    quotas = {
        "math_pool": {"validation": dict(zip(SOURCES, largest_remainder([1] * 4, N_VAL))),
                      "test": dict(zip(SOURCES, largest_remainder([1] * 4, N_TEST)))},
    }
    print("quotas:", json.dumps(quotas, indent=1))
    need_val = {s: max(q["validation"][s] for q in quotas.values()) for s in SOURCES}
    need_test = {s: max(q["test"][s] for q in quotas.values()) for s in SOURCES}

    blocker = PromptDecontaminator(_reference_prompts(a.data_dir))
    ds = _load_dolci(a.num_proc)
    cand = ds.filter(lambda ex, ids=in_pool: ex["eligible"] and ex["source_dataset"] in SOURCES and ex["n_turns"] == 2 and ex["id"] not in ids,
                     num_proc=a.num_proc, desc="math candidates")
    print(f"held-out single-turn math rows not in any pool: {len(cand):,} {dict(Counter(cand['source_dataset']))}")
    cand = cand.shuffle(seed=DOLCI_SAMPLE_SEED)

    # Per source: first need_val rows -> validation pool, next need_test rows -> test pool.
    val_pool, test_pool, blocked = {s: [] for s in SOURCES}, {s: [] for s in SOURCES}, 0
    for ex in cand:
        s = ex["source_dataset"]
        if len(val_pool[s]) >= need_val[s] and len(test_pool[s]) >= need_test[s]:
            if all(len(val_pool[t]) >= need_val[t] and len(test_pool[t]) >= need_test[t] for t in SOURCES):
                break
            continue
        msgs = _dolci_clean_messages(ex["messages"])
        if blocker.blocked_messages(msgs):
            blocked += 1; continue
        row = {"messages": msgs, "metadata": {"source_id": ex["id"], "source_dataset": s, "domain": ex["domain"],
                                              "source_repo": DOLCI_PIN["repo"], "source_revision": DOLCI_PIN["revision"]}}
        (val_pool if len(val_pool[s]) < need_val[s] else test_pool)[s].append(row)
    print(f"dropped {blocked} benchmark-overlapping rows")
    for s in SOURCES:
        assert len(val_pool[s]) >= need_val[s] and len(test_pool[s]) >= need_test[s], f"not enough rows for {s}"

    for name, q in quotas.items():
        out_dir = os.path.join(a.data_dir, "eval", name); os.makedirs(out_dir, exist_ok=True)
        for split, pool in (("validation", val_pool), ("test", test_pool)):
            part = [r for s in SOURCES for r in pool[s][: q[split][s]]]
            path = os.path.join(out_dir, f"{name}_{split}_data.jsonl")
            with open(path, "w") as f:
                for r in part:
                    f.write(json.dumps({"dataset": name, "id": f"{name}::{r['metadata']['source_id']}", **r}, ensure_ascii=False) + "\n")
            print(f"{name}/{split}: {len(part)} rows {dict(Counter(r['metadata']['source_dataset'] for r in part))} -> {path}")


if __name__ == "__main__":
    main()
