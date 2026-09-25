"""Paper table: rlhf_scoring_backend.tex (adxtab:rlhf-scoring-backend).

Experiments run on : the H200 Slurm cluster (training runs with the exact-PIP and compressed scoring backends; final-adapter rescoring)
Launcher           : RLHF/train/train.sh -c configs/toxicity -m <arm> (arms FullTraining-LoRA, IIF-LoRA[-cmp], GlobalSubset-LoRA[-cmp],
                     LayerWiseSubset-LoRA[-cmp]; --n_val 0|1024, --val_loss_type reward|train-loss); judges via RLHF/eval/rescore_final.py
Results read from  : <results root>/RLHF/toxicity-gpt-neo-2.7B-<arm>-lr1e-5-b256-<scenario>-pe4-mb4-kl0.02-s<seed>/rescore_final.json
Seeds / statistics : 2, 22, 42, 62, 82; mean +- SE (population std / sqrt n) of 100 x mean toxicity probability over 500 prompts
Usage              : python RLHF/tables/scoring_backend.py [--results-root DIR] [--paper-root DIR] [--venues ICLR] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import add_common_args, emit, mean_se, results_root  # noqa: E402

TABLE = "rlhf_scoring_backend.tex"
SEEDS = [2, 22, 42, 62, 82]
METHODS = [("IIF", "IIF"), ("GlobalSubset", "Global Subset"), ("LayerWiseSubset", "Layer-Wise Subset")]
SCENARIOS = [("v0-rew", "Self-ref., reward"), ("v0-tloss", "Self-ref., train loss"),
             ("v1024-rew-b256", "Held-out, reward"), ("v1024-tloss-b256", "Held-out, train loss")]
JUDGES = [("danlp", "DaNLP"), ("lftw", "LFTW"), ("snlp", "s-nlp"), ("detoxify", "toxic-bert")]
BACKENDS = [("", "Exact"), ("-cmp-normal-64x64", "Compressed")]
PPO = ("FullTraining", "v0-rew")  # standard PPO row, shared between the two backends


def run_dir(root: Path, method: str, scenario: str, seed: int) -> Path:
    return root / "RLHF" / f"toxicity-gpt-neo-2.7B-{method}-LoRA-lr1e-5-b256-{scenario}-pe4-mb4-kl0.02-s{seed}"


def load_arm(root: Path, method: str, scenario: str) -> dict[str, dict[int, float]]:
    """{judge: {seed: 100 * mean toxicity prob}} over the seeds whose rescore_final.json exists."""
    out: dict[str, dict[int, float]] = {j: {} for j, _ in JUDGES}
    for seed in SEEDS:
        path = run_dir(root, method, scenario, seed) / "rescore_final.json"
        if not path.exists():
            continue
        judges = json.loads(path.read_text())["judges"]
        for j, _ in JUDGES:
            if j in judges:
                out[j][seed] = 100.0 * float(judges[j]["mean_toxicity_prob"])
    return out


def cell(values: dict[int, float]) -> str:
    """``m \\(\\pm\\) se`` at two decimals; ``--`` if empty; ``(n=k)`` if k < 5."""
    ms = mean_se(values)
    if ms is None:
        return "--"
    m, se, n = ms
    return f"{m:.2f} \\(\\pm\\) {se:.2f}" + ("" if n == len(SEEDS) else f" (n={n})")


def build(root: Path) -> tuple[str, list[str]]:
    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Target} & \multirow{2}{*}{Method} & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\texttt{{{lab}}}}}" for _, lab in JUDGES) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(JUDGES))),
        " & & " + " & ".join(" & ".join(b for _, b in BACKENDS) for _ in JUDGES) + r" \\",
        r"\midrule",
    ]
    completeness: list[str] = []

    def found(arm: dict[str, dict[int, float]]) -> int:
        return len(arm[JUDGES[0][0]])

    ppo = load_arm(root, *PPO)
    completeness.append(f"FullTraining/{PPO[1]}: {found(ppo)}/{len(SEEDS)}")
    lines.append(r"\multicolumn{2}{l}{Full-Training (standard PPO)} & "
                 + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cell(ppo[j])}}}" for j, _ in JUDGES) + r" \\")
    for scenario, slabel in SCENARIOS:
        lines.append(r"\midrule")
        for i, (method, mlabel) in enumerate(METHODS):
            arms = [load_arm(root, method + suffix, scenario) for suffix, _ in BACKENDS]
            for (suffix, _), arm in zip(BACKENDS, arms):
                completeness.append(f"{method}{suffix}/{scenario}: {found(arm)}/{len(SEEDS)}")
            head = f"\\multirow{{{len(METHODS)}}}{{*}}{{{slabel}}}" if i == 0 else ""
            cells = [cell(arm[j]) for j, _ in JUDGES for arm in arms]
            lines.append(f"{head} & {mlabel} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", completeness


def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0])).parse_args()
    root = results_root(args.results_root)
    body, completeness = build(root)
    n_found = sum(int(c.split(": ")[1].split("/")[0]) for c in completeness)
    n_expected = len(SEEDS) * len(completeness)
    incomplete = [c for c in completeness if not c.endswith(f"{len(SEEDS)}/{len(SEEDS)}")]
    print(f"completeness: {n_found}/{n_expected} runs found over {len(completeness)} arms"
          + (" (incomplete: " + ", ".join(incomplete) + ")" if incomplete else ""))
    return emit({TABLE: body}, args, __file__, root)


if __name__ == "__main__":
    sys.exit(main())
