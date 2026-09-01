"""IFEval: verifiable instruction-following constraints on 541 prompts.

Reports the four official metrics (prompt- and instruction-level accuracy under
the strict and loose response interpretations) using the vendored
google-research verifier in ``ifeval_lib``. Primary metric: prompt-level strict.

Ported from Dr.Post-Training-Next (SFT/eval/tasks/ifeval.py) with the
content-addressed artifact plumbing replaced by ``bench_data.load_bench_records``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from SFT.eval.tasks.bench_data import first_user_content, load_bench_records
from SFT.eval.tasks.common import (
    clean_model_response,
    render_generation_chat,
    result_provenance,
    single_source_revision,
)
from SFT.eval.tasks.ifeval_scoring import aggregate, evaluate_instruction_following
from ..utils import generate_completions, get_eos_token_ids

# Long enough for multi-constraint answers ("at least 300 words", "N sections")
# without letting a degenerate model burn the whole eval budget.
DEFAULT_MAX_NEW_TOKENS = 2048
DATASET_REPOSITORY = "google/IFEval"
EVALUATOR_REPOSITORY = "google-research/google-research"
# Commit of instruction_following_eval that ifeval_lib was vendored from.
EVALUATOR_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    data_dir = getattr(args, "data_dir", "./data")
    n_test = getattr(args, "n_test", -1)

    records = load_bench_records(data_dir, "ifeval", k=n_test)
    print(f"Loaded {len(records)} IFEval prompts")

    prompts = [
        render_generation_chat(tokenizer, first_user_content(record), enable_thinking=False)
        for record in records
    ]
    print("Generating responses...")
    generations = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False,
        disable_tqdm=False,
    )

    strict_results: List[Dict[str, Any]] = []
    loose_results: List[Dict[str, Any]] = []
    per_example: List[Dict[str, Any]] = []

    for record, generation in zip(records, generations):
        metadata = record.get("metadata", {})
        prompt_text = metadata.get("prompt") or first_user_content(record)
        instruction_ids = metadata.get("instruction_id_list") or []
        kwargs_list = metadata.get("kwargs") or []
        # Only chat/think markup is removed; whitespace, capitalization,
        # punctuation, and markdown are load-bearing for IFEval constraints.
        response = clean_model_response(generation, strip_thinking=True)

        strict = evaluate_instruction_following(
            prompt=prompt_text, response=response,
            instruction_id_list=instruction_ids, kwargs_list=kwargs_list, strict=True,
        )
        loose = evaluate_instruction_following(
            prompt=prompt_text, response=response,
            instruction_id_list=instruction_ids, kwargs_list=kwargs_list, strict=False,
        )
        strict_results.append(strict)
        loose_results.append(loose)
        per_example.append({
            "key": metadata.get("key"),
            "instruction_id_list": instruction_ids,
            "raw_generation": generation,
            "response": response,
            "strict": strict["follow_instruction_list"],
            "loose": loose["follow_instruction_list"],
        })

    strict_scores = aggregate(strict_results)
    loose_scores = aggregate(loose_results)

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        with open(os.path.join(output_dir, "ifeval_generations.jsonl"), "w", encoding="utf-8") as handle:
            for row in per_example:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nIFEval Results:")
    print(f"  strict  prompt-level: {strict_scores['prompt_level_acc']:.2f}  "
          f"instruction-level: {strict_scores['inst_level_acc']:.2f}")
    print(f"  loose   prompt-level: {loose_scores['prompt_level_acc']:.2f}  "
          f"instruction-level: {loose_scores['inst_level_acc']:.2f}")
    print(f"  n={strict_scores['n_prompts']} prompts, {strict_scores['n_instructions']} instructions")

    return {
        "evaluation_scope": "full" if n_test <= 0 else "limited",
        "prompt_level_strict_acc": strict_scores["prompt_level_acc"],
        "inst_level_strict_acc": strict_scores["inst_level_acc"],
        "prompt_level_loose_acc": loose_scores["prompt_level_acc"],
        "inst_level_loose_acc": loose_scores["inst_level_acc"],
        # Primary metric: official prompt-level strict accuracy (percent).
        "accuracy": strict_scores["prompt_level_acc"],
        "n_test": strict_scores["n_prompts"],
        "n_instructions": strict_scores["n_instructions"],
        "provenance": result_provenance(
            dataset_repository=DATASET_REPOSITORY,
            dataset_revision=single_source_revision(records),
            dataset_split="train",
            evaluator={
                "repository": EVALUATOR_REPOSITORY,
                "revision": EVALUATOR_REVISION,
                "implementation": "vendored official instruction_following_eval",
                "primary_metric": "prompt_level_strict_acc",
            },
            n_tasks=len(records),
            max_new_tokens=max_new_tokens,
            thinking=False,
        ),
    }
