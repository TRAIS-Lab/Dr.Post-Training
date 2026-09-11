"""GSM8K: 1319 grade-school word problems (openai/gsm8k, main config, test split).
Same protocol as MATH500: chat prompt with the boxed-answer instruction, greedy or sampled
generation, math-verify scoring of the final answer against the gold number (the text after
``####`` in the reference solution). Primary metric: accuracy.
"""
from __future__ import annotations
import json, os
from typing import Any, Dict, List
from SFT.eval.tasks.bench_data import bench_data_path, first_user_content, load_bench_records
from SFT.eval.tasks.common import clean_model_response, render_generation_chat, result_provenance, sampling_kwargs, single_source_revision
from SFT.eval.tasks.math500 import _PROMPT_TEMPLATE, _load_scorer
from ..utils import generate_completions, get_eos_token_ids

DEFAULT_MAX_NEW_TOKENS = 1024
DATASET_REPOSITORY = "openai/gsm8k"


def compute_accuracy(args, model, tokenizer, batch_size: int = 4, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Dict[str, Any]:
    data_dir = getattr(args, "data_dir", "./data"); requested_n = getattr(args, "n_test", -1)
    if not os.path.exists(bench_data_path(data_dir, "gsm8k")):
        raise FileNotFoundError(f"GSM8K bench file missing: {bench_data_path(data_dir, 'gsm8k')} (python SFT/data/prepare_datasets.py --datasets gsm8k)")
    score, scorer_info = _load_scorer()
    records = load_bench_records(data_dir, "gsm8k", k=requested_n)
    print(f"Loaded {len(records)} GSM8K problems; scorer={scorer_info['package']}")
    prompts = [render_generation_chat(tokenizer, _PROMPT_TEMPLATE.format(problem=(r.get("metadata", {}).get("problem") or first_user_content(r))), enable_thinking=False) for r in records]
    generations = generate_completions(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=get_eos_token_ids(tokenizer),
                                       **sampling_kwargs(args), disable_tqdm=False)
    per_example: List[Dict[str, Any]] = []
    for record, generation in zip(records, generations):
        gold = record.get("metadata", {}).get("answer")
        if gold is None:
            raise ValueError(f"GSM8K record has no metadata.answer: {record.get('id')}")
        response = clean_model_response(generation)
        scored = score(str(gold), response)
        per_example.append({"id": record.get("id"), "gold_answer": gold, "raw_generation": generation, "response": response, **scored})
    n_correct = sum(1 for r in per_example if r["correct"]); n_parse = sum(1 for r in per_example if "parse_error" in r["status"])
    accuracy = n_correct / len(records) * 100.0 if records else 0.0
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        with open(os.path.join(output_dir, "gsm8k_generations.jsonl"), "w", encoding="utf-8") as h:
            for r in per_example: h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nGSM8K Results:"); print(f"  accuracy: {accuracy:.2f}%  ({n_correct}/{len(records)}), parse errors: {n_parse}")
    return {"evaluation_scope": "full" if requested_n <= 0 else "limited", "accuracy": accuracy, "n_test": len(records),
            "n_correct": n_correct, "n_parse_errors": n_parse,
            "provenance": result_provenance(dataset_repository=DATASET_REPOSITORY, dataset_revision=single_source_revision(records), dataset_split="test",
                                            evaluator={**scorer_info, "primary_metric": "accuracy"}, n_tasks=len(records), max_new_tokens=max_new_tokens,
                                            thinking=False, sampling=sampling_kwargs(args), seed=getattr(args, "seed", None))}
