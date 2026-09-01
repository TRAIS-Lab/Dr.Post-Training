"""
SamSUM evaluation module for dialogue summarization.

Uses ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) for evaluation.
"""

import json
import os
from typing import List, Tuple

import evaluate
from tqdm import tqdm

from SFT.data.get_val_dataset import render_chat
from ..utils import generate_completions, get_eos_token_ids

# Load ROUGE metric
rouge_metric = evaluate.load("rouge")

# BERTScore: lazy-loaded so import doesn't pay the model-download cost.
_bertscore_metric = None
def _get_bertscore():
    global _bertscore_metric
    if _bertscore_metric is None:
        _bertscore_metric = evaluate.load("bertscore")
    return _bertscore_metric


def load_samsum_test_data(tokenizer, data_dir: str, k: int = -1) -> List[Tuple[str, str]]:
    """Load SamSUM test data and render prompts with the model's chat template."""
    file_path = os.path.join(data_dir, "eval", "samsum", "samsum_test_data.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SamSUM test data not found: {file_path}\n"
            f"Please run: python SFT/data/prepare_datasets.py --datasets samsum"
        )

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            messages = example.get('messages', [])
            if len(messages) < 2:
                continue

            user_content = messages[0]['content']
            reference = messages[1]['content']

            prompt = render_chat(tokenizer, user_content)
            examples.append((prompt, reference))

            if k > 0 and len(examples) >= k:
                break

    return examples


def compute_rouge_scores(predictions: List[str], references: List[str]) -> dict:
    """
    Compute ROUGE scores between predictions and references.

    Args:
        predictions: List of generated summaries
        references: List of reference summaries

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results.get("rougeLsum", results["rougeL"])
    }


def compute_bertscore(predictions: List[str], references: List[str], lang: str = "en") -> dict:
    """Mean BERTScore P/R/F1 across the corpus (rescaled with baseline).

    SamSum is English; using lang="en" lets bert_score pick a sensible default
    encoder (currently roberta-large) and the matching baseline-rescale tensor.
    """
    metric = _get_bertscore()
    res = metric.compute(
        predictions=predictions,
        references=references,
        lang=lang,
        rescale_with_baseline=True,
    )
    return {
        "bertscore_p":  float(sum(res["precision"]) / len(res["precision"])),
        "bertscore_r":  float(sum(res["recall"])    / len(res["recall"])),
        "bertscore_f1": float(sum(res["f1"])        / len(res["f1"])),
    }


def compute_accuracy(
    args,
    model,
    tokenizer,
    batch_size: int = 1,
    max_new_tokens: int = 128
) -> dict:
    """
    Evaluate model on SamSUM test set using ROUGE metrics.

    Args:
        args: Arguments containing n_test and data_dir
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with ROUGE scores
    """
    data_dir = getattr(args, 'data_dir', './data')
    n_test = getattr(args, 'n_test', -1)

    test_data = load_samsum_test_data(tokenizer, data_dir, k=n_test)
    print(f"Loaded {len(test_data)} SamSUM test examples")

    prompts = [prompt for prompt, _ in test_data]
    references = [ref for _, ref in test_data]

    print("Generating summaries...")
    predictions = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_token_ids(tokenizer),
        temperature=1.0,
        top_p=0.95,
        disable_tqdm=False
    )

    # Clean predictions: strip anything past the assistant turn delimiter.
    # Generated text is decoded with skip_special_tokens=True, so <|im_end|>
    # / <|im_start|> won't usually appear, but keep them as defensive fallbacks
    # in case the user happens to inspect raw outputs.
    cleaned_predictions = []
    for pred in predictions:
        pred = pred.strip()
        for stop_pattern in ["<|im_end|>", "<|im_start|>"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]
        cleaned_predictions.append(pred.strip())

    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(cleaned_predictions, references)

    # Compute BERTScore (semantic-similarity sanity check alongside ROUGE).
    # Failure shouldn't break the eval — fall back gracefully.
    try:
        bertscores = compute_bertscore(cleaned_predictions, references, lang="en")
    except Exception as e:
        print(f"  [warn] BERTScore failed: {e}")
        bertscores = {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}

    print(f"\nSamSUM Evaluation Results:")
    print(f"  ROUGE-1:       {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2:       {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L:       {rouge_scores['rougeL']:.4f}")
    print(f"  ROUGE-Lsum:    {rouge_scores['rougeLsum']:.4f}")
    if bertscores["bertscore_f1"] is not None:
        print(f"  BERTScore F1:  {bertscores['bertscore_f1']:.4f}"
              f"  (P={bertscores['bertscore_p']:.4f}, R={bertscores['bertscore_r']:.4f})")
    print(f"  Examples evaluated: {len(test_data)}")

    return {**rouge_scores, **bertscores}


def evaluate_samsum(
    model,
    tokenizer,
    data_dir: str = "./data",
    n_test: int = -1,
    batch_size: int = 1,
    max_new_tokens: int = 128
) -> dict:
    """
    Standalone function to evaluate SamSUM without args object.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        data_dir: Path to data directory
        n_test: Number of test examples (-1 for all)
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with evaluation results
    """
    # Create a simple namespace object
    class Args:
        pass

    args = Args()
    args.data_dir = data_dir
    args.n_test = n_test

    return compute_accuracy(args, model, tokenizer, batch_size, max_new_tokens)
