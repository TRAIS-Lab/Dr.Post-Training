"""
SamSUM evaluation module for dialogue summarization.

Uses ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) for evaluation.
"""

import json
import os
from typing import List, Tuple

import evaluate
from tqdm import tqdm

from ..utils import generate_completions

# Load ROUGE metric
rouge_metric = evaluate.load("rouge")


def load_samsum_test_data(data_dir: str, k: int = -1) -> List[Tuple[str, str]]:
    """
    Load SamSUM test data from unified JSONL format.

    Args:
        data_dir: Base data directory containing eval/samsum/
        k: Number of examples to load (-1 for all)

    Returns:
        List of (prompt, reference_summary) tuples
    """
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

            # Format prompt with chat template
            prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"

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

    # Load test data
    test_data = load_samsum_test_data(data_dir, k=n_test)
    print(f"Loaded {len(test_data)} SamSUM test examples")

    # Extract prompts and references
    prompts = [prompt for prompt, _ in test_data]
    references = [ref for _, ref in test_data]

    # Generate summaries
    print("Generating summaries...")
    predictions = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=1.0,
        top_p=0.95,
        disable_tqdm=False
    )

    # Clean predictions
    cleaned_predictions = []
    for pred in predictions:
        # Remove any trailing special tokens or repeated content
        pred = pred.strip()
        # Stop at common ending patterns
        for stop_pattern in ["<|user|>", "<|assistant|>", "</s>", "<|end|>"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]
        cleaned_predictions.append(pred.strip())

    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(cleaned_predictions, references)

    # Print results
    print(f"\nSamSUM Evaluation Results:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"  Examples evaluated: {len(test_data)}")

    return rouge_scores


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
