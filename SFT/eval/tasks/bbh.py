"""
BBH (Big Bench Hard) evaluation module.

Uses exact match accuracy as the primary metric.
BBH contains multiple reasoning tasks with short answer outputs.
"""

import json
import os
from typing import List, Tuple

from tqdm import tqdm

from ..utils import generate_completions


def load_bbh_test_data(data_dir: str, k: int = -1, task: str = None) -> List[Tuple[str, str, str]]:
    """
    Load BBH test data from unified JSONL format.

    Args:
        data_dir: Base data directory containing eval/bbh/
        k: Number of examples to load (-1 for all)
        task: Optional BBH task to filter by (e.g., "boolean_expressions")

    Returns:
        List of (prompt, reference_answer, task_name) tuples
    """
    file_path = os.path.join(data_dir, "eval", "bbh", "bbh_test_data.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"BBH test data not found: {file_path}\n"
            f"Please run: python SFT/data/prepare_datasets.py --datasets bbh"
        )

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())

            # Filter by task if specified
            if task is not None and example.get('task') != task:
                continue

            messages = example.get('messages', [])
            if len(messages) < 2:
                continue

            user_content = messages[0]['content']
            reference = messages[1]['content']
            task_name = example.get('task', 'unknown')

            # Format prompt with chat template
            prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"

            examples.append((prompt, reference, task_name))

            if k > 0 and len(examples) >= k:
                break

    return examples


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Strip whitespace and convert to lowercase
    answer = answer.strip().lower()
    # Remove common punctuation
    for char in ['.', ',', '!', '?', ';', ':']:
        answer = answer.replace(char, '')
    return answer


def compute_accuracy(args, model, tokenizer, batch_size: int = 1, max_new_tokens: int = 64) -> dict:
    """
    Evaluate model on BBH test set using exact match accuracy.

    Args:
        args: Arguments containing n_test and data_dir
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with accuracy and task breakdown
    """
    data_dir = getattr(args, 'data_dir', './data')
    n_test = getattr(args, 'n_test', -1)
    if n_test <= 0:
        n_test = 10000  # Load all available
    task_filter = getattr(args, 'bbh_task', None)

    # Load test data
    test_data = load_bbh_test_data(data_dir, k=n_test, task=task_filter)
    print(f"Loaded {len(test_data)} BBH test examples")

    # Extract prompts and references
    prompts = [prompt for prompt, _, _ in test_data]
    references = [ref for _, ref, _ in test_data]
    tasks = [task for _, _, task in test_data]

    # Generate answers
    print("Generating answers...")
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

    # Clean predictions and compute accuracy
    correct = 0
    task_correct = {}
    task_total = {}

    for i, (pred, ref, task) in enumerate(zip(predictions, references, tasks)):
        # Clean prediction - take first line/sentence
        pred = pred.strip()
        for stop_pattern in ["<|user|>", "<|assistant|>", "</s>", "<|end|>", "\n"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]
        pred = pred.strip()

        # Normalize for comparison
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)

        # Track per-task accuracy
        if task not in task_correct:
            task_correct[task] = 0
            task_total[task] = 0
        task_total[task] += 1

        if pred_norm == ref_norm:
            correct += 1
            task_correct[task] += 1

    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0

    # Print results
    print(f"\nBBH Evaluation Results:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Examples evaluated: {len(test_data)}")
    print(f"  Correct: {correct}")

    # Print per-task breakdown if multiple tasks
    if len(task_total) > 1:
        print(f"\n  Per-task breakdown:")
        for task in sorted(task_total.keys()):
            task_acc = task_correct[task] / task_total[task]
            print(f"    {task}: {task_acc:.4f} ({task_correct[task]}/{task_total[task]})")

    return {
        "accuracy": accuracy,
        "n_test": len(test_data),
        "n_correct": correct,
        "task_breakdown": {t: task_correct[t] / task_total[t] for t in task_total}
    }
