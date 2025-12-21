"""
BBH (Big Bench Hard) evaluation module.

Uses exact match accuracy as the primary metric.
BBH contains multiple reasoning tasks with short answer outputs.
"""

import json
import os
import re
from typing import Dict, List, Tuple

from ..utils import generate_completions


def load_fewshot_examples(data_dir: str) -> Dict[str, str]:
    """
    Load few-shot examples for each BBH task.

    Args:
        data_dir: Base data directory containing eval/bbh/

    Returns:
        Dictionary mapping task name to few-shot prompt string
    """
    fewshot_path = os.path.join(data_dir, "eval", "bbh", "bbh_fewshot.json")

    if not os.path.exists(fewshot_path):
        print(f"Warning: Few-shot examples not found at {fewshot_path}")
        return {}

    with open(fewshot_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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

            examples.append((user_content, reference, task_name))

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


def extract_answer(prediction: str) -> str:
    """
    Extract the final answer from a prediction.
    Looks for patterns like "So the answer is X." or "the answer is X"
    """
    # Try to find "So the answer is X" or "the answer is X" pattern
    patterns = [
        r"[Ss]o the answer is[:\s]*([^\.\n]+)",
        r"[Tt]he answer is[:\s]*([^\.\n]+)",
        r"[Aa]nswer[:\s]*([^\.\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()

    # If no pattern found, return the last line/sentence
    lines = prediction.strip().split('\n')
    return lines[-1].strip() if lines else prediction


def compute_accuracy(args, model, tokenizer, batch_size: int = 1, max_new_tokens: int = 512) -> dict:
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

    # Load few-shot examples
    fewshot_examples = load_fewshot_examples(data_dir)

    # Load test data
    test_data = load_bbh_test_data(data_dir, k=n_test, task=task_filter)
    print(f"Loaded {len(test_data)} BBH test examples")

    # Build prompts with few-shot examples
    prompts = []
    for user_content, _, task_name in test_data:
        # Get few-shot examples for this task
        fewshot = fewshot_examples.get(task_name, "")

        if fewshot:
            # Format: few-shot examples followed by the test question
            prompt = f"<|user|>\n{fewshot}\n\nQ: {user_content}\nA: Let's think step by step.\n<|assistant|>\n"
        else:
            # No few-shot examples available
            prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"

        prompts.append(prompt)

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
        do_sample=False,  # Use greedy decoding for consistency
        disable_tqdm=False
    )

    # Clean predictions and compute accuracy
    correct = 0
    task_correct = {}
    task_total = {}

    for pred, ref, task in zip(predictions, references, tasks):
        # Clean prediction
        pred = pred.strip()

        # Remove any trailing special tokens
        for stop_pattern in ["<|user|>", "<|assistant|>", "</s>", "<|end|>", "<|eot_id|>"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]
        pred = pred.strip()

        # Extract final answer from the prediction
        pred_answer = extract_answer(pred)

        # Normalize for comparison
        pred_norm = normalize_answer(pred_answer)
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
