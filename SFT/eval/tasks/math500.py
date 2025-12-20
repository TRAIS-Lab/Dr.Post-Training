"""
MATH500 evaluation module for competition-level math problems.

Uses accuracy on the final answer as the primary metric.
Answers are extracted from the LaTeX format: \\boxed{answer}
"""

import json
import os
import re
from typing import List, Tuple, Optional

from tqdm import tqdm

from ..utils import generate_completions


def load_math500_test_data(data_dir: str, k: int = -1) -> List[Tuple[str, str, str, dict]]:
    """
    Load MATH500 test data from unified JSONL format.

    Args:
        data_dir: Base data directory containing eval/math500/
        k: Number of examples to load (-1 for all)

    Returns:
        List of (prompt, full_solution, final_answer, metadata) tuples
    """
    file_path = os.path.join(data_dir, "eval", "math500", "math500_test_data.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"MATH500 test data not found: {file_path}\n"
            f"Please run: python SFT/data/prepare_datasets.py --datasets math500"
        )

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            messages = example.get('messages', [])
            if len(messages) < 2:
                continue

            user_content = messages[0]['content']
            full_solution = messages[1]['content']
            metadata = example.get('metadata', {})

            # Extract final answer from \boxed{} format
            final_answer = extract_boxed_answer(full_solution)
            if final_answer is None:
                continue

            # Format prompt with chat template
            prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"

            examples.append((prompt, full_solution, final_answer, metadata))

            if k > 0 and len(examples) >= k:
                break

    return examples


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from \\boxed{...} format.

    Handles nested braces properly.
    """
    # Find \boxed{ pattern
    match = re.search(r'\\boxed\{', text)
    if not match:
        return None

    start = match.end()
    brace_count = 1
    i = start

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start:i-1]
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize mathematical answer for comparison.

    Handles common LaTeX formatting differences.
    """
    if answer is None:
        return ""

    # Remove whitespace
    answer = answer.strip()

    # Remove common LaTeX formatting
    answer = answer.replace('\\,', '')
    answer = answer.replace('\\;', '')
    answer = answer.replace('\\!', '')
    answer = answer.replace('\\ ', '')
    answer = answer.replace('\\text{', '')
    answer = answer.replace('\\mathrm{', '')
    answer = answer.replace('\\textbf{', '')
    answer = answer.replace('}', '')

    # Normalize fractions
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\\dfrac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\\tfrac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)

    # Remove spaces
    answer = answer.replace(' ', '')

    return answer.lower()


def answers_match(pred: str, ref: str) -> bool:
    """Check if predicted answer matches reference answer."""
    pred_norm = normalize_answer(pred)
    ref_norm = normalize_answer(ref)

    # Exact match after normalization
    if pred_norm == ref_norm:
        return True

    # Try numeric comparison if both are numbers
    try:
        pred_num = float(pred_norm)
        ref_num = float(ref_norm)
        if abs(pred_num - ref_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def compute_accuracy(args, model, tokenizer, batch_size: int = 1, max_new_tokens: int = 512) -> dict:
    """
    Evaluate model on MATH500 test set.

    Args:
        args: Arguments containing n_test and data_dir
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with accuracy and level breakdown
    """
    data_dir = getattr(args, 'data_dir', './data')
    n_test = getattr(args, 'n_test', -1)
    if n_test <= 0:
        n_test = 10000  # Load all available

    # Load test data
    test_data = load_math500_test_data(data_dir, k=n_test)
    print(f"Loaded {len(test_data)} MATH500 test examples")

    # Extract prompts
    prompts = [prompt for prompt, _, _, _ in test_data]
    references = [answer for _, _, answer, _ in test_data]
    metadata_list = [meta for _, _, _, meta in test_data]

    # Generate solutions
    print("Generating solutions...")
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

    # Evaluate predictions
    correct = 0
    invalid_format = 0
    level_correct = {}
    level_total = {}

    for i, (pred, ref, meta) in enumerate(zip(predictions, references, metadata_list)):
        level = meta.get('level', 0)

        # Track per-level stats
        if level not in level_correct:
            level_correct[level] = 0
            level_total[level] = 0
        level_total[level] += 1

        # Clean prediction
        pred = pred.strip()
        for stop_pattern in ["<|user|>", "<|assistant|>", "</s>", "<|end|>"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]

        # Extract predicted answer
        pred_answer = extract_boxed_answer(pred)

        if pred_answer is None:
            invalid_format += 1
            continue

        # Compare answers
        if answers_match(pred_answer, ref):
            correct += 1
            level_correct[level] += 1

    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0

    # Print results
    print(f"\nMATH500 Evaluation Results:")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Examples evaluated: {len(test_data)}")
    print(f"  Correct: {correct}")
    print(f"  Invalid format: {invalid_format}")

    # Print per-level breakdown
    if len(level_total) > 1:
        print(f"\n  Per-level breakdown:")
        for level in sorted(level_total.keys()):
            level_acc = level_correct[level] / level_total[level] if level_total[level] > 0 else 0
            print(f"    Level {level}: {level_acc:.4f} ({level_correct[level]}/{level_total[level]})")

    return {
        "accuracy": accuracy,
        "n_test": len(test_data),
        "n_correct": correct,
        "n_invalid_format": invalid_format,
        "level_breakdown": {l: level_correct[l] / level_total[l] for l in level_total if level_total[l] > 0}
    }
