"""
GSM8K evaluation module for grade school math problems.

Uses accuracy on the final numerical answer as the primary metric.
Answers are extracted from the format: #### <number>
"""

import json
import os
import re
from typing import List, Tuple, Optional

from tqdm import tqdm

from ..utils import generate_completions


def load_gsm8k_test_data(data_dir: str, k: int = -1) -> List[Tuple[str, str, str]]:
    """
    Load GSM8K test data from unified JSONL format.

    Args:
        data_dir: Base data directory containing eval/gsm8k/
        k: Number of examples to load (-1 for all)

    Returns:
        List of (prompt, full_solution, final_answer) tuples
    """
    file_path = os.path.join(data_dir, "eval", "gsm8k", "gsm8k_test_data.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"GSM8K test data not found: {file_path}\n"
            f"Please run: python SFT/data/prepare_datasets.py --datasets gsm8k"
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

            # Use answer from metadata if available, otherwise extract from #### format
            final_answer = metadata.get('answer', '')
            if not final_answer:
                final_answer = extract_answer(full_solution)
            if final_answer is None:
                continue

            # Format prompt with chat template
            prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"

            examples.append((prompt, full_solution, final_answer))

            if k > 0 and len(examples) >= k:
                break

    return examples


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from GSM8K format.

    GSM8K answers end with #### followed by the numerical answer.
    """
    # Look for #### pattern
    match = re.search(r'####\s*(-?[\d,\.]+)', text)
    if match:
        # Remove commas from numbers
        answer = match.group(1).replace(',', '')
        return answer
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """Normalize a number string to a float for comparison."""
    try:
        # Remove commas and extra whitespace
        num_str = num_str.strip().replace(',', '')
        return float(num_str)
    except (ValueError, TypeError):
        return None


def compute_accuracy(args, model, tokenizer, batch_size: int = 1, max_new_tokens: int = 256) -> dict:
    """
    Evaluate model on GSM8K test set using final answer accuracy.

    Args:
        args: Arguments containing n_test and data_dir
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with accuracy and evaluation stats
    """
    data_dir = getattr(args, 'data_dir', './data')
    n_test = getattr(args, 'n_test', -1)
    if n_test <= 0:
        n_test = 10000  # Load all available

    # Load test data
    test_data = load_gsm8k_test_data(data_dir, k=n_test)
    print(f"Loaded {len(test_data)} GSM8K test examples")

    # Extract prompts
    prompts = [prompt for prompt, _, _ in test_data]
    references = [answer for _, _, answer in test_data]

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

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Clean prediction
        pred = pred.strip()
        for stop_pattern in ["<|user|>", "<|assistant|>", "</s>", "<|end|>"]:
            if stop_pattern in pred:
                pred = pred[:pred.find(stop_pattern)]

        # Extract predicted answer
        pred_answer = extract_answer(pred)

        if pred_answer is None:
            # Try to find any number at the end as fallback
            numbers = re.findall(r'-?[\d,\.]+', pred)
            if numbers:
                pred_answer = numbers[-1].replace(',', '')
            else:
                invalid_format += 1
                continue

        # Compare numerically
        pred_num = normalize_number(pred_answer)
        ref_num = normalize_number(ref)

        if pred_num is not None and ref_num is not None:
            if abs(pred_num - ref_num) < 1e-6:
                correct += 1

    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0

    # Print results
    print(f"\nGSM8K Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Examples evaluated: {len(test_data)}")
    print(f"  Correct: {correct}")
    print(f"  Invalid format: {invalid_format}")

    return {
        "accuracy": accuracy,
        "n_test": len(test_data),
        "n_correct": correct,
        "n_invalid_format": invalid_format,
    }
