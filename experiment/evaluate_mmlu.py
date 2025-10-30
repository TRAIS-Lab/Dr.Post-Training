#!/usr/bin/env python
"""
Standalone script to evaluate a trained model on MMLU dataset.

Usage:
    python evaluate_mmlu.py \
        --model_path ./out/GREATS-llama3-1b-p-lora-seed3 \
        --subject sociology \
        --n_val 5 \
        --n_test 500 \
        --output_file ./out/GREATS-llama3-1b-p-lora-seed3/mmlu_results.json
"""

import argparse
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.train.mmlu_eval import compute_accuracy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path, base_model=None):
    """
    Load trained model and tokenizer.

    Args:
        model_path: Path to the trained model (can be a LoRA adapter)
        base_model: Base model name/path if model_path is a LoRA adapter

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Check if this is a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # This is a LoRA adapter
        logger.info("Detected LoRA adapter")

        # Load adapter config to get base model
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        if base_model is None:
            base_model = adapter_config.get("base_model_name_or_path")
            if base_model is None:
                raise ValueError(
                    "Could not determine base model from adapter config. "
                    "Please specify --base_model"
                )

        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )

        logger.info(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge adapter weights into base model

    else:
        # This is a full model
        logger.info("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    model.eval()
    logger.info("Model loaded successfully")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on MMLU dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (can be LoRA adapter or full model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name/path (required if model_path is a LoRA adapter and base model cannot be inferred)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sociology",
        help="MMLU subject to evaluate on",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=5,
        help="Number of validation examples for in-context learning",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=500,
        help="Number of evaluation examples to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON file (default: model_path/mmlu_results.json)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing MMLU data",
    )

    args = parser.parse_args()

    # Set output file path
    if args.output_file is None:
        args.output_file = os.path.join(args.model_path, "mmlu_results.json")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Get answer choice token IDs
    choices = ["A", "B", "C", "D"]
    answer_choice_ids = [
        tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1]
        for answer_choice in choices
    ]

    logger.info(f"Evaluating on MMLU subject: {args.subject}")
    logger.info(f"Number of validation examples: {args.n_val}")
    logger.info(f"Number of evaluation examples: {args.n_eval}")

    # Run evaluation
    cors, accuracy, all_probs = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        answer_choice_ids=answer_choice_ids,
        batch_size=args.batch_size,
    )

    # Save results
    results = {
        "subject": args.subject,
        "n_val": args.n_val,
        "n_eval": args.n_eval,
        "accuracy": accuracy,
        "num_correct": cors.sum().item(),
        "num_total": len(cors),
    }

    logger.info(f"Accuracy: {accuracy:.4f} ({cors.sum().item()}/{len(cors)})")

    # Save to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
