#!/usr/bin/env python
"""
Unified evaluation script for MMLU and TyDiQA datasets.

Usage:
    # MMLU evaluation
    python evaluate.py \
        --task mmlu \
        --model_path ./out/model \
        --subject sociology \
        --n_val 5 \
        --n_eval 500

    # TyDiQA evaluation
    python evaluate.py \
        --task tydiqa \
        --model_path ./out/model \
        --n_test 100
"""

import argparse
import json
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmlu import compute_accuracy as compute_mmlu_accuracy
from tydiqa import compute_accuracy as compute_tydiqa_accuracy

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

        # Resize embeddings if tokenizer has additional tokens
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            logger.info(
                f"Resizing model embeddings from {embedding_size} to {len(tokenizer)} "
                f"to match tokenizer vocabulary size"
            )
            model.resize_token_embeddings(len(tokenizer))

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


def evaluate_mmlu(args, model, tokenizer):
    """Run MMLU evaluation."""
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
    cors, accuracy, all_probs = compute_mmlu_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        answer_choice_ids=answer_choice_ids,
        batch_size=args.batch_size,
    )

    # Prepare results
    results = {
        "task": "mmlu",
        "subject": args.subject,
        "n_val": args.n_val,
        "n_eval": args.n_eval,
        "accuracy": accuracy,
        "num_correct": cors.sum().item(),
        "num_total": len(cors),
    }

    logger.info(f"Accuracy: {accuracy:.4f} ({cors.sum().item()}/{len(cors)})")

    return results


def evaluate_tydiqa(args, model, tokenizer):
    """Run TyDiQA evaluation."""
    logger.info(f"Evaluating on TyDiQA")
    logger.info(f"Number of test examples: {args.n_test}")

    # Run evaluation
    f1_score = compute_tydiqa_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
    )

    # Prepare results
    results = {
        "task": "tydiqa",
        "n_test": args.n_test,
        "f1_score": f1_score,
    }

    logger.info(f"F1 Score: {f1_score:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for MMLU and TyDiQA"
    )

    # Common arguments
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mmlu", "tydiqa"],
        help="Evaluation task to run",
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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON file (default: model_path/{task}_results.json)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing evaluation data",
    )

    # MMLU-specific arguments
    parser.add_argument(
        "--subject",
        type=str,
        default="sociology",
        help="MMLU subject to evaluate on (only for --task mmlu)",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=5,
        help="Number of validation examples for in-context learning (only for --task mmlu)",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=500,
        help="Number of evaluation examples to evaluate (only for --task mmlu)",
    )

    # TyDiQA-specific arguments
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of test examples to evaluate (only for --task tydiqa)",
    )

    args = parser.parse_args()

    # Set output file path
    if args.output_file is None:
        args.output_file = os.path.join(args.model_path, f"{args.task}_results.json")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Run appropriate evaluation
    if args.task == "mmlu":
        results = evaluate_mmlu(args, model, tokenizer)
    elif args.task == "tydiqa":
        results = evaluate_tydiqa(args, model, tokenizer)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
