#!/usr/bin/env python
"""
Unified evaluation script for SFT experiments.

Supports:
- SamSUM: Dialogue summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- TyDiQA: Multilingual QA (F1 score)
- MMLU: Multiple-choice QA (Accuracy)
- BBH: Big Bench Hard reasoning tasks (Accuracy)
- GSM8K: Grade school math (Accuracy)
- MATH500: Competition math (Accuracy)
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device():
    """Get the appropriate CUDA device (respects CUDA_VISIBLE_DEVICES)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model_and_tokenizer(model_path: str, base_model: Optional[str] = None):
    """Load trained model and tokenizer."""
    device = get_device()
    logger.info(f"Loading model from {model_path} to {device}")

    # Check if tokenizer exists in model_path
    tokenizer_path = model_path
    if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            tokenizer_path = adapter_config.get("base_model_name_or_path", model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Check if this is a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        logger.info("Detected LoRA adapter")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        if base_model is None:
            base_model = adapter_config.get("base_model_name_or_path")
            if base_model is None:
                raise ValueError("Could not determine base model. Please specify --base_model")

        logger.info(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        ).to(device)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)

    model.eval()

    # Check for NaN/Inf in model weights
    nan_inf_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_inf_params.append(name)
    if nan_inf_params:
        logger.warning(f"Model contains NaN/Inf values in {len(nan_inf_params)} parameters")
        if len(nan_inf_params) > 10:
            # Model is severely corrupted - raise error to skip
            raise ValueError(f"Model is corrupted: NaN/Inf in {len(nan_inf_params)} parameters (training diverged)")

    logger.info("Model loaded successfully")
    return model, tokenizer


def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse model name to extract experiment configuration."""
    config = {
        "model_name": model_name,
        "train_dataset": "",
        "eval_task": "",
        "selection": "",
        "compression": "",
        "model": "",
        "training_type": "",
        "percentage": "",
        "learning_rate": "",
        "batch_size": "",
        "n_val": "",
        "seed": "",
    }

    parts = model_name.split("-")
    if len(parts) >= 6:
        train_task = parts[0].split("_")
        if len(train_task) >= 2:
            config["train_dataset"] = train_task[0]
            config["eval_task"] = train_task[1]
        else:
            config["train_dataset"] = parts[0]

        config["selection"] = parts[1]

        idx = 2
        if parts[idx] == "LoGra" and len(parts) > idx + 1 and parts[idx + 1] == "2nd":
            config["compression"] = "LoGra-2nd"
            idx = 4
        else:
            config["compression"] = parts[idx]
            idx = 3

        if idx < len(parts):
            config["model"] = parts[idx]
            idx += 1
        if idx < len(parts):
            config["training_type"] = parts[idx]
            idx += 1

        for part in parts[idx:]:
            if part.startswith("p") and "." in part:
                config["percentage"] = part[1:]
            elif part.startswith("lr"):
                config["learning_rate"] = part[2:]
            elif part.startswith("b") and part[1:].isdigit():
                config["batch_size"] = part[1:]
            elif part.startswith("v") and part[1:].isdigit():
                config["n_val"] = part[1:]
            elif part.startswith("s") and part[1:].isdigit():
                config["seed"] = part[1:]

    return config


def find_models(models_dir: str, train_dataset: Optional[str] = None) -> List[str]:
    """Find all model directories, optionally filtering by prefix pattern."""
    model_paths = []
    for entry in os.listdir(models_dir):
        entry_path = os.path.join(models_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        has_model = (
            os.path.exists(os.path.join(entry_path, "config.json")) or
            os.path.exists(os.path.join(entry_path, "adapter_config.json"))
        )
        if has_model:
            if train_dataset is None:
                model_paths.append(entry_path)
            else:
                # Match if entry starts with the filter pattern (followed by - or _)
                if entry.startswith(train_dataset + "-") or entry.startswith(train_dataset + "_"):
                    model_paths.append(entry_path)
    return sorted(model_paths)


def evaluate_samsum(args, model, tokenizer) -> dict:
    """Run SamSUM evaluation."""
    from .tasks.samsum import compute_accuracy

    logger.info("Evaluating on SamSUM")
    rouge_scores = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    return {
        "task": "samsum",
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
    }


def evaluate_tydiqa(args, model, tokenizer) -> dict:
    """Run TyDiQA evaluation."""
    from .tasks.tydiqa import compute_accuracy

    logger.info("Evaluating on TyDiQA")
    results = compute_accuracy(args=args, model=model, tokenizer=tokenizer)
    return {
        "task": "tydiqa",
        "f1_score": results["f1_score"],
        "exact_match": results["exact_match"],
        "n_test": results["n_test"],
    }


def evaluate_mmlu(args, model, tokenizer) -> dict:
    """Run MMLU evaluation."""
    from .tasks.mmlu import compute_accuracy

    logger.info("Evaluating on MMLU")
    results = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )
    return {
        "task": "mmlu",
        "accuracy": results["accuracy"],
        "n_test": results["n_test"],
    }


def evaluate_bbh(args, model, tokenizer) -> dict:
    """Run BBH (Big Bench Hard) evaluation."""
    from .tasks.bbh import compute_accuracy

    logger.info("Evaluating on BBH")
    results = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=512  # Chain-of-thought requires more tokens
    )
    return {
        "task": "bbh",
        "accuracy": results["accuracy"],
        "n_test": results["n_test"],
    }


def evaluate_gsm8k(args, model, tokenizer) -> dict:
    """Run GSM8K evaluation."""
    from .tasks.gsm8k import compute_accuracy

    logger.info("Evaluating on GSM8K")
    results = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=256
    )
    return {
        "task": "gsm8k",
        "accuracy": results["accuracy"],
        "n_test": results["n_test"],
    }


def evaluate_math500(args, model, tokenizer) -> dict:
    """Run MATH500 evaluation."""
    from .tasks.math500 import compute_accuracy

    logger.info("Evaluating on MATH500")
    results = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=512
    )
    return {
        "task": "math500",
        "accuracy": results["accuracy"],
        "n_test": results["n_test"],
    }


def get_task_from_model_name(model_name: str) -> Optional[str]:
    """Extract the evaluation task from model name (e.g., alpaca_samsum -> samsum)."""
    valid_tasks = ["samsum", "tydiqa", "mmlu", "bbh", "gsm8k", "math500"]
    parts = model_name.split("-")
    if parts:
        train_task = parts[0].split("_")
        if len(train_task) >= 2:
            task = train_task[1].lower()
            if task in valid_tasks:
                return task
    return None


def evaluate_model(
    model_path: str,
    data_dir: str,
    n_test: int = -1,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    base_model: Optional[str] = None,
    task_override: Optional[str] = None,
    subject: Optional[str] = None,
) -> Dict:
    """Evaluate a single model. Task is auto-detected from model name."""
    model_name = os.path.basename(model_path)
    results = parse_model_name(model_name)
    results["model_path"] = model_path

    # Auto-detect task from model name, or use override
    task = task_override or get_task_from_model_name(model_name)
    if task is None:
        logger.error(f"Could not detect task from model name: {model_name}")
        results["error"] = "Could not detect task from model name"
        return results

    logger.info(f"Auto-detected task: {task}")

    try:
        model, tokenizer = load_model_and_tokenizer(model_path, base_model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        results["error"] = str(e)
        return results

    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.n_test = n_test
    args.batch_size = batch_size
    args.max_new_tokens = max_new_tokens
    args.subject = subject  # For MMLU
    args.bbh_task = subject  # For BBH (uses same value)
    args.n_val = 5  # Few-shot examples for MMLU

    try:
        if task == "samsum":
            logger.info(f"Evaluating {model_name} on SamSUM...")
            samsum_results = evaluate_samsum(args, model, tokenizer)
            results["samsum_rouge1"] = samsum_results["rouge1"]
            results["samsum_rouge2"] = samsum_results["rouge2"]
            results["samsum_rougeL"] = samsum_results["rougeL"]

            with open(os.path.join(model_path, "samsum_results.json"), "w") as f:
                json.dump(samsum_results, f, indent=2)

        elif task == "tydiqa":
            logger.info(f"Evaluating {model_name} on TyDiQA...")
            tydiqa_results = evaluate_tydiqa(args, model, tokenizer)
            results["tydiqa_f1"] = tydiqa_results["f1_score"]
            results["tydiqa_em"] = tydiqa_results["exact_match"]

            with open(os.path.join(model_path, "tydiqa_results.json"), "w") as f:
                json.dump(tydiqa_results, f, indent=2)

        elif task == "mmlu":
            logger.info(f"Evaluating {model_name} on MMLU...")
            mmlu_results = evaluate_mmlu(args, model, tokenizer)
            results["mmlu_accuracy"] = mmlu_results["accuracy"]

            with open(os.path.join(model_path, "mmlu_results.json"), "w") as f:
                json.dump(mmlu_results, f, indent=2)

        elif task == "bbh":
            logger.info(f"Evaluating {model_name} on BBH...")
            bbh_results = evaluate_bbh(args, model, tokenizer)
            results["bbh_accuracy"] = bbh_results["accuracy"]

            with open(os.path.join(model_path, "bbh_results.json"), "w") as f:
                json.dump(bbh_results, f, indent=2)

        elif task == "gsm8k":
            logger.info(f"Evaluating {model_name} on GSM8K...")
            gsm8k_results = evaluate_gsm8k(args, model, tokenizer)
            results["gsm8k_accuracy"] = gsm8k_results["accuracy"]

            with open(os.path.join(model_path, "gsm8k_results.json"), "w") as f:
                json.dump(gsm8k_results, f, indent=2)

        elif task == "math500":
            logger.info(f"Evaluating {model_name} on MATH500...")
            math500_results = evaluate_math500(args, model, tokenizer)
            results["math500_accuracy"] = math500_results["accuracy"]

            with open(os.path.join(model_path, "math500_results.json"), "w") as f:
                json.dump(math500_results, f, indent=2)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        results["error"] = str(e)

    del model
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
            # Try to reset CUDA state
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    parser = argparse.ArgumentParser(description="SFT Evaluation Script")

    # Model selection (batch is default)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--models_dir", type=str,
        default="/scratch/pbb/Project/Gradient-Streaming/SFT",
        help="Directory containing trained models (default)")
    model_group.add_argument("--model_path", type=str,
        help="Path to single model to evaluate")

    parser.add_argument("--train", type=str, default=None,
        help="Filter by training dataset (e.g., alpaca, less, tulu3, wizardlm)")
    parser.add_argument("--task", type=str, default=None,
        choices=["samsum", "tydiqa", "mmlu", "bbh", "gsm8k", "math500"],
        help="Override auto-detected task (optional)")
    parser.add_argument("--subject", type=str, default=None,
        help="MMLU subject or BBH task to evaluate on (default: all)")
    parser.add_argument("--data_dir", type=str, default=None,
        help="Data directory (default: auto-detect)")
    parser.add_argument("--n_test", type=int, default=-1,
        help="Number of test examples (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
        help="Maximum tokens to generate")
    parser.add_argument("--base_model", type=str, default=None,
        help="Base model for LoRA adapters")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Auto-detect data directory
    if args.data_dir is None:
        sft_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        args.data_dir = sft_data_dir if os.path.exists(sft_data_dir) else "./data"

    logger.info(f"Data directory: {args.data_dir}")

    # Single model evaluation
    if args.model_path:
        results = evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            n_test=args.n_test,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            base_model=args.base_model,
            task_override=args.task,
            subject=args.subject,
        )
        print("\n" + "=" * 60)
        print("Results:")
        for k, v in results.items():
            if k not in ["model_path", "timestamp", "model_name"]:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return

    # Batch evaluation
    # Construct filter prefix from train, task, and subject
    filter_prefix = args.train
    if filter_prefix and args.task:
        filter_prefix = f"{filter_prefix}_{args.task}"
        if args.subject:
            filter_prefix = f"{filter_prefix}_{args.subject}"
    model_paths = find_models(args.models_dir, filter_prefix)
    logger.info(f"Found {len(model_paths)} models to evaluate")

    if not model_paths:
        logger.error(f"No models found in {args.models_dir}")
        sys.exit(1)

    all_results = []
    for i, model_path in enumerate(model_paths):
        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(model_paths)}] {os.path.basename(model_path)}")
        print(f"{'=' * 70}")

        results = evaluate_model(
            model_path=model_path,
            data_dir=args.data_dir,
            n_test=args.n_test,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            base_model=args.base_model,
            task_override=args.task,
            subject=args.subject,
        )
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    # SamSUM results
    samsum_results = [r for r in all_results if "samsum_rougeL" in r]
    if samsum_results:
        print(f"\nSamSUM Results:")
        print(f"{'Model':<70} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
        print("-" * 100)
        for r in sorted(samsum_results, key=lambda x: x.get("samsum_rougeL", 0), reverse=True):
            print(f"{r['model_name'][:70]:<70} "
                  f"{r['samsum_rouge1']:>7.4f} {r['samsum_rouge2']:>7.4f} {r['samsum_rougeL']:>7.4f}")

    # TyDiQA results
    tydiqa_results = [r for r in all_results if "tydiqa_f1" in r]
    if tydiqa_results:
        print(f"\nTyDiQA Results:")
        print(f"{'Model':<80} {'F1':>8} {'EM':>8}")
        print("-" * 100)
        for r in sorted(tydiqa_results, key=lambda x: x.get("tydiqa_f1", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['tydiqa_f1']:>8.4f} {r.get('tydiqa_em', 0):>8.4f}")

    # MMLU results
    mmlu_results = [r for r in all_results if "mmlu_accuracy" in r]
    if mmlu_results:
        print(f"\nMMLU Results:")
        print(f"{'Model':<90} {'Acc':>8}")
        print("-" * 100)
        for r in sorted(mmlu_results, key=lambda x: x.get("mmlu_accuracy", 0), reverse=True):
            print(f"{r['model_name'][:90]:<90} "
                  f"{r['mmlu_accuracy']:>8.4f}")

    # BBH results
    bbh_results = [r for r in all_results if "bbh_accuracy" in r]
    if bbh_results:
        print(f"\nBBH Results:")
        print(f"{'Model':<90} {'Acc':>8}")
        print("-" * 100)
        for r in sorted(bbh_results, key=lambda x: x.get("bbh_accuracy", 0), reverse=True):
            print(f"{r['model_name'][:90]:<90} "
                  f"{r['bbh_accuracy']:>8.4f}")

    # GSM8K results
    gsm8k_results = [r for r in all_results if "gsm8k_accuracy" in r]
    if gsm8k_results:
        print(f"\nGSM8K Results:")
        print(f"{'Model':<90} {'Acc':>8}")
        print("-" * 100)
        for r in sorted(gsm8k_results, key=lambda x: x.get("gsm8k_accuracy", 0), reverse=True):
            print(f"{r['model_name'][:90]:<90} "
                  f"{r['gsm8k_accuracy']:>8.4f}")

    # MATH500 results
    math500_results = [r for r in all_results if "math500_accuracy" in r]
    if math500_results:
        print(f"\nMATH500 Results:")
        print(f"{'Model':<90} {'Acc':>8}")
        print("-" * 100)
        for r in sorted(math500_results, key=lambda x: x.get("math500_accuracy", 0), reverse=True):
            print(f"{r['model_name'][:90]:<90} "
                  f"{r['math500_accuracy']:>8.4f}")

    errors = [r for r in all_results if r.get("error")]
    if errors:
        print(f"\nErrors: {len(errors)} models failed")


if __name__ == "__main__":
    main()
