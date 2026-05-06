#!/usr/bin/env python
"""
Unified evaluation script for SFT experiments.

Supports:
- SamSUM: Dialogue summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- TyDiQA: Multilingual QA (F1, EM)
- NQ-open: Closed-book factoid QA (EM, F1)
- SQuAD: Closed-book reading-comprehension QA, no context (EM, F1)
"""

import argparse
import json
import logging
import os
import random
import re
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
    from SFT.data.get_val_dataset import ensure_chat_template
    ensure_chat_template(tokenizer)

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


# Output dir naming convention:
#   main:         {train}_{task}-{model}-{curation}-{finetuning}-p{pct}-lr{lr}-b{bs}-v{nv}-s{seed}
#   target-only:  {task}_val_{task}-{model}-{curation}-{finetuning}-ms{steps}-lr{lr}-b{bs}-v{nv}-s{seed}
# Both {model} (e.g. "Llama-3.2-1B") and {curation}-{finetuning} (e.g. "FullTraining-LoRA")
# contain hyphens, so positional split-on-"-" parsing is wrong. Anchor on the
# fixed suffix tokens (-p|-ms, -lr, -b, -v, -s) instead.
_NAME_RE = re.compile(
    r"^"
    r"(?P<prefix>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"
    r"-(?P<model>.+?)"
    r"-(?P<curation>FullTraining|GlobalSubset|LayerWiseSubset|Standard)"
    r"-(?P<finetuning>MeSO-LoRA|Full|LoRA|MeSO)"
    r"-(?:p(?P<percentage>[\d.]+)|ms(?P<max_steps>\d+))"
    r"-lr(?P<learning_rate>[\d.]+e-?\d+)"
    r"-b(?P<batch_size>\d+)"
    r"-v(?P<n_val>\d+)"
    r"-s(?P<seed>\d+)"
    r"$"
)


def parse_model_name(model_name: str) -> Dict[str, str]:
    """Parse a model output-dir name into experiment fields. Cosmetic — used
    only for the printed eval summary and the master log row.
    """
    config = {
        "model_name": model_name,
        "train_dataset": "", "eval_task": "",
        "model": "",
        "selection": "", "training_type": "",
        "percentage": "", "max_steps": "",
        "learning_rate": "", "batch_size": "", "n_val": "", "seed": "",
    }
    m = _NAME_RE.match(model_name)
    if not m:
        return config
    g = m.groupdict()

    # Prefix splits as either "{train}_{task}" (main) or "{task}_val_{task}" (target-only)
    prefix_parts = g["prefix"].split("_")
    if len(prefix_parts) == 3 and prefix_parts[1] == "val":
        config["train_dataset"] = f"{prefix_parts[0]}_val"
        config["eval_task"] = prefix_parts[2]
    elif len(prefix_parts) == 2:
        config["train_dataset"] = prefix_parts[0]
        config["eval_task"] = prefix_parts[1]
    else:
        config["train_dataset"] = g["prefix"]

    config["model"] = g["model"]
    config["selection"] = g["curation"]
    config["training_type"] = g["finetuning"]
    config["percentage"] = g["percentage"] or ""
    config["max_steps"] = g["max_steps"] or ""
    config["learning_rate"] = g["learning_rate"]
    config["batch_size"] = g["batch_size"]
    config["n_val"] = g["n_val"]
    config["seed"] = g["seed"]
    return config


def find_models(models_dir: str, train_dataset: Optional[str] = None, method: Optional[str] = None) -> List[str]:
    """Find all model directories, optionally filtering by prefix pattern and method.

    Args:
        models_dir: Directory containing model directories
        train_dataset: Filter prefix (e.g., "alpaca_samsum")
        method: Method filter (e.g., "FullTraining-MeSO", "LayerWiseSubset-Full")
    """
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
            # Check train_dataset filter
            if train_dataset is not None:
                if not (entry.startswith(train_dataset + "-") or entry.startswith(train_dataset + "_")):
                    continue

            # Check method filter (e.g., "FullTraining-MeSO" matches "-FullTraining-MeSO-")
            if method is not None:
                # Method appears in directory name as -{method}-{finetuning}-
                # e.g., alpaca_samsum-Llama-3.2-1B-FullTraining-MeSO-p0.4-...
                method_pattern = f"-{method}-"
                if method_pattern not in entry:
                    continue

            model_paths.append(entry_path)
    return sorted(model_paths)


def evaluate_samsum(args, model, tokenizer) -> dict:
    """Run SamSUM evaluation."""
    from .tasks.samsum import compute_accuracy

    logger.info("Evaluating on SamSUM")
    scores = compute_accuracy(
        args=args,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    out = {"task": "samsum"}
    out.update(scores)
    return out


def evaluate_tydiqa(args, model, tokenizer) -> dict:
    """Run TyDiQA evaluation."""
    from .tasks.tydiqa import compute_accuracy

    logger.info("Evaluating on TyDiQA")
    results = compute_accuracy(args=args, model=model, tokenizer=tokenizer)
    out = {"task": "tydiqa"}
    out.update(results)
    return out


def evaluate_nq_open(args, model, tokenizer) -> dict:
    """Run NQ-open closed-book QA evaluation (EM/F1)."""
    from .tasks.nq_open import compute_accuracy
    logger.info("Evaluating on NQ-open")
    out = {"task": "nq_open"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def evaluate_squad(args, model, tokenizer) -> dict:
    """Run SQuAD closed-book (no context) evaluation (EM/F1)."""
    from .tasks.squad import compute_accuracy
    logger.info("Evaluating on SQuAD (closed-book)")
    out = {"task": "squad"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def evaluate_triviaqa(args, model, tokenizer) -> dict:
    """Run TriviaQA closed-book evaluation (EM/F1)."""
    from .tasks.triviaqa import compute_accuracy
    logger.info("Evaluating on TriviaQA (closed-book)")
    out = {"task": "triviaqa"}
    out.update(compute_accuracy(
        args=args, model=model, tokenizer=tokenizer,
        batch_size=args.batch_size, max_new_tokens=32,
    ))
    return out


def get_task_from_model_name(model_name: str) -> Optional[str]:
    """Extract the evaluation task from a run-dir name. Two layouts:

    Main runs:        ``<train>_<task>-<model>-...``      (e.g. ``alpaca_samsum-...``)
    Target-only runs: ``<task>_val_<task>-<model>-...``    (e.g. ``samsum_val_samsum-...``)
    """
    # nq_open is two underscores ("nq_open"), so match longest known task first.
    valid_tasks = ["nq_open", "samsum", "tydiqa", "squad", "triviaqa"]
    head = model_name.split("-", 1)[0]
    # Target-only: <task>_val_<task>
    for t in valid_tasks:
        if head == f"{t}_val_{t}":
            return t
    # Main: <train>_<task> — task is the suffix
    for t in valid_tasks:
        if head.endswith("_" + t):
            return t
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
    args.subject = subject  # legacy arg; unused in current scope

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

        elif task == "nq_open":
            logger.info(f"Evaluating {model_name} on NQ-open...")
            nq_results = evaluate_nq_open(args, model, tokenizer)
            results["nq_open_em"] = nq_results["em"]
            results["nq_open_f1"] = nq_results["f1"]
            with open(os.path.join(model_path, "nq_open_results.json"), "w") as f:
                json.dump(nq_results, f, indent=2)

        elif task == "squad":
            logger.info(f"Evaluating {model_name} on SQuAD (closed-book)...")
            sq_results = evaluate_squad(args, model, tokenizer)
            results["squad_em"] = sq_results["em"]
            results["squad_f1"] = sq_results["f1"]
            with open(os.path.join(model_path, "squad_results.json"), "w") as f:
                json.dump(sq_results, f, indent=2)

        elif task == "triviaqa":
            logger.info(f"Evaluating {model_name} on TriviaQA (closed-book)...")
            tq_results = evaluate_triviaqa(args, model, tokenizer)
            results["triviaqa_em"] = tq_results["em"]
            results["triviaqa_f1"] = tq_results["f1"]
            with open(os.path.join(model_path, "triviaqa_results.json"), "w") as f:
                json.dump(tq_results, f, indent=2)

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
        default=os.environ.get("SCRATCH_DIR", "/scratch") + "/Dr.Post-Training/SFT",
        help="Directory containing trained models (default)")
    model_group.add_argument("--model_path", type=str,
        help="Path to single model to evaluate")

    parser.add_argument("--train", type=str, default=None,
        help="Filter by training dataset (e.g., alpaca, less, tulu3, wizardlm)")
    parser.add_argument("--task", type=str, default=None,
        choices=["samsum", "tydiqa", "nq_open", "squad"],
        help="Override auto-detected task (optional)")
    parser.add_argument("--subject", type=str, default=None,
        help="(legacy; unused in current scope)")
    parser.add_argument("--method", type=str, default=None,
        help="Filter by method (e.g., FullTraining-MeSO, LayerWiseSubset-Full)")
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
    model_paths = find_models(args.models_dir, filter_prefix, args.method)
    logger.info(f"Found {len(model_paths)} models to evaluate")
    if args.method:
        logger.info(f"Filtering by method: {args.method}")

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

    # NQ-open results
    nq_results = [r for r in all_results if "nq_open_em" in r]
    if nq_results:
        print(f"\nNQ-open Results:")
        print(f"{'Model':<80} {'EM':>8} {'F1':>8}")
        print("-" * 100)
        for r in sorted(nq_results, key=lambda x: x.get("nq_open_em", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['nq_open_em']:>8.4f} {r.get('nq_open_f1', 0):>8.4f}")

    # SQuAD (closed-book) results
    squad_results = [r for r in all_results if "squad_em" in r]
    if squad_results:
        print(f"\nSQuAD (closed-book) Results:")
        print(f"{'Model':<80} {'EM':>8} {'F1':>8}")
        print("-" * 100)
        for r in sorted(squad_results, key=lambda x: x.get("squad_em", 0), reverse=True):
            print(f"{r['model_name'][:80]:<80} "
                  f"{r['squad_em']:>8.4f} {r.get('squad_f1', 0):>8.4f}")

    errors = [r for r in all_results if r.get("error")]
    if errors:
        print(f"\nErrors: {len(errors)} models failed")


if __name__ == "__main__":
    main()
