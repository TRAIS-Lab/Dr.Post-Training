#!/usr/bin/env python
"""
Training script for RLHF with layer-wise selection.

This script implements PPO training with the layer-wise selection streaming
paradigm, enabling efficient data selection during policy optimization.

Usage:
    # Standard PPO training (no selection)
    python train_rlhf.py --model_name EleutherAI/gpt-neo-125m --no_selection

    # PPO with layer-wise selection (GREATS)
    python train_rlhf.py --model_name EleutherAI/gpt-neo-125m \
        --selection_method GREATS --selection_frac 0.5

    # With accelerate for multi-GPU
    accelerate launch train_rlhf.py --model_name EleutherAI/gpt-neo-2.7B
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer
from accelerate import Accelerator

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compress_gradient import GradientHook, setup_model_compressors, create_sample_inputs
from experiment_rlhf.ppo_trainer import PPOTrainerWithSelection
from experiment_rlhf.data.get_prompts import build_toxicity_promptdata, collator
from experiment_rlhf.rewards import (
    load_toxicity_model,
    compute_toxicity_scores,
    format_for_reward_model,
    CrossVocabRewardWrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """Arguments for the RLHF training script with layer-wise selection."""

    # Model arguments
    model_name: str = field(
        default="EleutherAI/gpt-neo-125m",
        metadata={"help": "Model name or path"}
    )
    reward_model_name: str = field(
        default="facebook/roberta-hate-speech-dynabench-r4-target",
        metadata={"help": "Reward model name (toxicity classifier)"}
    )

    # Dataset arguments
    dataset_name: str = field(
        default="toxicity",
        metadata={"help": "Dataset name: toxicity, imdb"}
    )
    val_size: int = field(
        default=1024,
        metadata={"help": "Total validation pool size (n_val equivalent from SFT)"}
    )
    val_batch_size: int = field(
        default=None,
        metadata={
            "help": (
                "Batch size for validation data used during selection. "
                "If None, defaults to val_size (use all validation data). "
                "This allows independent control of how much validation data is used per selection step."
            )
        }
    )
    val_strategy: str = field(
        default="random",
        metadata={"help": "Validation strategy: random or top (most toxic)"}
    )

    # Selection arguments
    no_selection: bool = field(
        default=False,
        metadata={"help": "Disable layer-wise selection (standard PPO)"}
    )
    selection_method: str = field(
        default="GREATS",
        metadata={"help": "Selection method: GREATS or GradNorm"}
    )
    selection_frac: float = field(
        default=0.5,
        metadata={"help": "Fraction of samples to select per layer (0-1)"}
    )
    selection_mode: str = field(
        default="per_layer",
        metadata={
            "help": (
                "Data selection granularity. "
                "Options: 'global' or 'per_layer' (default). "
                "global: Accumulates scores across all layers, selects globally, then does second pass for gradients. "
                "per_layer: Each layer independently selects samples (single pass, uses block-diagonal approximation)."
            )
        }
    )
    val_loss_type: str = field(
        default="logprob",
        metadata={"help": "Validation loss type: logprob, reward_weighted, advantage_weighted"}
    )
    use_second_order: bool = field(
        default=False,
        metadata={"help": "Use second-order selection (slower but more accurate)"}
    )

    # Compression arguments
    use_compression: bool = field(
        default=True,
        metadata={"help": "Use gradient compression"}
    )
    sparsifier_dim: int = field(
        default=32,
        metadata={"help": "Sparsifier projection dimension (GraSS)"}
    )
    projector_dim: int = field(
        default=512,
        metadata={"help": "Projector projection dimension (LoGra)"}
    )
    sparsifier_type: str = field(
        default="random_mask",
        metadata={"help": "Sparsifier type: random_mask, identity"}
    )
    projector_type: str = field(
        default="normal",
        metadata={"help": "Projector type: normal, identity"}
    )
    update_freq: int = field(
        default=200,
        metadata={"help": "Frequency to update compressors"}
    )

    # PPO training arguments
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    per_device_train_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for PPO training (per device)"}
    )
    mini_batch_size: int = field(
        default=1,
        metadata={"help": "Mini-batch size for PPO updates"}
    )
    epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs per PPO step (consistent with SFT naming)"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps"}
    )
    steps: int = field(
        default=200,
        metadata={"help": "Total training steps"}
    )
    init_kl_coef: float = field(
        default=0.04,
        metadata={"help": "Initial KL penalty coefficient"}
    )
    target_kl: float = field(
        default=0.1,
        metadata={"help": "Target KL divergence"}
    )

    # Generation arguments
    max_length: int = field(
        default=30,
        metadata={"help": "Maximum generation length"}
    )
    min_length: int = field(
        default=20,
        metadata={"help": "Minimum generation length"}
    )
    gen_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for generation"}
    )

    # LoRA arguments
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for training"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )

    # Other arguments
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    output_dir: str = field(
        default="outputs/rlhf_selection",
        metadata={"help": "Output directory"}
    )
    save_freq: Optional[int] = field(
        default=None,
        metadata={"help": "Save model every N steps"}
    )
    log_interval: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Log training metrics every N steps. "
                "If None, defaults to steps // 10 (producing ~10 log entries)."
            )
        }
    )
    run_name: str = field(
        default="rlhf-selection",
        metadata={"help": "Run name for logging"}
    )


def find_trainable_layers(model, lora_only: bool = True) -> List[str]:
    """
    Find trainable layers in the model for gradient hooks.

    Args:
        model: The model to search
        lora_only: If True, only find LoRA layers

    Returns:
        List of layer names
    """
    layer_names = []

    for name, module in model.named_modules():
        if lora_only:
            # Find LoRA adapter layers
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Handle different LoRA implementations
                if hasattr(module.lora_A, 'default'):
                    layer_names.append(f"{name}.lora_A.default")
                elif isinstance(module.lora_A, torch.nn.Linear):
                    layer_names.append(f"{name}.lora_A")

                if hasattr(module.lora_B, 'default'):
                    layer_names.append(f"{name}.lora_B.default")
                elif isinstance(module.lora_B, torch.nn.Linear):
                    layer_names.append(f"{name}.lora_B")
        else:
            # Find all Linear layers (excluding special heads)
            if isinstance(module, torch.nn.Linear):
                if 'v_head' not in name and 'lm_head' not in name:
                    layer_names.append(name)

    return layer_names


def setup_gradient_hooks(model, args, device: str, tokenizer=None) -> Optional[GradientHook]:
    """
    Set up gradient hooks and compressors for layer-wise selection.

    Args:
        model: The model
        args: Script arguments
        device: Device string
        tokenizer: Tokenizer for creating sample inputs (needed for compression)

    Returns:
        GradientHook instance or None if selection disabled
    """
    if args.no_selection:
        logger.info("Layer-wise selection disabled")
        return None

    # Find trainable layers
    layer_names = find_trainable_layers(model, lora_only=args.use_lora)
    logger.info(f"Found {len(layer_names)} trainable layers for hooks")

    if len(layer_names) == 0:
        logger.warning("No trainable layers found! Disabling selection.")
        return None

    # Create gradient hook
    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=device,
    )

    # Setup compressors if enabled
    if args.use_compression and tokenizer is not None:
        # Create sample inputs for compressor initialization
        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer,
            max_seq_length=args.max_length,
            device=device,
        )

        # Configure sparsifier kwargs (GraSS-style)
        sparsifier_kwargs = {
            "proj_dim": args.sparsifier_dim,
            "proj_max_batch_size": args.per_device_train_batch_size,
            "proj_seed": args.seed,
            "device": device,
            "proj_type": args.sparsifier_type,
        }

        # Configure projector kwargs (LoGra-style)
        projector_kwargs = {
            "proj_dim": args.projector_dim,
            "proj_max_batch_size": args.per_device_train_batch_size,
            "proj_seed": args.seed,
            "device": device,
            "proj_type": args.projector_type,
        }

        compressors = setup_model_compressors(
            model=model,
            layer_names=layer_names,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            sample_inputs=sample_inputs,
            device=device,
            update_freq=args.update_freq,
        )
        grad_hook.set_compressors(compressors)
        logger.info(f"Compressors configured: sparsifier_dim={args.sparsifier_dim}, projector_dim={args.projector_dim}")
    elif args.use_compression:
        logger.warning("Compression requested but no tokenizer provided. Skipping compression setup.")

    return grad_hook


def train_loop(
    args: ScriptArguments,
    ppo_trainer: PPOTrainer,
    selection_trainer: PPOTrainerWithSelection,
    reward_tokenizer,
    reward_model,
    val_queries: List[str],
    val_input_ids: List[torch.Tensor],
    train_dataset,
    val_dataset,
):
    """
    Main training loop for PPO with optional layer-wise selection.

    Always uses the custom PPOTrainerWithSelection for consistent logging.
    When selection is disabled (NA), it performs standard PPO updates.
    When selection is enabled, it performs layer-wise gradient selection.

    Args:
        args: Script arguments
        ppo_trainer: TRL PPOTrainer instance
        selection_trainer: PPOTrainerWithSelection (handles both baseline and selection)
        reward_tokenizer: Tokenizer for reward model
        reward_model: Reward model
        val_queries: Validation query strings
        val_input_ids: Validation input IDs
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    from torch.utils.data import DataLoader

    # Create train dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Create validation dataloader (only needed if selection is enabled)
    val_dataloader = None
    if not args.no_selection:
        effective_val_batch_size = args.val_batch_size if args.val_batch_size is not None else args.val_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=effective_val_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    # Calculate number of epochs based on steps
    num_batches = len(train_dataloader)
    num_epochs = max(1, (args.steps + num_batches - 1) // num_batches)

    logger.info(f"Training for up to {num_epochs} epochs ({args.steps} target steps)")

    # Determine log interval
    if args.log_interval is not None:
        log_interval = args.log_interval
    else:
        log_interval = max(1, args.steps // 10)  # Default: ~10 log entries

    # Use selection trainer's training loop (handles both baseline and selection modes)
    selection_trainer.train_with_selection(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        log_interval=log_interval,
        max_steps=args.steps,
    )

    # Save final model
    final_path = os.path.join(args.output_dir, "final")
    ppo_trainer.save_model(final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log configuration
    logger.info(f"Run name: {args.run_name}")

    # Device setup
    accelerator = Accelerator()
    current_device = accelerator.local_process_index
    device = f"cuda:{current_device}" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("RLHF Training with Layer-wise Selection")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Selection: {'Disabled' if args.no_selection else f'{args.selection_method} @ {args.selection_frac}'}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    train_dataset, val_dataset = build_toxicity_promptdata(
        tokenizer,
        num_samples=args.val_size,
        seed=args.seed,
        val_strategy=args.val_strategy,
    )
    val_queries = val_dataset["query"]
    val_input_ids = val_dataset["input_ids"]

    # Create PPO config (TRL 0.26.1 API)
    # Calculate total_episodes based on steps and per_device_train_batch_size
    total_episodes = args.steps * args.per_device_train_batch_size
    ppo_config = PPOConfig(
        output_dir=args.output_dir,  # Required by TrainingArguments
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,  # TRL 0.26.1 uses this
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.epochs,  # Renamed from ppo_epochs for consistency with SFT
        seed=args.seed,
        kl_coef=args.init_kl_coef,
        remove_unused_columns=False,
        report_to=None,  # Disable logging
        # PPO specific params
        total_episodes=total_episodes,
        response_length=args.max_length - args.min_length,  # Max new tokens
        local_rollout_forward_batch_size=args.gen_batch_size,
        num_mini_batches=max(1, args.per_device_train_batch_size // args.mini_batch_size),
        # Disable gradient checkpointing for now (can cause issues with PEFT)
        gradient_checkpointing=False,
        # Disable evaluation (we don't have eval_dataset)
        eval_strategy="no",
        num_sample_generations=0,  # Disable sample generation during training
    )

    # Load reward model first (needed by PPOTrainer)
    logger.info(f"Loading reward model: {args.reward_model_name}")
    reward_tokenizer, reward_model_raw = load_toxicity_model(
        args.reward_model_name,
        device=device,
        wrap_for_trl=False,  # Don't wrap yet, we'll use CrossVocabRewardWrapper
    )

    # Wrap reward model with CrossVocabRewardWrapper to handle vocabulary mismatch
    # TRL passes policy model (GPT-Neo) token IDs to reward model (RoBERTa)
    # This wrapper decodes tokens to text and re-tokenizes for the reward model
    reward_model = CrossVocabRewardWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
    )
    logger.info("Wrapped reward model with CrossVocabRewardWrapper for vocabulary mismatch handling")

    # Load policy model (TRL 0.26.1 uses regular AutoModelForCausalLM)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": current_device},
    )

    # Load value model (TRL 0.26.1 expects a model with .score() method)
    # AutoModelForSequenceClassification has a score Linear layer
    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,  # Single value output
        torch_dtype=torch.bfloat16,
        device_map={"": current_device},
    )
    logger.info("Loaded value model for PPO value estimates")

    # Apply LoRA if requested
    if args.use_lora:
        from peft import get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with r={args.lora_r}, alpha={args.lora_alpha}")

    # Create PPO trainer (TRL 0.26.1 API)
    # Note: 'config' is passed as 'args', 'tokenizer' as 'processing_class'
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        value_model=value_model,  # Sequence classification model with .score()
        reward_model=reward_model,  # Toxicity classifier
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # Setup gradient hooks and selection trainer
    # Always use custom trainer for consistent logging
    grad_hook = None
    if not args.no_selection:
        grad_hook = setup_gradient_hooks(model, args, device, tokenizer=tokenizer)

    # Determine selection method (NA if no_selection, else use args)
    selection_method = "NA" if args.no_selection else args.selection_method

    # Always create selection trainer (handles both baseline and selection modes)
    selection_trainer = PPOTrainerWithSelection(
        ppo_trainer=ppo_trainer,
        grad_hook=grad_hook,  # None for baseline/NA
        selection_frac=args.selection_frac,
        selection_method=selection_method,
        selection_mode=args.selection_mode,
        val_loss_type=args.val_loss_type,
        use_second_order=args.use_second_order,
    )
    if grad_hook is not None:
        logger.info(f"Selection trainer initialized with {len(grad_hook.layer_names)} layers, mode={args.selection_mode}")
    else:
        logger.info("Selection trainer initialized in baseline mode (no selection)")

    # Run training loop
    logger.info("Starting training...")
    train_loop(
        args=args,
        ppo_trainer=ppo_trainer,
        selection_trainer=selection_trainer,
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        val_queries=val_queries,
        val_input_ids=val_input_ids,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
