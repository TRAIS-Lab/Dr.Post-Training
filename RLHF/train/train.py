#!/usr/bin/env python
# coding=utf-8
"""
Training script for RLHF with gradient streaming.

This script implements PPO training with optional layer-wise data selection,
following the same design patterns as SFT/train/train.py.

Usage:
    # Baseline (no selection)
    python RLHF/train/train.py --method NA --output_dir outputs/baseline

    # Streaming (per-layer selection)
    python RLHF/train/train.py --method Streaming --selection_frac 0.5

    # GREATS (global selection)
    python RLHF/train/train.py --method GREATS --selection_frac 0.5

    # With compression (implies MeSO optimizer)
    python RLHF/train/train.py --method Streaming --sparsification Rademacher-64*64
"""

import logging
import os
import sys
import warnings

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed, get_scheduler
from trl import AutoModelForCausalLMWithValueHead

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

from RLHF.data.get_prompts import get_prompt_dataset, collator
from RLHF.data.get_validation import get_validation_dataset
from RLHF.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from RLHF.train.training_arguments import TrainingArguments
from RLHF.train.trainer import StreamingPPOTrainer
from RLHF.train.rewards import load_reward_model, RewardModelWrapper

from gradstream import GradientHook, setup_model_compressors, create_sample_inputs, MeSOAdamW

logger = logging.getLogger(__name__)


def find_trainable_layers(model, lora_only: bool = True):
    """
    Find trainable layers for gradient hooks.

    Args:
        model: The model (can be AutoModelForCausalLMWithValueHead or base model)
        lora_only: If True, only find LoRA layers

    Returns:
        List of layer names (with correct prefix for wrapper models)
    """
    layer_names = []

    # Handle AutoModelForCausalLMWithValueHead wrapper
    # The pretrained_model attribute contains the actual model
    target_model = model
    prefix = ""
    if hasattr(model, "pretrained_model"):
        target_model = model.pretrained_model
        prefix = "pretrained_model."

    for name, module in target_model.named_modules():
        if lora_only:
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if hasattr(module.lora_A, "default"):
                    layer_names.append(f"{prefix}{name}.lora_A.default")
                elif isinstance(module.lora_A, torch.nn.Linear):
                    layer_names.append(f"{prefix}{name}.lora_A")

                if hasattr(module.lora_B, "default"):
                    layer_names.append(f"{prefix}{name}.lora_B.default")
                elif isinstance(module.lora_B, torch.nn.Linear):
                    layer_names.append(f"{prefix}{name}.lora_B")
        else:
            if isinstance(module, torch.nn.Linear):
                # Skip special heads
                if "lm_head" not in name and "v_head" not in name:
                    layer_names.append(f"{prefix}{name}")

    return layer_names


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("RLHF Training with Gradient Streaming")
    logger.info(f"  Task: {training_args.task}")
    logger.info(f"  Method: {training_args.method}")
    logger.info(f"  Model: {model_args.model_name_or_path}")
    logger.info(f"  LoRA: {model_args.lora}")
    if training_args.has_selection:
        logger.info(f"  Selection fraction: {training_args.selection_frac}")
    if training_args.has_compression:
        logger.info(f"  Sparsification: {training_args.sparsification}")
        logger.info(f"  Projection: {training_args.projection}")
    logger.info("=" * 60)

    # Set seed
    set_seed(training_args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_padding_to_tokenizer(tokenizer)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = get_prompt_dataset(
        task=training_args.task,
        tokenizer=tokenizer,
        seed=training_args.seed,
    )
    logger.info(f"  Train dataset: {len(train_dataset)} samples")

    val_dataset = None
    if training_args.has_selection:
        val_dataset = get_validation_dataset(
            task=training_args.task,
            tokenizer=tokenizer,
            n_val=training_args.n_val,
            seed=training_args.seed,
            strategy=training_args.val_strategy,
        )
        logger.info(f"  Validation dataset: {len(val_dataset)} samples")

    # Load policy model with value head for PPO
    logger.info("Loading policy model with value head...")
    model_kwargs = {"torch_dtype": getattr(torch, model_args.torch_dtype)}
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Setup LoRA config if enabled
    peft_config = None
    if model_args.lora:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules if model_args.lora_target_modules else None,
        )

    # Load model with value head (and LoRA if configured)
    # AutoModelForCausalLMWithValueHead adds a v_head for value estimation
    # NOTE: Set summary_dropout_prob=0.0 to disable value head dropout
    # This ensures consistent behavior between train/eval modes for forward passes
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        peft_config=peft_config,
        summary_dropout_prob=0.1,
        **model_kwargs,
    ).to(device)

    if model_args.lora:
        logger.info(f"Applied LoRA with r={model_args.lora_r}")
        model.pretrained_model.print_trainable_parameters()

        # Enable input gradients for checkpointing
        if hasattr(model.pretrained_model, "enable_input_require_grads"):
            model.pretrained_model.enable_input_require_grads()

    # Log value head info
    logger.info(f"Value head: {model.v_head}")

    # Verify and log trainable parameters
    n_trainable = 0
    n_frozen = 0
    n_policy_trainable = 0
    n_v_head_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_trainable += param.numel()
            if "v_head" in name:
                n_v_head_trainable += param.numel()
            else:
                n_policy_trainable += param.numel()
        else:
            n_frozen += param.numel()
    logger.info(f"Trainable parameters: {n_trainable:,} (policy: {n_policy_trainable:,}, v_head: {n_v_head_trainable:,})")
    logger.info(f"Frozen parameters: {n_frozen:,}")
    if n_policy_trainable == 0:
        logger.warning("WARNING: No policy parameters are trainable! This will break PPO training.")

    # Reference model handling:
    # - For PEFT (LoRA) models: No separate reference model needed
    #   The trainer uses disable_adapter() on the policy model (matching reference implementation)
    # - For non-PEFT models: Load a frozen copy for KL penalty
    ref_model = None
    if not model_args.lora:
        logger.info("Loading reference model (frozen) for non-PEFT training...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        ).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        logger.info("Using disable_adapter() for reference logprobs (PEFT model)")

    # Load reward model
    logger.info("Loading reward model...")
    reward_tokenizer, reward_model_raw = load_reward_model(
        model_args.reward_model_name,
        device=device,
    )

    # Wrap reward model to handle vocabulary mismatch
    reward_model = RewardModelWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
    )

    # Find trainable layers
    layer_names = find_trainable_layers(model, lora_only=model_args.lora)
    logger.info(f"Found {len(layer_names)} trainable layers")

    # Setup gradient hooks if selection is enabled
    grad_hook = None
    if training_args.has_selection or training_args.has_compression:
        logger.info("Setting up gradient hooks...")

        grad_hook = GradientHook(
            model=model,
            layer_names=layer_names,
            device=device,
            register_hooks=True,
        )

        # Setup compression if enabled
        if training_args.has_compression:
            logger.info("Setting up gradient compression...")

            # Parse sparsification
            if training_args.sparsification is not None:
                method, dim = training_args.sparsification.split("-")
                assert "*" in dim, "Sparsification dimension must be factorized (e.g., '64*64')"
                dim_parts = dim.split("*")
                sparsification_dim = int(dim_parts[0])

                sparsifier_kwargs = {
                    "proj_dim": sparsification_dim,
                    "proj_max_batch_size": training_args.per_device_train_batch_size,
                    "proj_seed": training_args.seed,
                    "device": device,
                    "proj_type": method,
                }
            else:
                sparsifier_kwargs = {
                    "proj_dim": -1,
                    "proj_type": "identity",
                    "device": device,
                }

            # Parse projection
            if training_args.projection is not None:
                method, dim = training_args.projection.split("-")
                proj_dim = int(dim)

                projector_kwargs = {
                    "proj_dim": proj_dim,
                    "proj_max_batch_size": training_args.per_device_train_batch_size,
                    "proj_seed": training_args.seed,
                    "device": device,
                    "proj_type": method,
                }
            else:
                projector_kwargs = {
                    "proj_dim": -1,
                    "proj_type": "identity",
                    "device": device,
                }

            # Create sample inputs for compressor initialization
            sample_inputs = create_sample_inputs(
                tokenizer=tokenizer,
                max_seq_length=training_args.max_new_tokens + 20,
                device=device,
            )

            # Setup compressors
            compressors = setup_model_compressors(
                model=model,
                layer_names=layer_names,
                sparsifier_kwargs=sparsifier_kwargs,
                projector_kwargs=projector_kwargs,
                sample_inputs=sample_inputs,
                device=device,
                update_freq=training_args.update_compressor_freq,
            )

            grad_hook.set_compressors(compressors)
            logger.info(f"Compression setup complete: {len(compressors)} layers")

    # Create optimizer - only pass trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizer will optimize {len(trainable_params)} parameter groups")

    if training_args.has_compression:
        # Use MeSO optimizer
        optimizer = MeSOAdamW(
            trainable_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        logger.info("Using MeSO optimizer")
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        logger.info("Using AdamW optimizer")

    # Create dataloaders (needed for step calculation)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_batch_size = training_args.val_batch_size or training_args.n_val
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    # Calculate training steps for lr_scheduler
    if training_args.max_steps > 0:
        num_training_steps = training_args.max_steps
    else:
        num_training_steps = len(train_dataloader) * int(training_args.num_train_epochs)

    # Calculate warmup steps
    if training_args.warmup_steps > 0:
        num_warmup_steps = training_args.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)

    # Create lr_scheduler
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        f"LR scheduler: {training_args.lr_scheduler_type}, "
        f"warmup_steps={num_warmup_steps}, total_steps={num_training_steps}"
    )

    # Create trainer
    # Note: For PEFT models, ref_model=None and trainer uses disable_adapter()
    trainer = StreamingPPOTrainer(
        model=model,
        ref_model=ref_model,  # None for PEFT, frozen model for non-PEFT
        reward_model=reward_model,
        tokenizer=tokenizer,
        args=training_args,
        grad_hook=grad_hook,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Calculate training steps
    num_epochs = training_args.num_train_epochs
    max_steps = training_args.max_steps if training_args.max_steps > 0 else None

    # Use logging_steps from training args
    log_interval = int(training_args.logging_steps)

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=int(num_epochs),
        max_steps=max_steps,
        log_interval=log_interval,
    )

    # Save model
    final_output_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_output_dir)

    # Clean up hooks
    if grad_hook is not None and grad_hook.hooks_registered:
        grad_hook.remove_hooks()
        logger.info("Removed gradient hooks")

    logger.info("Training complete!")
    return history


if __name__ == "__main__":
    main()
