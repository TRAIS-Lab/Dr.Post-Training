#!/usr/bin/env python
# coding=utf-8
"""
Training script that uses hook-based gradient computation.
"""

import logging
import os
import sys
import time
import warnings

import datasets
import torch
import torch.distributed as dist
import transformers

# Suppress torch.compile warnings about custom CUDA kernels (SJLT)
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, set_seed)

from experiment.data.get_train_dataset import get_training_dataset
from experiment.data.get_val_dataset import get_dataset

from compress_gradient import (
    GradientHook,
    CompGradTrainer,
    setup_model_compressors,
    create_sample_inputs
)

from experiment.train.data_arguments import DataArguments, get_data_statistics
from experiment.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from experiment.train.training_arguments import TrainingArguments


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_trainable_layers(model, lora_only=True):
    """
    Find trainable layers in the model.

    Args:
        model: The model to search
        lora_only: If True, only find LoRA layers (lora_A and lora_B). If False, find all Linear layers.

    Returns:
        List of layer names
        - For LoRA: returns paths to lora_A and lora_B modules (e.g., "layer.lora_A.default", "layer.lora_B.default")
        - For full fine-tuning: returns paths to all Linear layers
    """
    layer_names = []

    for name, module in model.named_modules():
        if lora_only:
            # Find LoRA adapter layers - these are the actual trainable parameters
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # This is a PEFT LoRA wrapper
                # lora_A and lora_B can be either ModuleDict or direct nn.Linear

                # Handle lora_A
                if hasattr(module.lora_A, 'default'):
                    # ModuleDict case: lora_A['default']
                    lora_a_name = f"{name}.lora_A.default"
                    layer_names.append(lora_a_name)
                elif isinstance(module.lora_A, torch.nn.Linear):
                    # Direct nn.Linear case
                    lora_a_name = f"{name}.lora_A"
                    layer_names.append(lora_a_name)

                # Handle lora_B
                if hasattr(module.lora_B, 'default'):
                    # ModuleDict case: lora_B['default']
                    lora_b_name = f"{name}.lora_B.default"
                    layer_names.append(lora_b_name)
                elif isinstance(module.lora_B, torch.nn.Linear):
                    # Direct nn.Linear case
                    lora_b_name = f"{name}.lora_B"
                    layer_names.append(lora_b_name)
        else:
            # Find all Linear layers (for full fine-tuning)
            if isinstance(module, torch.nn.Linear):
                layer_names.append(name)

    return layer_names


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load training dataset
    train_dataset = get_training_dataset(
        data_dir=data_args.data_dir,
        task=training_args.analysis_dataset,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        seed=data_args.sample_data_seed,
        train_files=data_args.train_files if data_args.train_files else None,
        train_dataset_names=training_args.train_dataset_names
    )

    get_data_statistics(train_dataset)

    # Load model - NO CUSTOM LAYER REPLACEMENT!
    # Auto-derive model dtype from training precision if not explicitly set
    # This ensures model dtype matches training precision (critical for flash attention)
    if model_args.torch_dtype is None and (training_args.bf16 or training_args.fp16):
        model_args.torch_dtype = "bfloat16" if training_args.bf16 else "float16"
        logger.info(f"Auto-setting model dtype to {model_args.torch_dtype} to match training precision")

    model_kwargs = {"torch_dtype": model_args.torch_dtype}
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 enabled")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs)
    add_padding_to_tokenizer(tokenizer)

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    # Apply LoRA using standard PEFT (no custom layers!)
    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA to model using PEFT.")
        model.print_trainable_parameters()

        # For checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Find trainable layers (LoRA layers or all Linear layers)
    layer_names = find_trainable_layers(model, lora_only=model_args.lora)
    logger.info(f"Found {len(layer_names)} trainable layers for hook attachment")

    # Log layer names for verification
    if len(layer_names) > 0:
        logger.info("=== Layer Names for Hook Attachment ===")
        for i, name in enumerate(layer_names[:5]):  # Show first 5
            logger.info(f"  [{i}] {name}")
        if len(layer_names) > 5:
            logger.info(f"  ... and {len(layer_names) - 5} more layers")
    else:
        logger.warning("WARNING: No trainable layers found! Check model and lora_only setting.")

    # Determine if hooks should be registered based on training method
    # Register hooks for methods that require gradient computation (GREATS, GradNorm)
    # OR when using MeSO (use_compressed_optimizer) with gradient compression
    should_register_hooks = (
        training_args.method in ['GREATS', 'GradNorm'] or
        (training_args.use_compressed_optimizer and (training_args.sparsification is not None or training_args.projection is not None))
    )
    logger.info(f"Training method: {training_args.method} - Hooks will be {'registered' if should_register_hooks else 'NOT registered'}")
    if training_args.use_compressed_optimizer and should_register_hooks:
        logger.info("  Hooks enabled for MeSO (use_compressed_optimizer=True) with gradient compression")

    # Create gradient hook
    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(training_args.device),
        register_hooks=should_register_hooks
    )
    logger.info("Gradient Hook created successfully.")

    # Optional: Set up gradient compression
    if training_args.sparsification is not None or training_args.projection is not None:
        logger.info("=== Gradient Compression Setup ===")

        # Parse sparsification argument
        if training_args.sparsification is None:
            sparsifier_kwargs = {
                "proj_dim": -1,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "device": str(training_args.device),
                "proj_type": "identity",
            }
            logger.info("  Sparsification: Disabled")
        else:
            sparsification_method, sparsification_dim = training_args.sparsification.split("-")
            assert "*" in sparsification_dim, "Sparsification dimension must be factorized (e.g., '64*64')."

            sparsification_dim_parts = sparsification_dim.split("*")
            assert sparsification_dim_parts[0] == sparsification_dim_parts[1], \
                "Sparsification dimension must be the same for factorized projection."
            sparsification_dim = int(sparsification_dim_parts[0])

            sparsifier_kwargs = {
                "proj_dim": sparsification_dim,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "device": str(training_args.device),
                "proj_type": sparsification_method,
            }
            logger.info(f"  Sparsification: {sparsification_method} -> {sparsification_dim}*{sparsification_dim} dimension ")

        # Parse projection argument
        if training_args.projection is None:
            projector_kwargs = {
                "proj_dim": -1,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "device": str(training_args.device),
                "proj_type": "identity",
            }
            logger.info("  Projection: Disabled")
        else:
            proj_method, proj_dim = training_args.projection.split("-")
            assert "*" not in proj_dim, "Projection dimension must not be factorized."

            proj_dim = int(proj_dim)

            projector_kwargs = {
                "proj_dim": proj_dim,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "device": str(training_args.device),
                "proj_type": proj_method,
            }
            logger.info(f"  Projection: {proj_method} -> {proj_dim} dimension.")

        # Create sample inputs for compression initialization
        # This runs a forward pass to determine the dimensions needed for each layer's projector
        logger.info("Creating sample inputs for compressor initialization...")
        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            device=str(training_args.device)
        )

        # Set up compressors using sample inputs
        logger.info("Setting up model compressors...")
        compressors = setup_model_compressors(
            model=model,
            layer_names=layer_names,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            sample_inputs=sample_inputs,
            device=str(training_args.device),
            update_freq=training_args.update_compressor_freq
        )

        grad_hook.set_compressors(compressors)
        logger.info(f"  Set {len(compressors)} unified compressors (refreshed every {training_args.update_compressor_freq} steps)")

        logger.info("Gradient compression setup completed!")
    else:
        logger.info("Gradient compression disabled (sparsification and projection both None)")

    # Prepare validation dataset (used for data selection in GREATS)
    val_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        validation=True,
        k=training_args.n_val,
        subject=training_args.subject
    )

    # Prepare evaluation dataset (held-out test set for measuring generalization)
    eval_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        validation=False,
        k=training_args.n_eval,
        subject=training_args.subject
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Initialize trainer with compressed gradient capabilities
    trainer = CompGradTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        grad_hook=grad_hook,
    )

    # Initial evaluation before training (step 0)
    logger.info("*** Running initial evaluation before training ***")
    trainer.evaluate()

    # Train
    logger.info("*** Starting training with hook-based gradient computation ***")
    train_result = trainer.train()

    # Save final evaluation results
    trainer.on_train_end()

    # Save model
    trainer.save_model()

    # Clean up hooks (only if they were registered)
    if grad_hook.hooks_registered:
        grad_hook.remove_hooks()
        logger.info("Removed all hooks after training")
    else:
        logger.info("No hooks to remove (hooks were not registered)")

    return train_result


if __name__ == "__main__":
    main()
