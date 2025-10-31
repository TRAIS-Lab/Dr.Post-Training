#!/usr/bin/env python
# coding=utf-8
"""
Training script that uses hook-based gradient computation.
"""

import logging
import os
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, set_seed)

from ..data_selection.get_training_dataset import get_training_dataset
from ..data_selection.get_validation_dataset import get_dataset
from ..GradComp.core.hook import HookManager

from .hook_trainer import HookTrainer
from .data_arguments import DataArguments, get_data_statistics
from .model_arguments import ModelArguments, add_padding_to_tokenizer
from .training_arguments import TrainingArguments


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_trainable_layers(model, lora_only=True):
    """
    Find trainable layers in the model.

    Args:
        model: The model to search
        lora_only: If True, only find LoRA layers. If False, find all Linear layers.

    Returns:
        List of layer names (for LoRA, returns base_layer paths)
    """
    layer_names = []

    for name, module in model.named_modules():
        if lora_only:
            # Find LoRA layers - PEFT wraps layers and the actual computation happens in base_layer
            if hasattr(module, 'base_layer') and isinstance(module.base_layer, torch.nn.Linear):
                # This is a PEFT LoRA wrapper, attach hooks to the base_layer
                base_layer_name = f"{name}.base_layer"
                layer_names.append(base_layer_name)
                logger.debug(f"Found LoRA layer: {name} -> will attach hooks to {base_layer_name}")
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
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed)
    print('Training Set')
    get_data_statistics(train_dataset)

    # Load model - NO CUSTOM LAYER REPLACEMENT!
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
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
    # Only register hooks for methods that require gradient computation (GREATS, GradNorm)
    should_register_hooks = training_args.method in ['GREATS', 'GradNorm']
    logger.info(f"Training method: {training_args.method} - Hooks will be {'registered' if should_register_hooks else 'NOT registered'}")

    # Create hook manager using GradComp.core.hook.HookManager
    hook_manager = HookManager(
        model=model,
        layer_names=layer_names,
        profile=False,
        device=str(training_args.device),
        register_hooks=should_register_hooks
    )
    logger.info("Hook manager created successfully (using GradComp.core.hook.HookManager)")

    # Optional: Set up gradient compression
    if training_args.sparsification is not None or training_args.projection is not None:
        from ..GradComp.projection.projector import setup_model_compressors

        logger.info("=== Gradient Compression Setup ===")

        # Parse sparsification argument
        if training_args.sparsification is None:
            sparsifier_kwargs = None
            logger.info("  Sparsification: Disabled")
        else:
            sparsification_method, sparsification_dim = training_args.sparsification.split("-")
            if "*" in sparsification_dim:
                sparsification_factorize = True
                sparsification_dim_parts = sparsification_dim.split("*")
                assert sparsification_dim_parts[0] == sparsification_dim_parts[1], \
                    "Sparsification dimension must be the same for factorized projection."
                sparsification_dim = int(sparsification_dim_parts[0])
            else:
                sparsification_factorize = False
                sparsification_dim = int(sparsification_dim)

            sparsifier_kwargs = {
                "proj_dim": sparsification_dim,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "proj_factorize": sparsification_factorize,
                "device": str(training_args.device),
                "method": sparsification_method,
                "use_half_precision": True,
            }
            logger.info(f"  Sparsification: {sparsification_method} -> {sparsification_dim} dimension "
                       f"(factorized: {sparsification_factorize})")

        # Parse projection argument
        if training_args.projection is None:
            projector_kwargs = {
                "proj_dim": -1,
                "proj_max_batch_size": -1,
                "proj_seed": training_args.seed,
                "proj_factorize": False,
                "device": str(training_args.device),
                "method": "Identity",
                "use_half_precision": True,
            }
            logger.info("  Projection: Identity (no projection)")
        else:
            proj_method, proj_dim = training_args.projection.split("-")
            if "*" in proj_dim:
                proj_factorize = True
                proj_dim_parts = proj_dim.split("*")
                assert proj_dim_parts[0] == proj_dim_parts[1], \
                    "Projection dimension must be the same for factorized projection."
                proj_dim = int(proj_dim_parts[0])
            else:
                proj_factorize = False
                proj_dim = int(proj_dim)

            projector_kwargs = {
                "proj_dim": proj_dim,
                "proj_max_batch_size": 64,
                "proj_seed": training_args.seed,
                "proj_factorize": proj_factorize,
                "device": str(training_args.device),
                "method": proj_method,
                "use_half_precision": True,
            }
            logger.info(f"  Projection: {proj_method} -> {proj_dim} dimension (factorized: {proj_factorize})")

        # Create a small dummy batch for compression initialization
        dummy_text = ["This is a test sentence for compression initialization."]
        dummy_inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=data_args.max_seq_length
        )
        dummy_inputs = {k: v.to(training_args.device) for k, v in dummy_inputs.items()}
        dummy_inputs['labels'] = dummy_inputs['input_ids'].clone()

        class DummyDataset:
            def __init__(self, inputs):
                # Remove batch dimension from inputs
                self.inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            def __getitem__(self, idx):
                return self.inputs
            def __len__(self):
                return 1

        dummy_loader = torch.utils.data.DataLoader(
            DummyDataset(dummy_inputs),
            batch_size=1
        )

        # Set up compressors
        sparsifiers, projectors = setup_model_compressors(
            model=model,
            layer_names=layer_names,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            train_dataloader=dummy_loader,
            device=str(training_args.device)
        )

        if sparsifiers:
            hook_manager.set_sparsifiers(sparsifiers)
            logger.info(f"  ✓ Set {len(sparsifiers)} sparsifiers")

        if projectors:
            hook_manager.set_projectors(projectors)
            logger.info(f"  ✓ Set {len(projectors)} projectors")

        logger.info("Gradient compression setup completed!")
    else:
        logger.info("Gradient compression disabled (sparsification and projection both None)")

    # Prepare validation dataset (used for data selection in GREATS)
    val_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir='data',
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        validation=True,
        k=training_args.n_val,
        subject=training_args.subject
    )

    # Prepare evaluation dataset (held-out test set for measuring generalization)
    eval_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir='data',
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        validation=False,
        k=training_args.n_eval,
        subject=training_args.subject
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Initialize trainer with hook manager
    trainer = HookTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        hook_manager=hook_manager,
    )

    # Train
    logger.info("*** Starting training with hook-based gradient computation ***")
    train_result = trainer.train()

    # Save final evaluation results
    trainer.on_train_end()

    # Save model
    trainer.save_model()

    # Clean up hooks (only if they were registered)
    if hook_manager.hooks_registered:
        hook_manager.remove_hooks()
        logger.info("Removed all hooks after training")
    else:
        logger.info("No hooks to remove (hooks were not registered)")

    return train_result


if __name__ == "__main__":
    main()
