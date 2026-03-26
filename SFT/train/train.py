#!/usr/bin/env python
# coding=utf-8
"""
Training script for SFT with layer-wise descent.
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

from SFT.data.get_train_dataset import get_training_dataset
from SFT.data.get_val_dataset import get_dataset, DEFAULT_SEQ_LENGTH_MULTIPLIER

from drpt import (
    GradientHook,
    setup_model_compressors,
    create_sample_inputs,
    CompressionMode,
)
from SFT.train.trainer import LayerwiseTrainer

from SFT.train.data_arguments import DataArguments, get_data_statistics
from SFT.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from SFT.train.training_arguments import TrainingArguments


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
        seed=training_args.data_seed if training_args.data_seed is not None else training_args.seed,
        train_files=data_args.train_files if data_args.train_files else None,
        train_dataset_names=training_args.train_dataset_names
    )

    avg_train_seq_length = get_data_statistics(train_dataset, return_avg_length=True)
    logger.info(f"Average training sequence length: {avg_train_seq_length:.1f}")

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

    # Log layer names for verification
    if len(layer_names) > 0:
        logger.info("=== Layer Names for Hook Attachment ===")
        for i, name in enumerate(layer_names[:5]):  # Show first 5
            logger.info(f"  [{i}] {name}")
        if len(layer_names) > 5:
            logger.info(f"  ... and {len(layer_names) - 5} more layers")
    else:
        logger.warning("WARNING: No trainable layers found! Check model and lora_only setting.")

    # Determine if gradient hooks are needed based on training method
    # Hooks are needed for:
    # 1. Selection method is not NA (Layerwise or Subset)
    # 2. Compression enabled (implies MeSO optimizer)
    has_explicit_compression = (training_args.sparsification is not None or training_args.projection is not None)
    needs_selection = training_args.method in ('Layerwise', 'Subset', 'LayerwiseWithVal')
    needs_grad_hook = needs_selection or has_explicit_compression

    # Determine compression needs independently for score and update
    has_score_compression = (needs_selection
                             and training_args.score_compression is not None)
    has_update_compression = (training_args.sparsification is not None
                              or training_args.projection is not None)

    # Create gradient hook only when needed
    grad_hook = None
    if needs_grad_hook:
        grad_hook = GradientHook(
            model=model,
            layer_names=layer_names,
            device=str(training_args.device),
        )
    else:
        logger.info(f"Training method: {training_args.method} - No gradient hooks needed")

    # Helper: parse sparsification string (e.g., "normal-512*512") into kwargs
    def _parse_sparsifier(spec_str):
        method, dim_str = spec_str.split("-")
        assert "*" in dim_str, f"Sparsification dimension must be factorized (e.g., 'normal-64*64'), got '{spec_str}'"
        dim = int(dim_str.split("*")[0])
        return {
            "proj_dim": dim, "proj_max_batch_size": 64,
            "proj_seed": training_args.seed, "device": str(training_args.device),
            "proj_type": method,
        }, f"{method}-{dim}*{dim}"

    def _parse_projector(spec_str):
        method, dim_str = spec_str.split("-")
        assert "*" not in dim_str, f"Projection dimension must not be factorized, got '{spec_str}'"
        dim = int(dim_str)
        return {
            "proj_dim": dim, "proj_max_batch_size": 64,
            "proj_seed": training_args.seed, "device": str(training_args.device),
            "proj_type": method,
        }, f"{method}-{dim}"

    _identity_sparsifier = {
        "proj_dim": -1, "proj_max_batch_size": 64,
        "proj_seed": training_args.seed, "device": str(training_args.device),
        "proj_type": "identity",
    }
    _identity_projector = {
        "proj_dim": -1, "proj_max_batch_size": 64,
        "proj_seed": training_args.seed, "device": str(training_args.device),
        "proj_type": "identity",
    }

    # Set up compressors
    if has_score_compression or has_update_compression:
        logger.info("=== Gradient Compression Setup ===")

        # Create sample inputs (shared for both compressor sets)
        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            device=str(training_args.device)
        )

        # --- Update compressors (MeSO optimizer) ---
        if has_update_compression:
            update_sparsifier_kwargs, update_desc = _parse_sparsifier(training_args.sparsification)
            if training_args.projection is not None:
                update_projector_kwargs, proj_desc = _parse_projector(training_args.projection)
            else:
                update_projector_kwargs = _identity_projector
                proj_desc = "none"
            logger.info(f"  Update compression (MeSO): sparsifier={update_desc}, projector={proj_desc}")

            update_compressors = setup_model_compressors(
                model=model, layer_names=layer_names,
                sparsifier_kwargs=update_sparsifier_kwargs,
                projector_kwargs=update_projector_kwargs,
                sample_inputs=sample_inputs,
                device=str(training_args.device),
                update_freq=training_args.update_compressor_freq,
            )
            grad_hook.set_update_compressors(update_compressors)
        else:
            logger.info("  Update compression: none")

        # --- Score compressors (influence scoring) ---
        if has_score_compression:
            # Check if score compression matches update compression → share objects
            score_spec = training_args.score_compression
            update_spec = training_args.sparsification
            if has_update_compression and score_spec == update_spec:
                logger.info(f"  Score compression: same as update ({score_spec}) → sharing compressors")
                grad_hook.set_score_compressors(grad_hook.update_compressors)
            else:
                score_sparsifier_kwargs, score_desc = _parse_sparsifier(score_spec)
                logger.info(f"  Score compression: sparsifier={score_desc}")

                score_compressors = setup_model_compressors(
                    model=model, layer_names=layer_names,
                    sparsifier_kwargs=score_sparsifier_kwargs,
                    projector_kwargs=_identity_projector,
                    sample_inputs=sample_inputs,
                    device=str(training_args.device),
                    update_freq=training_args.update_compressor_freq,
                )
                grad_hook.set_score_compressors(score_compressors)
        else:
            logger.info("  Score compression: none (exact scoring)")

        logger.info(f"  Compression mode: {grad_hook.compression_mode.value}")
        logger.info("Gradient compression setup completed!")
    else:
        if grad_hook is not None:
            logger.info(f"Compression mode: {grad_hook.compression_mode.value} (no compressors)")
        if needs_selection:
            logger.info("Gradient compression disabled (exact scoring)")
        else:
            logger.info("Gradient compression disabled (no selection and no MeSO)")

    # Validate scoring_method vs compression config (strict — no auto-detection)
    if has_score_compression and training_args.scoring_method != "compress":
        raise ValueError(
            f"scoring.compression is set ({training_args.score_compression}) but "
            f"scoring.method='{training_args.scoring_method}'. "
            f"Set scoring.method to 'compress' or remove scoring.compression."
        )
    if training_args.scoring_method == "compress" and not has_score_compression:
        raise ValueError(
            "scoring.method='compress' requires scoring.compression to be set. "
            "Add 'compression: normal-64*64' under the scoring section."
        )

    # Prepare validation dataset (used for data selection in layer-wise descent)
    # Use rejection sampling to filter out validation samples that are significantly
    # longer than the average training sequence length
    val_seq_length_threshold = int(avg_train_seq_length * DEFAULT_SEQ_LENGTH_MULTIPLIER)
    logger.info(f"Validation sequence length threshold: {val_seq_length_threshold} ({DEFAULT_SEQ_LENGTH_MULTIPLIER}x avg train length)")

    val_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        split="validation",
        k=training_args.n_val,
        subject=training_args.subject,
        seed=training_args.seed,
        max_seq_length_threshold=val_seq_length_threshold
    )

    # Prepare evaluation dataset (held-out test set for measuring generalization)
    eval_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        split=data_args.eval_split,
        k=training_args.n_eval,
        subject=training_args.subject
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Initialize layer-wise trainer with data selection capabilities
    trainer = LayerwiseTrainer(
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
    logger.info("*** Starting training ***")
    train_result = trainer.train()

    # Final evaluation after training
    logger.info("*** Running final evaluation after training ***")
    trainer.evaluate()

    # Save final evaluation results
    trainer.on_train_end()

    # Save model
    trainer.save_model()

    # Clean up hooks (only if grad_hook was created)
    if grad_hook is not None and grad_hook.hooks_registered:
        grad_hook.remove_hooks()
        logger.info("Removed gradient hooks")

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")

    return train_result


if __name__ == "__main__":
    main()
