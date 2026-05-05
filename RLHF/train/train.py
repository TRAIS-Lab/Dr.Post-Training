#!/usr/bin/env python
# coding=utf-8
"""
Training script for RLHF with layer_wise_subset data selection.
"""

import logging
import os
import sys
import warnings

import torch
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser, set_seed, get_scheduler
from trl import AutoModelForCausalLMWithValueHead
from trl.models import create_reference_model

# Suppress torch.compile warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

from RLHF.data.get_prompts import get_prompt_dataset, get_validation_prompt_dataset, collator
from RLHF.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from RLHF.train.training_arguments import TrainingArguments
from RLHF.train.trainer import LayerWiseSubsetPPOTrainer
from RLHF.train.rewards import load_reward_model, RewardModelWrapper
from RLHF.train.evaluator import create_evaluator

from drpt import GradientHook, setup_model_compressors, create_sample_inputs, MeSOAdamW, CompressionMode

logger = logging.getLogger(__name__)


def _patch_gpt_neo_flash_attention():
    """
    Monkey-patch fix for GPT-Neo Flash Attention 2 bug in transformers.

    Bug: GPTNeoFlashAttention2.forward assumes attention_mask is 4D, but when
    using FA2 with padded batches, _update_causal_mask returns the 2D mask directly.
    This causes: IndexError: too many indices for tensor of dimension 2

    This patch fixes the mask slicing to handle both 2D and 4D masks correctly.
    """
    try:
        from transformers.models.gpt_neo import modeling_gpt_neo

        if not hasattr(modeling_gpt_neo, "GPTNeoFlashAttention2"):
            return  # FA2 class doesn't exist in this version

        original_forward = modeling_gpt_neo.GPTNeoFlashAttention2.forward

        def patched_forward(self, hidden_states, attention_mask=None, layer_past=None,
                           head_mask=None, use_cache=False, output_attentions=False,
                           cache_position=None):
            bsz, _, _ = hidden_states.size()

            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

            if layer_past is not None:
                cache_kwargs = {"cache_position": cache_position}
                key, value = layer_past.update(key, value, self.layer_id, cache_kwargs)

            query_length = query.shape[2]
            tgt_len = key.shape[2]

            query = query.transpose(1, 2).view(bsz, query_length, self.num_heads, self.head_dim)
            key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
            value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

            attn_dropout = self.config.attention_dropout if self.training else 0.0

            # FIX: Only slice if attention_mask is 4D (the bug was assuming 4D always)
            if attention_mask is not None and attention_mask.dim() == 4:
                attention_mask = attention_mask[:, :, :, : key.shape[1]]
            # For 2D masks, pass as-is (Flash Attention handles them correctly)

            # Handle dtype casting for PEFT
            device_type = query.device.type if query.device.type != "mps" else "cpu"
            if query.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = (
                        torch.get_autocast_dtype(device_type)
                        if hasattr(torch, "get_autocast_dtype")
                        else torch.get_autocast_gpu_dtype()
                    )
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                query = query.to(target_dtype)
                key = key.to(target_dtype)
                value = value.to(target_dtype)

            from transformers.modeling_flash_attention_utils import _flash_attention_forward

            attn_output = _flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length,
                dropout=attn_dropout,
                softmax_scale=1.0,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )

            attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
            attn_output = self.out_proj(attn_weights_reshaped)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights_reshaped

        modeling_gpt_neo.GPTNeoFlashAttention2.forward = patched_forward
        logger.info("Applied GPT-Neo Flash Attention 2 monkey-patch fix")
    except Exception as e:
        logger.warning(f"Failed to apply GPT-Neo FA2 patch: {e}")


# Apply the patch at import time
_patch_gpt_neo_flash_attention()


def find_trainable_layers(model, lora_only: bool = True, include_v_head: bool = False):
    """
    Find trainable layers for gradient hooks.

    Args:
        model: The model (can be AutoModelForCausalLMWithValueHead or base model)
        lora_only: If True, only find LoRA layers
        include_v_head: If True, include value head in hooks. Default False because:
                       - Selection is based on policy gradient alignment with validation
                       - Validation loss (reward-weighted log-probs) has no gradient to v_head
                       - Including v_head would give zero/random selection scores
                       - Reference (LDA-ORL) explicitly excludes v_head from selection
                       The value head trains on full batch via standard autograd.

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
                # Skip lm_head (weight-tied with embeddings in many models)
                if "lm_head" not in name:
                    layer_names.append(f"{prefix}{name}")

    # Include value head if present and requested
    # v_head is a sibling of pretrained_model, not a child, so we check separately
    if include_v_head and hasattr(model, "v_head"):
        # v_head is typically a Sequential with Linear layers
        for name, module in model.v_head.named_modules():
            if isinstance(module, torch.nn.Linear):
                full_name = f"v_head.{name}" if name else "v_head"
                layer_names.append(full_name)

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
    logger.info("RLHF Training with Dr. Post-Training")
    logger.info(f"  Task: {training_args.task}")
    logger.info(f"  Method: {training_args.method}")
    logger.info(f"  Model: {model_args.model_name_or_path}")
    logger.info(f"  LoRA: {model_args.lora}")
    if training_args.has_selection:
        logger.info(f"  Filter fraction (drop negative): {training_args.filter_frac}")
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

    # Load validation dataset (if n_val > 0 AND method needs it)
    # Skip for Standard (NA) to avoid perturbing the global RNG state,
    # which would cause Standard results to differ across n_val settings.
    val_dataset = None
    if training_args.use_validation_set and training_args.has_selection:
        val_dataset = get_validation_prompt_dataset(
            task=training_args.task,
            tokenizer=tokenizer,
            n_val=training_args.n_val,
            seed=training_args.seed,
        )
        logger.info(f"  Validation dataset: {len(val_dataset)} samples (fixed set)")
    elif training_args.has_selection:
        logger.info("  Validation: self-referencing (training buffer)")
    else:
        logger.info("  Validation: none (Standard method)")

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
        summary_dropout_prob=0.0,
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

    # Reference model handling (matching reference implementation ppo_trainer.py:222-233):
    # - For PEFT (LoRA) models: No separate reference model needed
    #   The trainer uses disable_adapter() on the policy model
    # - For non-PEFT models: Use create_reference_model() with shared layers
    #   This is more memory efficient than loading a separate frozen copy
    ref_model = None
    if not model_args.lora:
        logger.info("Creating reference model with shared layers for non-PEFT training...")
        ref_model = create_reference_model(model)
        logger.info("Reference model created (shares layers with policy model)")
    else:
        logger.info("Using disable_adapter() for reference logprobs (PEFT model)")

    # Load reward model
    logger.info("Loading reward model...")
    reward_tokenizer, reward_model_raw = load_reward_model(
        model_args.reward_model_name,
        device=device,
        task=training_args.task,
    )

    # Wrap reward model to handle vocabulary mismatch
    reward_model = RewardModelWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
        task=training_args.task,
    )

    # Find trainable layers
    layer_names = find_trainable_layers(model, lora_only=model_args.lora)
    logger.info(f"Found {len(layer_names)} trainable layers")

    # Create gradient hook only when needed (selection or compression enabled)
    grad_hook = None

    # Determine compression needs independently
    has_score_compression = (training_args.has_selection
                             and training_args.score_compression is not None)
    has_update_compression = training_args.has_compression

    if training_args.has_selection or has_update_compression:
        grad_hook = GradientHook(
            model=model,
            layer_names=layer_names,
            device=device,
        )
    else:
        logger.info(f"Training method: {training_args.method} - No gradient hooks needed")

    # Helper: parse sparsification string into kwargs
    def _parse_sparsifier(spec_str):
        method, dim_str = spec_str.split("-")
        assert "*" in dim_str, f"Sparsification dimension must be factorized, got '{spec_str}'"
        dim = int(dim_str.split("*")[0])
        return {
            "proj_dim": dim, "proj_max_batch_size": training_args.per_device_train_batch_size,
            "proj_seed": training_args.seed, "device": device, "proj_type": method,
        }, f"{method}-{dim}*{dim}"

    _identity_projector = {
        "proj_dim": -1, "proj_max_batch_size": training_args.per_device_train_batch_size,
        "proj_type": "identity", "device": device,
    }

    if has_score_compression or has_update_compression:
        logger.info("=== Gradient Compression Setup ===")

        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer,
            max_seq_length=training_args.max_new_tokens + 20,
            device=device,
        )

        # --- Update compressors (MeSO) ---
        if has_update_compression:
            update_sparsifier_kwargs, update_desc = _parse_sparsifier(training_args.sparsification)
            logger.info(f"  Update compression (MeSO): sparsifier={update_desc}")
            update_compressors = setup_model_compressors(
                model=model, layer_names=layer_names,
                sparsifier_kwargs=update_sparsifier_kwargs,
                projector_kwargs=_identity_projector,
                sample_inputs=sample_inputs, device=device,
                update_freq=training_args.update_compressor_freq,
            )
            grad_hook.set_update_compressors(update_compressors)
        else:
            logger.info("  Update compression: none")

        # --- Score compressors ---
        if has_score_compression:
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
                    sample_inputs=sample_inputs, device=device,
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

    # Create optimizer - separate parameter groups for policy and value head
    # This allows using different learning rates for each
    policy_params = []
    vhead_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "v_head" in name:
                vhead_params.append(param)
            else:
                policy_params.append(param)

    # Determine value head learning rate
    vhead_lr = training_args.learning_rate_vhead if training_args.learning_rate_vhead is not None else training_args.learning_rate

    # Create parameter groups
    param_groups = [
        {"params": policy_params, "lr": training_args.learning_rate},
        {"params": vhead_params, "lr": vhead_lr},
    ]
    logger.info(f"Optimizer: {len(policy_params)} policy params (lr={training_args.learning_rate}), "
                f"{len(vhead_params)} v_head params (lr={vhead_lr})")

    if training_args.has_compression:
        # Use MeSO optimizer with gradient hook for compressed gradient access
        optimizer = MeSOAdamW(
            param_groups,
            grad_hook=grad_hook,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            compressed_layer_names=layer_names,
        )
        logger.info("Using MeSO optimizer")
    else:
        # Standard AdamW
        optimizer = torch.optim.AdamW(
            param_groups,
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

    # Create evaluator (uses different metrics than reward model)
    # This provides unbiased evaluation during training
    evaluator = None
    if training_args.enable_eval:
        logger.info(f"Creating evaluator for task: {training_args.task}...")
        evaluator = create_evaluator(
            task=training_args.task,
            device=device,
            batch_size=64,
        )
        logger.info("Evaluator ready")

    # Create trainer
    trainer = LayerWiseSubsetPPOTrainer(
        model=model,
        ref_model=ref_model,  # None for PEFT, shared-layer ref model for non-PEFT
        reward_model=reward_model,
        tokenizer=tokenizer,
        args=training_args,
        grad_hook=grad_hook,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        evaluator=evaluator,
        val_dataset=val_dataset,  # Fixed validation dataset (None if using self-referencing)
    )

    # Calculate training steps
    num_epochs = training_args.num_train_epochs
    max_steps = training_args.max_steps if training_args.max_steps > 0 else None

    # Use logging_steps from training args
    log_interval = int(training_args.logging_steps)

    # Initial evaluation before training (step 0)
    initial_eval_results = None
    if training_args.enable_eval and evaluator is not None:
        logger.info("*** Running initial evaluation before training ***")
        initial_eval_results = evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            n_samples=training_args.n_eval,
            max_new_tokens=training_args.max_new_tokens,
            min_new_tokens=training_args.min_new_tokens,  # Safe for eval (no KL computation)
            generation_batch_size=training_args.eval_batch_size,
            temperature=training_args.temperature,
            top_p=training_args.top_p,
        )
        # Log toxicity metrics
        logger.info(
            f"[Initial Eval] toxicity: {initial_eval_results['mean_toxicity_prob']:.4f} "
            f"(rate={initial_eval_results['toxicity_rate']:.1%})"
        )

    # Train
    logger.info("*** Starting training ***")
    train_result = trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=int(num_epochs),
        max_steps=max_steps,
        log_interval=log_interval,
    )

    # Save model
    final_output_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_output_dir)

    # Add initial evaluation results to history
    if initial_eval_results is not None:
        train_result["initial_eval_toxicity_prob"] = initial_eval_results["mean_toxicity_prob"]
        train_result["initial_eval_toxicity_rate"] = initial_eval_results["toxicity_rate"]

    # Save training history with evaluation results
    import json
    history_file = os.path.join(training_args.output_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(train_result, f, indent=2)
    logger.info(f"Training history saved to {history_file}")

    # Clean up hooks (only if grad_hook was created)
    if grad_hook is not None and grad_hook.hooks_registered:
        grad_hook.remove_hooks()
        logger.info("Removed gradient hooks")

    logger.info("Training complete!")
    return train_result


if __name__ == "__main__":
    main()
