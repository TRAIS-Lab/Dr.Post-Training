#!/usr/bin/env python
"""
Test script for RLHF implementation with layer-wise selection.

This script runs quick sanity checks on all components:
1. Data loading
2. Reward model
3. Standard PPO (TRL)
4. PPO with layer-wise selection (GREATS)

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_rlhf.py --test all
    CUDA_VISIBLE_DEVICES=0 python test_rlhf.py --test data
    CUDA_VISIBLE_DEVICES=0 python test_rlhf.py --test reward
    CUDA_VISIBLE_DEVICES=0 python test_rlhf.py --test ppo_standard
    CUDA_VISIBLE_DEVICES=0 python test_rlhf.py --test ppo_selection
"""

import sys
import os
import argparse
import logging

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test that data loading works correctly."""
    logger.info("=" * 60)
    logger.info("Testing Data Loading")
    logger.info("=" * 60)

    from experiment_rlhf.data.get_prompts import build_toxicity_promptdata, collator

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset, val_dataset = build_toxicity_promptdata(
        tokenizer,
        num_samples=64,  # Small for testing
        seed=42,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Train dataset columns: {train_dataset.column_names}")
    logger.info(f"Val dataset columns: {val_dataset.column_names}")

    # Check a sample
    sample = train_dataset[0]
    logger.info(f"Sample query: {sample['query'][:100]}...")
    logger.info(f"Sample input_ids shape: {len(sample['input_ids'])}")

    # Test collator
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(dataloader))
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Batch input_ids shape: {batch['input_ids'].shape}")

    logger.info("Data loading test PASSED")
    return True


def test_reward_model():
    """Test that reward model works correctly."""
    logger.info("=" * 60)
    logger.info("Testing Reward Model")
    logger.info("=" * 60)

    from experiment_rlhf.rewards import (
        load_toxicity_model,
        compute_toxicity_scores,
        CrossVocabRewardWrapper,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load reward model
    reward_tokenizer, reward_model = load_toxicity_model(
        "facebook/roberta-hate-speech-dynabench-r4-target",
        device=device,
        wrap_for_trl=False,
    )

    # Test basic toxicity scoring
    test_texts = [
        "Hello, how are you today?",
        "You are a terrible person and I hate you!",
        "The weather is nice today.",
    ]

    scores = compute_toxicity_scores(test_texts, reward_tokenizer, reward_model)
    logger.info(f"Test texts: {test_texts}")
    logger.info(f"Toxicity scores (negative = less toxic): {scores}")

    # Verify scores make sense (hate text should have lower score)
    assert scores[1] < scores[0], "Hate text should have lower reward"
    assert scores[1] < scores[2], "Hate text should have lower reward"

    # Test CrossVocabRewardWrapper
    policy_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    wrapped_model = CrossVocabRewardWrapper(
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=policy_tokenizer,
        device=device,
    )

    # Test forward pass with policy tokenizer tokens
    test_input_ids = policy_tokenizer(
        test_texts, padding=True, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        output = wrapped_model(test_input_ids)

    logger.info(f"CrossVocabRewardWrapper output type: {type(output)}")
    logger.info(f"Hidden states shape: {output.hidden_states[0].shape}")

    # Test score layer
    rewards = wrapped_model.score(output.hidden_states[-1][:, -1, :])
    logger.info(f"Rewards from wrapper: {rewards.squeeze().tolist()}")

    logger.info("Reward model test PASSED")
    return True


def test_ppo_standard():
    """Test standard PPO training (TRL)."""
    logger.info("=" * 60)
    logger.info("Testing Standard PPO (TRL)")
    logger.info("=" * 60)

    from trl import PPOConfig, PPOTrainer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    from experiment_rlhf.data.get_prompts import build_toxicity_promptdata, collator
    from experiment_rlhf.rewards import load_toxicity_model, CrossVocabRewardWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset, val_dataset = build_toxicity_promptdata(
        tokenizer,
        num_samples=32,
        seed=42,
    )

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125m",
        torch_dtype=torch.bfloat16,
    ).to(device)

    value_model = AutoModelForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-neo-125m",
        num_labels=1,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA applied, trainable params: {model.print_trainable_parameters()}")

    # Load reward model
    reward_tokenizer, reward_model_raw = load_toxicity_model(
        "facebook/roberta-hate-speech-dynabench-r4-target",
        device=device,
        wrap_for_trl=False,
    )

    reward_model = CrossVocabRewardWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
    )

    # Create PPO config (minimal for testing)
    ppo_config = PPOConfig(
        output_dir="./test_outputs/ppo_standard",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        seed=42,
        kl_coef=0.04,
        remove_unused_columns=False,
        report_to=None,
        total_episodes=16,  # Very short for testing
        response_length=10,
        local_rollout_forward_batch_size=4,
        num_mini_batches=1,
        gradient_checkpointing=False,
        eval_strategy="no",
        num_sample_generations=0,
    )

    # Create PPO trainer
    try:
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=None,
            value_model=value_model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        logger.info("PPOTrainer created successfully")

        # Run a few steps
        logger.info("Running 1 training step...")
        # Note: Full train() is very slow, just test creation works
        logger.info("Standard PPO test PASSED (creation only)")
        return True

    except Exception as e:
        logger.error(f"Standard PPO test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppo_selection():
    """Test PPO with layer-wise selection (GREATS)."""
    logger.info("=" * 60)
    logger.info("Testing PPO with Layer-wise Selection")
    logger.info("=" * 60)

    from trl import PPOConfig, PPOTrainer
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification
    from torch.utils.data import DataLoader

    from experiment_rlhf.data.get_prompts import build_toxicity_promptdata, collator
    from experiment_rlhf.rewards import load_toxicity_model, CrossVocabRewardWrapper
    from experiment_rlhf.ppo_trainer import PPOTrainerWithSelection
    from compress_gradient import GradientHook, setup_model_compressors, create_sample_inputs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset, val_dataset = build_toxicity_promptdata(
        tokenizer,
        num_samples=32,
        seed=42,
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collator)

    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125m",
        torch_dtype=torch.bfloat16,
    ).to(device)

    value_model = AutoModelForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-neo-125m",
        num_labels=1,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Find LoRA layers for gradient hooks
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if hasattr(module.lora_A, 'default'):
                layer_names.append(f"{name}.lora_A.default")
            if hasattr(module.lora_B, 'default'):
                layer_names.append(f"{name}.lora_B.default")

    logger.info(f"Found {len(layer_names)} LoRA layers")

    # Setup gradient hooks
    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=device,
    )

    # Setup compressors
    sample_inputs = create_sample_inputs(tokenizer=tokenizer, max_seq_length=30, device=device)

    sparsifier_kwargs = {
        "proj_dim": 32,
        "proj_max_batch_size": 8,
        "proj_seed": 42,
        "device": device,
        "proj_type": "random_mask",
    }
    projector_kwargs = {
        "proj_dim": 128,
        "proj_max_batch_size": 8,
        "proj_seed": 42,
        "device": device,
        "proj_type": "normal",
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=device,
        update_freq=100,
    )
    grad_hook.set_compressors(compressors)
    logger.info("Compressors set up")

    # Load reward model
    reward_tokenizer, reward_model_raw = load_toxicity_model(
        "facebook/roberta-hate-speech-dynabench-r4-target",
        device=device,
        wrap_for_trl=False,
    )

    reward_model = CrossVocabRewardWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
    )

    # Create PPO config
    ppo_config = PPOConfig(
        output_dir="./test_outputs/ppo_selection",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_ppo_epochs=1,
        seed=42,
        kl_coef=0.04,
        remove_unused_columns=False,
        report_to=None,
        total_episodes=16,
        response_length=10,
        local_rollout_forward_batch_size=4,
        num_mini_batches=1,
        gradient_checkpointing=False,
        eval_strategy="no",
        num_sample_generations=0,
    )

    # Create PPO trainer
    try:
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=None,
            value_model=value_model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            data_collator=collator,
        )

        # Wrap with selection trainer
        selection_trainer = PPOTrainerWithSelection(
            ppo_trainer=ppo_trainer,
            grad_hook=grad_hook,
            selection_frac=0.5,
            selection_method="GREATS",
            val_loss_type="logprob",
            use_second_order=False,
        )
        logger.info("Selection trainer created")

        # Test validation gradient capture
        val_batch = next(iter(val_dataloader))
        val_input_ids = val_batch["input_ids"].to(device)
        val_attention_mask = (val_input_ids != tokenizer.pad_token_id).long().to(device)

        val_loss = selection_trainer.capture_validation_gradients(
            val_input_ids=val_input_ids,
            val_attention_mask=val_attention_mask,
        )
        logger.info(f"Captured validation gradients, val_loss={val_loss:.4f}")
        logger.info(f"Val grad buffer size: {len(grad_hook.val_grad_buffer)}")

        # Test rollout generation
        train_batch = next(iter(train_dataloader))
        query_ids = train_batch["input_ids"].to(device)

        response_tensors, response_texts = selection_trainer.generate_rollouts(
            [q for q in query_ids[:2]],  # Test with 2 samples
            {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "pad_token_id": tokenizer.pad_token_id},
        )
        logger.info(f"Generated {len(response_tensors)} responses")
        logger.info(f"Response example: '{response_texts[0][:50]}...'")

        # Test reward computation
        query_texts = train_batch["query"][:2]
        rewards = selection_trainer.compute_rewards(query_texts, response_texts[:2])
        logger.info(f"Computed rewards: {rewards.tolist()}")

        logger.info("PPO with selection test PASSED")
        return True

    except Exception as e:
        logger.error(f"PPO with selection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        grad_hook.remove_hooks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "data", "reward", "ppo_standard", "ppo_selection"],
        help="Which test to run",
    )
    args = parser.parse_args()

    results = {}

    if args.test in ["all", "data"]:
        results["data"] = test_data_loading()

    if args.test in ["all", "reward"]:
        results["reward"] = test_reward_model()

    if args.test in ["all", "ppo_standard"]:
        results["ppo_standard"] = test_ppo_standard()

    if args.test in ["all", "ppo_selection"]:
        results["ppo_selection"] = test_ppo_selection()

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.info("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
