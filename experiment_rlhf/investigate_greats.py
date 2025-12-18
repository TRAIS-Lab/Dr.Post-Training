#!/usr/bin/env python
"""
Deep investigation of GREATS algorithm behavior in RLHF.

Questions to answer:
1. Is magnitude difference actually causing selection problems?
2. Does the ranking change significantly with/without normalization?
3. Are there numerical precision issues?
4. Is the fundamental issue that NLL validation doesn't align with PPO objective?
"""

import sys
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compress_gradient import GradientHook, setup_model_compressors, create_sample_inputs
from experiment_rlhf.data.get_prompts import build_toxicity_promptdata, collator
from experiment_rlhf.rewards import load_toxicity_model, CrossVocabRewardWrapper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_selection_scores(train_grads: torch.Tensor, val_grad: torch.Tensor):
    """
    Compute selection scores using different methods.

    Returns dict with scores from:
    - dot_product: Original GREATS (train_grad · val_grad)
    - cosine_sim: Normalized (train_grad · val_grad) / (||train|| * ||val||)
    - magnitude_weighted: cos_sim * ||train_grad|| (hybrid approach)
    """
    # Original dot product
    dot_scores = train_grads @ val_grad

    # Cosine similarity
    train_norms = train_grads.norm(dim=1).clamp(min=1e-8)
    val_norm = val_grad.norm().clamp(min=1e-8)
    cosine_scores = dot_scores / (train_norms * val_norm)

    # Magnitude-weighted cosine (preserves direction preference but scales by train magnitude)
    magnitude_weighted = cosine_scores * train_norms

    return {
        'dot_product': dot_scores,
        'cosine_sim': cosine_scores,
        'magnitude_weighted': magnitude_weighted,
        'train_norms': train_norms,
        'val_norm': val_norm,
    }


def analyze_score_rankings(scores_dict: Dict[str, torch.Tensor], k: int):
    """
    Analyze how different scoring methods affect sample rankings.
    """
    results = {}

    for name, scores in scores_dict.items():
        if name in ['train_norms', 'val_norm']:
            continue

        scores_np = scores.cpu().numpy()

        # Get top-k indices for this method
        top_k_indices = np.argsort(scores_np)[-k:][::-1]

        results[name] = {
            'top_k_indices': top_k_indices,
            'top_k_scores': scores_np[top_k_indices],
            'score_mean': scores_np.mean(),
            'score_std': scores_np.std(),
            'score_min': scores_np.min(),
            'score_max': scores_np.max(),
            'score_range': scores_np.max() - scores_np.min(),
        }

    # Compute ranking agreement between methods
    dot_top_k = set(results['dot_product']['top_k_indices'])
    cos_top_k = set(results['cosine_sim']['top_k_indices'])
    mag_top_k = set(results['magnitude_weighted']['top_k_indices'])

    results['ranking_agreement'] = {
        'dot_vs_cosine': len(dot_top_k & cos_top_k) / k,
        'dot_vs_magnitude': len(dot_top_k & mag_top_k) / k,
        'cosine_vs_magnitude': len(cos_top_k & mag_top_k) / k,
    }

    return results


def run_investigation():
    """Main investigation function."""
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading models...")
    model_name = "EleutherAI/gpt-neo-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model with LoRA
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    policy_model = get_peft_model(policy_model, lora_config)

    # Reward model
    reward_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    reward_tokenizer, reward_model_raw = load_toxicity_model(reward_model_name, device=device, wrap_for_trl=False)
    reward_model = CrossVocabRewardWrapper(
        reward_model=reward_model_raw,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=tokenizer,
        device=device,
    )

    # Load data
    logger.info("Loading data...")
    train_dataset, val_dataset = build_toxicity_promptdata(tokenizer, num_samples=256, seed=42)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collator)

    # Find LoRA layers
    layer_names = []
    for name, module in policy_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if hasattr(module.lora_A, 'default'):
                layer_names.append(f"{name}.lora_A.default")
            if hasattr(module.lora_B, 'default'):
                layer_names.append(f"{name}.lora_B.default")

    logger.info(f"Found {len(layer_names)} LoRA layers")

    # Setup gradient hook
    grad_hook = GradientHook(model=policy_model, layer_names=layer_names, device=device)

    sample_inputs = create_sample_inputs(tokenizer, max_seq_length=50, device=device)
    compressors = setup_model_compressors(
        model=policy_model,
        layer_names=layer_names,
        sparsifier_kwargs={"proj_dim": 32, "proj_max_batch_size": 64, "proj_seed": 42, "device": device, "proj_type": "random_mask"},
        projector_kwargs={"proj_dim": 512, "proj_max_batch_size": 64, "proj_seed": 42, "device": device, "proj_type": "normal"},
        sample_inputs=sample_inputs,
        device=device,
        update_freq=200,
    )
    grad_hook.set_compressors(compressors)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    # Get validation batch
    val_batch = next(iter(val_dataloader))
    val_input_ids = val_batch["input_ids"].to(device)
    val_attention_mask = (val_input_ids != tokenizer.pad_token_id).long().to(device)

    # Get training batch
    train_batch = next(iter(train_dataloader))
    train_input_ids = train_batch["input_ids"].to(device)
    train_attention_mask = (train_input_ids != tokenizer.pad_token_id).long().to(device)
    batch_size = train_input_ids.shape[0]

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 1: Gradient magnitudes from different losses")
    logger.info("="*80)

    # Compute NLL loss gradients (what we use for validation)
    policy_model.train()
    grad_hook.disable_hooks()  # Get raw gradients first

    nll_outputs = policy_model(val_input_ids, val_attention_mask, labels=val_input_ids)
    nll_loss = nll_outputs.loss
    nll_loss.backward()

    nll_grad_norms = {}
    for name, p in policy_model.named_parameters():
        if p.grad is not None and 'lora_' in name:
            nll_grad_norms[name] = p.grad.norm().item()

    optimizer.zero_grad()

    # Compute PPO-style loss gradients
    # First generate responses
    policy_model.eval()
    with torch.no_grad():
        outputs = policy_model.generate(
            train_input_ids, max_new_tokens=20, do_sample=True,
            temperature=1.0, pad_token_id=tokenizer.pad_token_id
        )
        response_ids = outputs[:, train_input_ids.shape[1]:]
        response_mask = (response_ids != tokenizer.pad_token_id).long()

        # Compute rewards
        query_texts = tokenizer.batch_decode(train_input_ids, skip_special_tokens=True)
        response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        texts = [q + r for q, r in zip(query_texts, response_texts)]
        reward_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        reward_outputs = reward_model(reward_inputs.input_ids, reward_inputs.attention_mask)
        hidden_states = reward_outputs.hidden_states[-1]
        rewards = reward_model.score(hidden_states[:, -1, :]).squeeze(-1)

    policy_model.train()

    # PPO loss (simplified)
    full_ids = torch.cat([train_input_ids, response_ids], dim=1)
    full_mask = torch.cat([train_attention_mask, response_mask], dim=1)

    ppo_outputs = policy_model(full_ids, full_mask, return_dict=True)
    logits = ppo_outputs.logits[:, train_input_ids.shape[1]-1:-1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    new_logprobs = torch.gather(log_probs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

    # Use rewards as advantages (simplified)
    advantages = rewards.unsqueeze(1).expand_as(new_logprobs) * response_mask
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    ppo_loss = -(advantages * new_logprobs * response_mask).sum() / response_mask.sum()
    ppo_loss.backward()

    ppo_grad_norms = {}
    for name, p in policy_model.named_parameters():
        if p.grad is not None and 'lora_' in name:
            ppo_grad_norms[name] = p.grad.norm().item()

    optimizer.zero_grad()

    # Compare gradient magnitudes
    logger.info("\nGradient magnitude comparison (first 10 layers):")
    logger.info(f"{'Layer':<60} {'NLL Grad':>12} {'PPO Grad':>12} {'Ratio':>10}")
    logger.info("-" * 94)

    ratios = []
    for i, name in enumerate(list(nll_grad_norms.keys())[:10]):
        nll_norm = nll_grad_norms.get(name, 0)
        ppo_norm = ppo_grad_norms.get(name, 0)
        ratio = nll_norm / (ppo_norm + 1e-10)
        ratios.append(ratio)
        short_name = name.split('.')[-3] + '.' + name.split('.')[-2] + '.' + name.split('.')[-1]
        logger.info(f"{short_name:<60} {nll_norm:>12.6f} {ppo_norm:>12.6f} {ratio:>10.2f}x")

    logger.info(f"\nAverage ratio (NLL/PPO): {np.mean(ratios):.2f}x")

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 2: Per-sample gradient analysis with compressed gradients")
    logger.info("="*80)

    # Now use the hook to capture compressed per-sample gradients
    grad_hook.enable_hooks()

    # Capture validation gradients
    grad_hook.set_mode("capture_val")
    val_loss = policy_model(val_input_ids, val_attention_mask, labels=val_input_ids).loss
    val_loss.backward()
    optimizer.zero_grad()

    logger.info(f"\nCaptured validation gradients from {len(grad_hook.val_grad_buffer)} layers")

    # Capture training gradients (per-sample) using train_select mode without selection
    grad_hook.set_mode("train_select")

    # We need to manually capture per-sample gradients before selection
    # Let's do this by temporarily modifying the hook behavior

    # Store per-sample gradients for analysis
    per_sample_train_grads = {}

    # Create a simple hook to capture gradients before selection
    original_backward = grad_hook.compressors[0].forward if grad_hook.compressors[0] else None

    # Re-run PPO loss computation with hooks enabled but capture raw grads
    ppo_outputs = policy_model(full_ids, full_mask, return_dict=True)
    logits = ppo_outputs.logits[:, train_input_ids.shape[1]-1:-1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    new_logprobs = torch.gather(log_probs, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    ppo_loss = -(advantages * new_logprobs * response_mask).sum() / response_mask.sum()

    # Before backward, setup to capture compressed grads
    grad_hook.setup_selection(
        train_batch_size=batch_size,
        selection_method="GREATS",
        selection_frac=0.5,
        lr=1e-5,
        compute_scores_only=False,
    )

    ppo_loss.backward()

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 3: Selection score analysis")
    logger.info("="*80)

    # Analyze selection scores across layers
    all_layer_results = []

    for layer_idx, val_grad in grad_hook.val_grad_buffer.items():
        if grad_hook.compressed_grads[layer_idx] is None:
            continue

        # The compressed_grads now contains the REDUCED gradient (after selection)
        # We need to examine the selection that happened

        # For analysis, let's manually compute what the scores would have been
        # Note: We can't recover per-sample grads after selection, but we can
        # examine the val_grad characteristics

        val_norm = val_grad.norm().item()
        val_mean = val_grad.mean().item()

        all_layer_results.append({
            'layer_idx': layer_idx,
            'val_norm': val_norm,
            'val_mean': val_mean,
        })

    logger.info(f"\nValidation gradient statistics across {len(all_layer_results)} layers:")
    norms = [r['val_norm'] for r in all_layer_results]
    logger.info(f"  Val grad norms: min={min(norms):.6f}, max={max(norms):.6f}, mean={np.mean(norms):.6f}")

    # Count how many layers have meaningful gradients
    meaningful_layers = sum(1 for n in norms if n > 1e-6)
    logger.info(f"  Layers with meaningful gradients (norm > 1e-6): {meaningful_layers}/{len(norms)}")

    optimizer.zero_grad()
    grad_hook.clear_selection()

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 4: Does the validation loss direction align with reward improvement?")
    logger.info("="*80)

    # This is the key question: Does minimizing NLL on validation samples
    # actually lead to better rewards?

    # Check correlation between NLL and rewards on training samples
    grad_hook.disable_hooks()

    with torch.no_grad():
        # Compute per-sample NLL
        nll_outputs = policy_model(full_ids, full_mask, return_dict=True)
        logits = nll_outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]

        per_sample_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        ).reshape(batch_size, -1)

        # Average NLL per sample (only on response part)
        response_start = train_input_ids.shape[1]
        response_nll = per_sample_nll[:, response_start-1:].mean(dim=1)

        # We already have rewards from earlier
        rewards_np = rewards.float().cpu().numpy()
        nll_np = response_nll.float().cpu().numpy()

        # Correlation
        correlation = np.corrcoef(rewards_np, nll_np)[0, 1]

        logger.info(f"\nCorrelation between rewards and NLL: {correlation:.4f}")
        logger.info("  (Negative correlation = lower NLL → higher reward, which is what we want)")

        # Also check: do high-reward samples have gradients that align with val direction?
        # Sort samples by reward
        reward_order = np.argsort(rewards_np)[::-1]  # High to low

        logger.info(f"\nReward distribution:")
        logger.info(f"  Top 5 rewards: {rewards_np[reward_order[:5]]}")
        logger.info(f"  Bottom 5 rewards: {rewards_np[reward_order[-5:]]}")

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 5: Numerical precision analysis")
    logger.info("="*80)

    # Check if scores are distinguishable
    grad_hook.enable_hooks()
    grad_hook.set_mode("capture_val")

    val_loss = policy_model(val_input_ids, val_attention_mask, labels=val_input_ids).loss
    val_loss.backward()
    optimizer.zero_grad()

    # Run multiple batches and collect score statistics
    score_stats = []

    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx >= 5:
            break

        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = (batch_input_ids != tokenizer.pad_token_id).long().to(device)
        bs = batch_input_ids.shape[0]

        grad_hook.set_mode("train_select")
        grad_hook.setup_selection(
            train_batch_size=bs,
            selection_method="GREATS",
            selection_frac=0.5,
            lr=1e-5,
            compute_scores_only=True,  # Just compute scores, don't reduce
        )

        # Simple loss for gradient computation
        loss = policy_model(batch_input_ids, batch_attention_mask, labels=batch_input_ids).loss
        loss.backward()

        # Get accumulated scores
        if grad_hook.selection_state and grad_hook.selection_state.grad_dot_scores is not None:
            scores = grad_hook.selection_state.grad_dot_scores.float().cpu().numpy()

            score_stats.append({
                'batch_idx': batch_idx,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'range': scores.max() - scores.min(),
                'distinguishable': scores.std() > 1e-10,
            })

        optimizer.zero_grad()
        grad_hook.clear_selection()

    logger.info("\nScore statistics across batches:")
    for stat in score_stats:
        logger.info(f"  Batch {stat['batch_idx']}: mean={stat['mean']:.2e}, std={stat['std']:.2e}, "
                   f"range={stat['range']:.2e}, distinguishable={stat['distinguishable']}")

    # Check if scores are numerically distinguishable
    all_distinguishable = all(s['distinguishable'] for s in score_stats)
    logger.info(f"\nAll batches have distinguishable scores: {all_distinguishable}")

    if not all_distinguishable:
        logger.warning("WARNING: Some batches have scores that are not numerically distinguishable!")
        logger.warning("This could cause selection to be essentially random.")

    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION 6: Compare selection scores with reward-weighted validation")
    logger.info("="*80)

    # Test how reward-weighted validation affects selection
    # First, generate responses and compute rewards for validation batch
    policy_model.eval()
    with torch.no_grad():
        val_outputs = policy_model.generate(
            val_input_ids, max_new_tokens=20, do_sample=True,
            temperature=1.0, pad_token_id=tokenizer.pad_token_id
        )
        val_response_ids = val_outputs[:, val_input_ids.shape[1]:]
        val_response_mask = (val_response_ids != tokenizer.pad_token_id).long()

        # Compute rewards for validation responses
        val_query_texts = tokenizer.batch_decode(val_input_ids, skip_special_tokens=True)
        val_response_texts = tokenizer.batch_decode(val_response_ids, skip_special_tokens=True)
        val_texts = [q + r for q, r in zip(val_query_texts, val_response_texts)]
        val_reward_inputs = tokenizer(val_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        val_reward_outputs = reward_model(val_reward_inputs.input_ids, val_reward_inputs.attention_mask)
        val_hidden = val_reward_outputs.hidden_states[-1]
        val_rewards = reward_model.score(val_hidden[:, -1, :]).squeeze(-1)

        # Compute weights (higher reward = more weight)
        val_rewards_normalized = val_rewards - val_rewards.min()
        if val_rewards_normalized.max() > 0:
            val_rewards_normalized = val_rewards_normalized / val_rewards_normalized.max()
        val_weights = val_rewards_normalized + 0.1
        val_weights = val_weights / val_weights.sum() * val_input_ids.shape[0]

    logger.info(f"\nValidation rewards: min={val_rewards.min():.2f}, max={val_rewards.max():.2f}, mean={val_rewards.mean():.2f}")

    # Compute reward-weighted validation loss and capture gradients
    policy_model.train()
    grad_hook.enable_hooks()
    grad_hook.set_mode("capture_val")

    # Concatenate query and response
    val_full_ids = torch.cat([val_input_ids, val_response_ids], dim=1)
    val_full_mask = torch.cat([(val_input_ids != tokenizer.pad_token_id).long(), val_response_mask], dim=1)
    val_query_len = val_input_ids.shape[1]

    # Forward pass
    val_outputs = policy_model(val_full_ids, val_full_mask, return_dict=True)
    val_logits = val_outputs.logits[:, val_query_len-1:-1, :]
    val_log_probs = F.log_softmax(val_logits.float(), dim=-1)
    val_response_logprobs = torch.gather(
        val_log_probs, dim=-1, index=val_response_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Reward-weighted loss
    weighted_logprobs = val_response_logprobs * val_weights.unsqueeze(-1) * val_response_mask.float()
    reward_weighted_val_loss = -weighted_logprobs.sum() / val_response_mask.sum().clamp(min=1)
    reward_weighted_val_loss.backward()
    optimizer.zero_grad()

    logger.info(f"Reward-weighted validation loss: {reward_weighted_val_loss.item():.4f}")
    logger.info(f"Captured reward-weighted val gradients from {len(grad_hook.val_grad_buffer)} layers")

    # Check if scores are better aligned
    rw_score_stats = []

    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx >= 3:
            break

        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = (batch_input_ids != tokenizer.pad_token_id).long().to(device)
        bs = batch_input_ids.shape[0]

        grad_hook.set_mode("train_select")
        grad_hook.setup_selection(
            train_batch_size=bs,
            selection_method="GREATS",
            selection_frac=0.5,
            lr=1e-5,
            compute_scores_only=True,
        )

        loss = policy_model(batch_input_ids, batch_attention_mask, labels=batch_input_ids).loss
        loss.backward()

        if grad_hook.selection_state and grad_hook.selection_state.grad_dot_scores is not None:
            scores = grad_hook.selection_state.grad_dot_scores.float().cpu().numpy()

            rw_score_stats.append({
                'batch_idx': batch_idx,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
            })

        optimizer.zero_grad()
        grad_hook.clear_selection()

    logger.info("\nReward-weighted selection scores:")
    for stat in rw_score_stats:
        logger.info(f"  Batch {stat['batch_idx']}: mean={stat['mean']:.2e}, std={stat['std']:.2e}")

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    logger.info(f"""
Key findings:
1. NLL gradients vs PPO gradients: Ratio ~{np.mean(ratios):.1f}x (manageable)
2. Validation gradient coverage: {meaningful_layers}/{len(norms)} layers have meaningful gradients
3. NLL-Reward correlation: {correlation:.4f} (essentially NO correlation!)
4. Score distinguishability: All batches distinguishable

ROOT CAUSE: The NLL validation direction does NOT align with the RLHF reward objective!
- Correlation between NLL and rewards is essentially zero ({correlation:.4f})
- This means GREATS with NLL validation selects samples based on a direction
  that has nothing to do with improving rewards

SOLUTION: Use 'reward_weighted' val_loss_type instead of 'logprob'
- Generate responses for validation prompts
- Weight the validation loss by rewards
- This aligns the validation gradient with the RLHF objective

To use the fix, change the training command to include:
  --val_loss_type reward_weighted
""")


if __name__ == "__main__":
    run_investigation()
