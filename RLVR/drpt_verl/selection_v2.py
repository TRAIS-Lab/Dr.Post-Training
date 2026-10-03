"""
Gather-then-select ("v2") policy update for the RLVR selection actor.

The one-pass paths decide per rollout from the SIGN of its score inside each micro-batch of
``ppo_micro_batch_size_per_gpu`` samples (Layer-Wise in every layer's backward, Global in a
scoring pass per micro-batch). That is enough for negative filtering but not for rules that
need to see the whole mini-batch: top-k budgets, prompt-level decisions (all rollouts of a
prompt kept or dropped together, which preserves GRPO's within-group baseline), layer-
normalized Global scores, or re-centred advantages. This module restructures one mini-batch
update into

  1. scoring pass    : one forward/backward per micro-batch with the Global scoring state
                       recording every hooked layer's per-sample scores  -> S_rank [L, n_rank]
  2. gather          : all DP ranks exchange (S_rank, per-sample advantage, prompt uid)
  3. rule            : Global sums the (optionally layer-normalized) rows; Layer-Wise keeps all
                       rows; ``selection_rules.select_masks`` yields a keep-mask [rows, N]
  4. training pass   : Global trains on the kept rollouts of each micro-batch; Layer-Wise runs
                       the hooked backward with the per-layer masks fixed in advance.

Scores are the raw <per-sample update contribution, target gradient> values produced by the
hook under the actor's loss aggregation, so they are on the scale of the update that is actually
applied; no re-normalization across micro-batches is done.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.torch_functional import logprobs_from_logits

try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ImportError:  # pragma: no cover
    pad_input = unpad_input = rearrange = index_first_axis = None

from .selection_rules import (
    SelectionRule,
    group_ids_from_uids,
    normalize_layer_scores,
    recentered_advantage_shift,
    select_masks,
    selection_diagnostics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# small helpers
# =============================================================================

def _rank_world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_gather_objects(obj: Any) -> List[Any]:
    rank, world = _rank_world()
    if world == 1:
        return [obj]
    out: List[Any] = [None] * world
    dist.all_gather_object(out, obj)
    return out


def per_sample_advantage(advantages: Tensor, response_mask: Tensor) -> Tensor:
    """GRPO advantages are constant over a response; recover the per-sample scalar [b]."""
    m = response_mask.to(advantages.dtype)
    return (advantages * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def _forward_log_prob(actor, input_ids, attention_mask, position_ids, responses, temperature, with_entropy):
    """Log-probs [b, R] (and entropy [b, R] or None) of the given samples, packed or padded path."""
    import verl.utils.torch_functional as verl_F

    response_length = responses.size(1)
    batch_size = input_ids.size(0)
    entropy = None
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        if actor.use_remove_padding and unpad_input is not None:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)
            output = actor.actor_module(
                input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
            )
            logits_rmpad = output.logits.squeeze(0)
            if temperature != 1.0:
                logits_rmpad = logits_rmpad / temperature
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
            log_probs_rmpad = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)
            full_log_probs = pad_input(
                hidden_states=log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=input_ids.size(1)
            )
            log_prob = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
            if with_entropy:
                entropy_rmpad = verl_F.entropy_from_logits(logits_rmpad)
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=input_ids.size(1)
                )
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
        else:
            output = actor.actor_module(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
            )
            logits = output.logits
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits[:, -response_length - 1:-1, :]
            log_prob = logprobs_from_logits(logits.float(), responses)
            if with_entropy:
                entropy = verl_F.entropy_from_logits(logits)
    return log_prob, entropy


def _response_labels(responses: Tensor, response_mask: Tensor, attention_mask: Tensor) -> Tensor:
    labels = torch.full_like(attention_mask, -100)
    response_length = responses.size(1)
    labels[:, -response_length:] = responses * response_mask.long() + (-100) * (~response_mask.bool()).long()
    return labels


# =============================================================================
# scoring pass
# =============================================================================

def score_micro_batch_layers(actor, model_inputs: Dict[str, Any], temperature: float) -> Tuple[Tensor, Tensor]:
    """
    One scoring forward/backward over a micro-batch with the Global scoring state recording every
    layer's per-sample scores. Returns (S [L, b] float32 on device, scored [L] bool).
    """
    from verl.trainer.ppo import core_algos

    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    position_ids = model_inputs['position_ids']
    responses = model_inputs['responses']
    old_log_probs = model_inputs['old_log_probs']
    advantages = model_inputs['advantages']
    response_mask = model_inputs['response_mask']
    b = input_ids.size(0)

    hook = actor.grad_hook
    hook.setup_selection_with_stored_val(
        train_batch_size=b,
        selection_method='GlobalSubset',
        frac=1.0,
        lr=1.0,
        compute_scores_only=True,
        use_second_order=False,
        selection_mode='filtering',
        record_layer_scores=True,
    )
    hook.enable_hooks()
    labels = _response_labels(responses, response_mask, attention_mask)
    hook.set_token_counts(labels, b, attention_mask)
    actor.actor_module.zero_grad()

    log_prob, _ = _forward_log_prob(actor, input_ids, attention_mask, position_ids, responses, temperature, False)
    pg_loss, _ = core_algos.compute_policy_loss_vanilla(
        old_log_prob=old_log_probs,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=actor.config.loss_agg_mode,
        config=actor.config,
    )
    pg_loss.backward()

    S, scored = hook.selection_state.layer_score_matrix(len(hook.layer_names))
    hook.clear_selection()
    hook.disable_hooks()
    hook.clear_token_counts()
    actor.actor_optimizer.zero_grad()
    return S, scored


# =============================================================================
# training pass (one micro-batch)
# =============================================================================

def train_micro_batch(
    actor,
    model_inputs: Dict[str, Any],
    temperature: float,
    loss_scale_factor: float,
    metrics: Dict[str, Any],
    *,
    keep_idx: Optional[Tensor] = None,
    adv_shift: Optional[Tensor] = None,
    fixed_selections: Optional[Dict[int, Tensor]] = None,
) -> None:
    """
    Forward/backward for one micro-batch.

    Global: ``keep_idx`` (LongTensor of kept positions, may be empty -> dummy zero-weight backward
    to keep FSDP in sync) and optional ``adv_shift`` [b] subtracted from the advantages of kept
    samples. Layer-Wise: ``fixed_selections`` (layer_idx -> kept positions) consumed by the hooked
    backward; all samples are forwarded.
    """
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
    from verl.utils.py_functional import append_to_dict

    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    position_ids = model_inputs['position_ids']
    responses = model_inputs['responses']
    old_log_probs = model_inputs['old_log_probs']
    advantages = model_inputs['advantages']
    response_mask = model_inputs['response_mask']
    ref_log_prob = model_inputs.get('ref_log_prob')
    b = input_ids.size(0)

    use_zero_weight = False
    layer_wise = fixed_selections is not None
    if not layer_wise:
        if keep_idx is None:
            keep_idx = torch.arange(b, device=input_ids.device)
        if keep_idx.numel() == 0:
            logger.warning("selection v2: no samples kept in a micro-batch, dummy backward for FSDP sync")
            keep_idx = torch.zeros(1, dtype=torch.long, device=input_ids.device)
            use_zero_weight = True
        if adv_shift is not None:
            advantages = (advantages - adv_shift.to(advantages.dtype)[:, None]) * response_mask.to(advantages.dtype)
        input_ids, attention_mask, position_ids = input_ids[keep_idx], attention_mask[keep_idx], position_ids[keep_idx]
        responses, old_log_probs, advantages = responses[keep_idx], old_log_probs[keep_idx], advantages[keep_idx]
        response_mask = response_mask[keep_idx]
        if ref_log_prob is not None:
            ref_log_prob = ref_log_prob[keep_idx]
    else:
        labels = _response_labels(responses, response_mask, attention_mask)
        actor._setup_layer_wise_subset(b, labels, attention_mask)
        actor.grad_hook.selection_state.fixed_selections = fixed_selections

    try:
        entropy_coeff = actor.config.entropy_coeff
        log_prob, entropy = _forward_log_prob(
            actor, input_ids, attention_mask, position_ids, responses, temperature, entropy_coeff != 0
        )
        pg_loss, pg_metrics = core_algos.compute_policy_loss_vanilla(
            old_log_prob=old_log_probs,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=actor.config.loss_agg_mode,
            config=actor.config,
        )
        micro_batch_metrics = dict(pg_metrics)
        policy_loss = pg_loss
        if entropy_coeff != 0:
            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=actor.config.loss_agg_mode)
            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
            policy_loss = policy_loss - entropy_agg * entropy_coeff
        if actor.config.use_kl_loss:
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=actor.config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=actor.config.loss_agg_mode)
            policy_loss = policy_loss + kl_loss * actor.config.kl_loss_coef
            if not use_zero_weight:
                metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
            micro_batch_metrics["actor/kl_coef"] = actor.config.kl_loss_coef

        loss = policy_loss * loss_scale_factor
        if use_zero_weight:
            loss = loss * 0.0
        loss.backward()

        if not use_zero_weight:
            metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
        append_to_dict(metrics, micro_batch_metrics)
    finally:
        if layer_wise:
            actor._cleanup_layer_wise_subset()
            actor.grad_hook.disable_hooks()


# =============================================================================
# the update
# =============================================================================

class _DiagAccumulator:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def summary(self) -> Dict[str, float]:
        return {k: self.sums[k] / self.counts[k] for k in self.sums}


def update_policy_v2(actor, data: DataProto) -> Dict[str, Any]:
    """Gather-then-select mini-batch updates for the GlobalSubset / LayerWiseSubset actor."""
    from verl.utils.py_functional import append_to_dict

    if actor.config.use_dynamic_bsz:
        raise NotImplementedError("selection v2 assumes fixed micro-batches (use_dynamic_bsz=False)")
    if getattr(actor, 'tie_embeddings', False):
        raise NotImplementedError("selection v2 does not support tie_embeddings (one mask per tied pair)")

    rule = actor.selection_rule
    method = actor.selection_method
    recenter = bool(actor.recenter_advantages) and method == 'GlobalSubset' and rule.level == 'rollout'

    actor.actor_module.train()
    temperature = data.meta_info["temperature"]
    pad_token_id = data.meta_info.get("pad_token_id", 0)

    select_keys = ['responses', 'response_mask', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
    if actor.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    has_uid = 'uid' in data.non_tensor_batch
    data = data.select(batch_keys=select_keys, non_tensor_batch_keys=['uid'] if has_uid else None)
    if rule.level == 'prompt' and not has_uid:
        raise RuntimeError("prompt-level selection needs the prompt uid in the actor batch")

    mini_batches = data.split(actor.config.ppo_mini_batch_size)
    metrics: Dict[str, Any] = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}
    diag = _DiagAccumulator()
    rank, world = _rank_world()
    num_layers = len(actor.grad_hook.layer_names)

    for _ in range(actor.config.ppo_epochs):
        for mini_batch in mini_batches:
            actor.gradient_accumulation = actor.config.ppo_mini_batch_size // actor.config.ppo_micro_batch_size_per_gpu
            loss_scale_factor = 1.0 / actor.gradient_accumulation
            micro_batches_list = mini_batch.split(actor.config.ppo_micro_batch_size_per_gpu)

            # ---- 1. scoring pass ------------------------------------------------------------
            prepared: List[Tuple[Dict[str, Any], int]] = []
            score_blocks, adv_blocks, uid_blocks, scored_any = [], [], [], None
            for micro_batch in micro_batches_list:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                S_mb, scored = score_micro_batch_layers(actor, model_inputs, temperature)
                score_blocks.append(S_mb)
                scored_any = scored if scored_any is None else (scored_any | scored)
                adv_blocks.append(per_sample_advantage(model_inputs['advantages'], model_inputs['response_mask']))
                b = model_inputs['input_ids'].size(0)
                uid_blocks.extend([str(u) for u in model_inputs['uid']] if has_uid else [f"r{rank}_{len(uid_blocks) + i}" for i in range(b)])
                prepared.append((model_inputs, b))
            S_rank = torch.cat(score_blocks, dim=1).cpu()          # [L, n_rank]
            adv_rank = torch.cat(adv_blocks).float().cpu()          # [n_rank]
            scored_rank = scored_any.cpu()

            # ---- 2. gather across DP ranks ------------------------------------------------------
            gathered = _all_gather_objects((S_rank, adv_rank, uid_blocks, scored_rank))
            S_all = torch.cat([g[0] for g in gathered], dim=1)      # [L, N]
            adv_all = torch.cat([g[1] for g in gathered])           # [N]
            uid_all: List[str] = [u for g in gathered for u in g[2]]
            scored_all = torch.stack([g[3] for g in gathered]).any(dim=0)  # [L]
            col_offset = sum(g[0].shape[1] for g in gathered[:rank])
            n_rank = S_rank.shape[1]
            group_ids, num_groups = group_ids_from_uids(uid_all)

            # ---- 3. rule --------------------------------------------------------------------------
            if method == 'GlobalSubset':
                S_eff = normalize_layer_scores(S_all[scored_all], rule.score_normalization).sum(dim=0, keepdim=True)
            else:
                S_eff = S_all
            keep_all = select_masks(S_eff, group_ids, num_groups, rule)  # [rows, N]
            if method != 'GlobalSubset':
                keep_all[~scored_all] = True  # layers without a target gradient keep everything
            if getattr(actor, 'drop_zero_adv', False):
                # drop_zero_adv: rollouts of all-correct / all-wrong groups carry no
                # policy gradient; dropping them removes their KL-penalty gradient and their share of the token-mean denominator.
                keep_all[:, adv_all.abs() < 1e-8] = False
            shift_all = recentered_advantage_shift(adv_all, keep_all[0], group_ids, num_groups) if recenter else None
            d = selection_diagnostics(keep_all if method != 'GlobalSubset' else keep_all, adv_all, group_ids, num_groups)
            d['num_prompts'] = float(num_groups)
            d['num_samples'] = float(S_all.shape[1])
            if shift_all is not None:
                d['adv_shift_abs_mean'] = shift_all.abs().mean().item()
            diag.update(d)

            keep_rank = keep_all[:, col_offset:col_offset + n_rank]
            shift_rank = shift_all[col_offset:col_offset + n_rank] if shift_all is not None else None

            # ---- 4. training pass -------------------------------------------------------------
            actor.actor_optimizer.zero_grad()
            offset = 0
            for model_inputs, b in prepared:
                m = keep_rank[:, offset:offset + b]
                device = model_inputs['input_ids'].device
                if method == 'GlobalSubset':
                    keep_idx = m[0].nonzero(as_tuple=False).view(-1).to(device)
                    shift = shift_rank[offset:offset + b].to(device) if shift_rank is not None else None
                    train_micro_batch(actor, model_inputs, temperature, loss_scale_factor, metrics,
                                      keep_idx=keep_idx, adv_shift=shift)
                else:
                    fixed = {
                        int(l): m[l].nonzero(as_tuple=False).view(-1).to(device)
                        for l in range(num_layers) if bool(scored_all[l])
                    }
                    train_micro_batch(actor, model_inputs, temperature, loss_scale_factor, metrics,
                                      fixed_selections=fixed)
                offset += b

            grad_norm = actor._optimizer_step()
            append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

    stats = diag.summary()
    actor.selection_stats['train/total_selected'] = stats.get('keep_frac', 0.0) * stats.get('num_samples', 0.0)
    actor.selection_stats['train/total_samples'] = stats.get('num_samples', 0.0)
    actor.selection_stats['train/overall_selection_ratio'] = stats.get('keep_frac', 0.0)
    for k, v in stats.items():
        actor.selection_stats[f'train/{k}'] = v
    for key, val in actor.selection_stats.items():
        metrics[f'selection/{key}'] = val

    actor.actor_optimizer.zero_grad()
    return metrics
