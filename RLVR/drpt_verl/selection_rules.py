"""
Selection rules for the gather-then-select ("v2") RLVR update path.

Every function here is pure torch and works on a score matrix ``S`` of shape ``[rows, N]``:
one row per decision maker (a single row for Global Subset, one row per hooked layer for
Layer-Wise Subset) and one column per rollout of the current mini-batch, gathered over all
data-parallel ranks. A score is the inner product between a rollout's contribution to the
actual policy update and the target gradient, so summing scores over the rollouts of one
prompt gives the alignment of that prompt's whole GRPO group update with the target.

Rules
-----
mode = "filtering": keep every candidate with a non-negative score; with ``frac < 1`` only the
    most negative ``int(frac * n_negative)`` candidates are dropped (the legacy drpt semantics).
mode = "topk":      keep the ``max(1, int(frac * n))`` best-scoring candidates.
level = "rollout":  candidates are rollouts.
level = "prompt":   candidates are prompts (score = sum over the prompt's rollouts); the decision
    is broadcast to all rollouts of the prompt, so GRPO's within-group advantage baseline stays
    intact (a dropped prompt removes a zero-sum group, a kept prompt keeps the whole group).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

VALID_MODES = ("filtering", "topk")
VALID_LEVELS = ("rollout", "prompt")
VALID_NORMALIZATIONS = ("none", "layer_meanabs", "layer_std")
_ADV_EPS = 1e-8


@dataclass
class SelectionRule:
    mode: str = "filtering"
    frac: float = 1.0
    level: str = "rollout"
    score_normalization: str = "none"  # Global Subset only: rescale each layer's scores before summing

    def validate(self) -> "SelectionRule":
        if self.mode not in VALID_MODES:
            raise ValueError(f"selection_mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.level not in VALID_LEVELS:
            raise ValueError(f"selection_level must be one of {VALID_LEVELS}, got {self.level!r}")
        if self.score_normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"score_normalization must be one of {VALID_NORMALIZATIONS}, got {self.score_normalization!r}"
            )
        if not (0.0 <= self.frac <= 1.0):
            raise ValueError(f"frac must lie in [0, 1], got {self.frac}")
        return self


def normalize_layer_scores(S: Tensor, how: str) -> Tensor:
    """Rescale every row of ``S`` (one layer) so no layer dominates a subsequent sum by magnitude."""
    if how == "none":
        return S
    if how == "layer_meanabs":
        scale = S.abs().mean(dim=1, keepdim=True)
    elif how == "layer_std":
        scale = S.std(dim=1, keepdim=True)
    else:
        raise ValueError(f"unknown score_normalization {how!r}")
    scale = scale.clamp(min=torch.finfo(S.dtype).tiny)
    return S / scale


def group_ids_from_uids(uids: Sequence[str]) -> Tuple[Tensor, int]:
    """Map prompt uids to dense integer group ids (first-appearance order)."""
    order: Dict[str, int] = {}
    ids = [order.setdefault(str(u), len(order)) for u in uids]
    return torch.tensor(ids, dtype=torch.long), len(order)


def aggregate_by_group(S: Tensor, group_ids: Tensor, num_groups: int) -> Tensor:
    """Sum the columns of ``S`` [rows, N] per group -> [rows, num_groups]."""
    out = S.new_zeros(S.shape[0], num_groups)
    out.index_add_(1, group_ids.to(S.device), S)
    return out


def keep_mask_rows(S: Tensor, mode: str, frac: float) -> Tensor:
    """Apply the rule independently to every row of ``S`` [rows, n]; returns a bool mask."""
    rows, n = S.shape
    if mode == "filtering":
        if frac >= 1.0:
            return S >= 0
        if frac <= 0.0:
            return torch.ones_like(S, dtype=torch.bool)
        negative = S < 0
        n_drop = (negative.sum(dim=1).to(S.dtype) * frac).floor().long()  # per row
        order = torch.argsort(S, dim=1)  # ascending: the most negative candidates come first
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(n, device=S.device).expand(rows, n))
        drop = negative & (ranks < n_drop[:, None])
        return ~drop
    if mode == "topk":
        k = min(n, max(1, int(n * frac)))
        idx = torch.topk(S, k, dim=1, largest=True, sorted=False).indices
        keep = torch.zeros_like(S, dtype=torch.bool)
        keep.scatter_(1, idx, True)
        return keep
    raise ValueError(f"unknown selection mode {mode!r}")


def select_masks(S: Tensor, group_ids: Tensor, num_groups: int, rule: SelectionRule) -> Tensor:
    """Bool keep-mask [rows, N] for the score matrix under ``rule``."""
    if rule.level == "rollout":
        return keep_mask_rows(S, rule.mode, rule.frac)
    G = aggregate_by_group(S, group_ids, num_groups)
    keep_groups = keep_mask_rows(G, rule.mode, rule.frac)
    return keep_groups[:, group_ids.to(S.device)]


def recentered_advantage_shift(adv: Tensor, keep: Tensor, group_ids: Tensor, num_groups: int) -> Tensor:
    """
    Per-sample shift that re-centres the advantages of the KEPT rollouts within each prompt
    group (kept advantages sum to zero again after ``adv - shift``); zero for dropped rollouts.
    """
    keep_f = keep.to(adv.dtype)
    gid = group_ids.to(adv.device)
    count = adv.new_zeros(num_groups).index_add_(0, gid, keep_f)
    total = adv.new_zeros(num_groups).index_add_(0, gid, adv * keep_f)
    mean = total / count.clamp(min=1)
    return mean[gid] * keep_f


def selection_diagnostics(keep: Tensor, adv: Tensor, group_ids: Tensor, num_groups: int) -> Dict[str, float]:
    """
    Summaries of a keep-mask [rows, N] against the per-rollout advantages [N]:
    keep fractions by advantage sign, mean advantage of the kept set (the "baseline bias"),
    fraction of prompts with at least one kept rollout, and the per-row keep-rate spread.
    """
    keep_f = keep.float()
    adv = adv.float().to(keep_f.device)
    pos, neg = adv > _ADV_EPS, adv < -_ADV_EPS
    zero = ~(pos | neg)
    out: Dict[str, float] = {"keep_frac": keep_f.mean().item()}
    for name, m in (("pos", pos), ("neg", neg), ("zero", zero)):
        if bool(m.any()):
            out[f"keep_frac_{name}_adv"] = keep_f[:, m].mean().item()
    kept_adv = (keep_f * adv[None, :]).sum(dim=1) / keep_f.sum(dim=1).clamp(min=1)
    out["kept_adv_mean"] = kept_adv.mean().item()
    informative = pos | neg
    if bool(informative.any()):
        # signed advantage mass surviving selection, relative to the total |advantage| mass
        out["kept_adv_mass"] = ((keep_f * adv[None, :]).sum(dim=1) / adv.abs().sum()).mean().item()
    kept_groups = aggregate_by_group(keep_f, group_ids, num_groups) > 0
    out["kept_prompt_frac"] = kept_groups.float().mean().item()
    if keep.shape[0] > 1:
        rates = keep_f.mean(dim=1)
        out["layer_keep_min"] = rates.min().item()
        out["layer_keep_median"] = rates.median().item()
        out["layer_keep_max"] = rates.max().item()
    return out
