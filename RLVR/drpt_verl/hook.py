"""
GradientHook for RLVR with Verl.

Extends base GradientHook with:
- Packed sequence support via cu_seqlens
- Hook call tracking for debugging multi-GPU setups
- FSDP compatibility
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os
import weakref
import logging

from torch.autograd import Function

import torch.distributed as dist

from .selection_state import SelectionStateVerl, LayerWiseSubsetStateVerl, GlobalSubsetStateVerl
from .packed_ops import (
    compute_scores_packed_vectorized,
    compute_selected_gradients_packed_vectorized,
    compute_scores_and_similarity_standard,
    compute_selected_gradients_standard,
    compute_embedding_val_gradient,
    compute_embedding_scores_packed,
    compute_embedding_selected_gradient_packed,
    compute_embedding_scores_standard,
    compute_embedding_selected_gradient_standard,
)

logger = logging.getLogger(__name__)


def _get_rank() -> int:
    """Get current distributed rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    """Get distributed world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


class _ValCacheWrapper:
    """Wrapper for API compatibility with base library's _val_cache."""

    def __init__(self, hook: "GradientHookVerl"):
        self._hook = hook

    def get_num_captured(self) -> int:
        """Return number of layers with captured validation gradients."""
        return len(self._hook._val_grad_cache)


def split_train_val_batch(tensor: Tensor, train_batch_size: int) -> Tuple[Tensor, Tensor]:
    """Split a merged batch tensor into train and validation portions."""
    return tensor[:train_batch_size], tensor[train_batch_size:]


class GradientHookVerl:
    """
    Hook manager for custom gradient computation in RLVR/Verl.

    Key features:
    - Packed sequence support (use_remove_padding=True)
    - Hook call tracking for debugging
    - Optimized vectorized operations without D2H transfers
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cuda',
        loss_agg_mode: str = "token-mean",
        tie_embeddings: bool = False,
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (nn.Linear and nn.Embedding modules)
            device: Device for tensors
            loss_agg_mode: verl ``actor.loss_agg_mode`` of the policy loss the actor
                backpropagates. It decides what one "item" of the batch loss is (see
                ``drpt.losses``): ``token-mean`` (verl's default and official GRPO/DAPO
                practice) averages over response tokens, so an item is a token and
                samples are weighted by length (``item_convention == "token"``); the
                ``seq-mean-*`` modes average per-sequence losses over sequences, so every
                sample is one item and the curated update is a plain mean over the kept
                samples (``item_convention == "sample"``).
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.loss_agg_mode = loss_agg_mode
        self.item_convention = "sample" if loss_agg_mode.startswith("seq-mean") else "token"
        # Tied embedding / output matrix (opt-in). With ``tie_embeddings`` the two hooked uses of a shared
        # weight (nn.Embedding lookup and nn.Linear output layer) share ONE target gradient, the sum of the
        # two site gradients (what autograd gives the shared parameter), so the sum of the two site scores
        # is the exact full-parameter inner product including the cross terms; under Layer-Wise Subset the
        # output layer defers its curated gradient to the embedding's backward, which selects one subset for
        # the shared weight from the summed scores. Default False reproduces the independent-groups treatment.
        self.tie_embeddings = tie_embeddings
        self.tied_pairs: Dict[int, int] = {}
        self._tied_embedding_indices: set = set()
        self._tied_stash: Dict[int, dict] = {}
        self._tied_merged = False

        self.layer_name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(layer_names)}
        self.layer_name_to_module: Dict[str, nn.Module] = {}

        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True

        # Selection state
        self.selection_state: Optional[SelectionStateVerl] = None

        # Validation gradient cache (simple full gradient storage for RLVR)
        self._val_grad_cache: Dict[int, Tensor] = {}
        self._val_total_tokens: Optional[int] = None
        self._capturing_val: bool = False

        # Token tracking
        self.total_tokens: Optional[Tensor] = None

        # Hook call tracking for debugging
        self._hook_call_count = 0
        self._hook_call_count_by_layer: Dict[str, int] = {}

        # Register hooks
        self._register_hooks()
        if tie_embeddings:
            self._detect_tied_pairs()

        logger.info(
            f"Initialized GradientHookVerl with {len(layer_names)} layers "
            f"(loss_agg_mode={loss_agg_mode}, item={self.item_convention})"
        )

    def _register_hooks(self) -> None:
        """Monkey-patch Linear and Embedding layers to use our custom Functions."""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        n_linear = 0
        n_embedding = 0
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_name_to_module[name] = module

                if isinstance(module, nn.Linear):
                    module._original_forward = module.forward
                    module.forward = functools.partial(self._custom_linear_forward, module, idx)
                    n_linear += 1
                elif isinstance(module, nn.Embedding):
                    if module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
                        logger.warning(
                            f"Embedding {name} uses max_norm/scale_grad_by_freq/sparse, "
                            f"which the hook does not support; skipping"
                        )
                        continue
                    module._original_forward = module.forward
                    module.forward = functools.partial(self._custom_embedding_forward, module, idx)
                    n_embedding += 1
                else:
                    logger.warning(f"Layer {name} is neither nn.Linear nor nn.Embedding, skipping")
                    continue

        self.hooks_registered = True
        logger.info(
            f"Successfully wrapped {n_linear + n_embedding} layers "
            f"({n_linear} Linear, {n_embedding} Embedding)"
        )

    def _detect_tied_pairs(self) -> None:
        """Pair every hooked nn.Embedding with a hooked nn.Linear that shares its weight tensor."""
        embs = [(n, m) for n, m in self.layer_name_to_module.items() if isinstance(m, nn.Embedding)]
        lins = [(n, m) for n, m in self.layer_name_to_module.items() if isinstance(m, nn.Linear)]
        for en, em in embs:
            for ln, lm in lins:
                same = lm.weight is em.weight or (
                    lm.weight.shape == em.weight.shape and lm.weight.data_ptr() == em.weight.data_ptr()
                )
                if not same:
                    continue
                if lm.bias is not None:
                    raise NotImplementedError(f"tie_embeddings: tied output layer {ln} has a bias, which is not supported")
                ei, li = self.layer_name_to_idx[en], self.layer_name_to_idx[ln]
                self.tied_pairs[ei] = li
                self.tied_pairs[li] = ei
                self._tied_embedding_indices.add(ei)
                logger.info(f"tie_embeddings: {en} (layer {ei}) and {ln} (layer {li}) share one weight and form one group")
        if not self.tied_pairs:
            logger.warning("tie_embeddings=True but no hooked embedding/linear pair shares a weight; treatment unchanged")

    def _merge_tied_val_grads(self) -> None:
        """Replace both sites' target gradients by their sum (the shared parameter's gradient). Idempotent."""
        if not self.tie_embeddings or self._tied_merged or not self.tied_pairs:
            return
        for ei in self._tied_embedding_indices:
            li = self.tied_pairs[ei]
            ge, gl = self._val_grad_cache.get(ei), self._val_grad_cache.get(li)
            if ge is None or gl is None:
                continue
            merged = ge + gl.to(ge.dtype)
            self._val_grad_cache[ei] = merged
            self._val_grad_cache[li] = merged
        self._tied_merged = True

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """Replacement forward method with custom backward."""
        # Track hook calls for debugging
        self._hook_call_count += 1
        layer_name = self.layer_names[idx]
        self._hook_call_count_by_layer[layer_name] = self._hook_call_count_by_layer.get(layer_name, 0) + 1

        if not self.hooks_enabled:
            return module._original_forward(input)

        state = self.selection_state

        if isinstance(state, GlobalSubsetStateVerl):
            return GlobalSubsetLinearBackwardVerl.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, LayerWiseSubsetStateVerl):
            return LayerWiseSubsetLinearBackwardVerl.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self._capturing_val:
            return ValCaptureLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            return module._original_forward(input)

    def _custom_embedding_forward(self, module: nn.Embedding, idx: int, input_ids: Tensor) -> Tensor:
        """
        Replacement forward method for Embedding layers (lookup-based scoring, see packed_ops).

        Same dispatch as for Linear layers: GlobalSubset accumulates scores, LayerWiseSubset
        scores/selects/assembles the curated gradient, val capture stores the validation
        embedding gradient, otherwise the original forward runs.
        """
        self._hook_call_count += 1
        layer_name = self.layer_names[idx]
        self._hook_call_count_by_layer[layer_name] = self._hook_call_count_by_layer.get(layer_name, 0) + 1

        if not self.hooks_enabled:
            return module._original_forward(input_ids)

        padding_idx = module.padding_idx if module.padding_idx is not None else -1
        state = self.selection_state

        if isinstance(state, GlobalSubsetStateVerl):
            return GlobalSubsetEmbeddingBackwardVerl.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        elif isinstance(state, LayerWiseSubsetStateVerl):
            return LayerWiseSubsetEmbeddingBackwardVerl.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        elif self._capturing_val:
            return ValCaptureEmbeddingBackward.apply(
                input_ids, module.weight, self, idx, padding_idx
            )
        else:
            return module._original_forward(input_ids)

    def enable_hooks(self) -> None:
        """Enable hooks."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks."""
        self.hooks_enabled = False

    def get_hook_stats(self) -> dict:
        """Get hook call statistics for debugging."""
        return {
            'total_calls': self._hook_call_count,
            'unique_layers_called': len(self._hook_call_count_by_layer),
            'total_layers': len(self.layer_names),
            'calls_by_layer': dict(self._hook_call_count_by_layer),
        }

    def reset_hook_stats(self) -> None:
        """Reset hook call counters."""
        self._hook_call_count = 0
        self._hook_call_count_by_layer = {}

    # =========================================================================
    # Selection State Management
    # =========================================================================

    def setup_selection(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ) -> None:
        """Set up selection state."""
        if selection_method == "Regular" or selection_method == "NA":
            self.selection_state = None
            return

        self._merge_tied_val_grads()
        self._tied_stash.clear()
        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "GlobalSubset":
            self.selection_state = GlobalSubsetStateVerl(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode
            )
        elif selection_method == "LayerWiseSubset":
            self.selection_state = LayerWiseSubsetStateVerl(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode
            )
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")

        logger.debug(f"Set up {selection_method} state: {train_batch_size} train samples")

    def clear_selection(self) -> None:
        self._tied_stash.clear()
        """Clear selection state."""
        self.selection_state = None

    # =========================================================================
    # Token Count Tracking
    # =========================================================================

    def set_token_counts(
        self,
        labels: Tensor,
        train_batch_size: Optional[int] = None,
        attention_mask: Optional[Tensor] = None
    ) -> None:
        """
        Set per-sample item counts for gradient scaling (and cu_seqlens for packing).

        Item counts follow ``self.item_convention``: one per response under the
        ``seq-mean-*`` loss modes (a row counts as soon as it has a valid position),
        the response token count under ``token-mean``. The curated-update scale in
        the state is train_total / selected in these units (n/k for samples).

        cu_seqlens (where each sample's tokens live in the packed sequence) always
        come from the attention mask — that is a layout question, not a loss one.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored
            train_batch_size: Number of training samples
            attention_mask: Attention mask for cu_seqlens computation
        """
        valid_mask = (labels != -100)
        response_tokens = valid_mask.sum(dim=1)
        if self.item_convention == "sample":
            tokens_per_sample = valid_mask.any(dim=1).long()
        else:
            tokens_per_sample = response_tokens
        self.total_tokens = tokens_per_sample.sum()

        if train_batch_size is not None and self.selection_state is not None:
            train_tokens = tokens_per_sample[:train_batch_size]
            total_train_tokens = train_tokens.sum()

            # For packed sequences, use attention_mask for cu_seqlens
            if attention_mask is not None:
                packed_tokens = attention_mask[:train_batch_size].sum(dim=1)
            else:
                packed_tokens = response_tokens[:train_batch_size]

            self.selection_state.set_token_counts(
                train_tokens, total_train_tokens, packed_tokens
            )

    def clear_token_counts(self) -> None:
        """Clear token counts."""
        self.total_tokens = None

    # =========================================================================
    # Validation Gradient Management
    # =========================================================================

    def start_val_capture(self, use_factorized: bool = False) -> None:
        """
        Start validation gradient capture mode.

        Args:
            use_factorized: Ignored for RLVR (always uses full gradients)
        """
        self._capturing_val = True
        self._val_grad_cache.clear()
        self._tied_merged = False
        self._tied_stash.clear()
        logger.debug("Started validation gradient capture")

    def end_val_capture(self, val_total_tokens: Optional[int] = None) -> None:
        """End validation gradient capture mode."""
        self._capturing_val = False
        self._val_total_tokens = val_total_tokens
        logger.debug(f"Ended val capture, {len(self._val_grad_cache)} layers captured")

    def sync_val_grads(self) -> None:
        """
        Synchronize validation gradients across all ranks via all-reduce.

        In data parallel training, each rank computes validation gradients on
        different data shards. For consistent gradient-based selection, all ranks
        must have the same validation gradient (sum of all shards).

        Since validation loss is normalized by the GLOBAL item count (responses or
        tokens, synced before capture), the all-reduce SUM directly gives us the
        correct global gradient. No division by world_size is needed.

        This should be called after end_val_capture() and before using the
        validation gradients for selection.
        """
        if not dist.is_initialized() or _get_world_size() == 1:
            self._merge_tied_val_grads()
            return

        rank = _get_rank()
        world_size = _get_world_size()

        for layer_idx, val_grad in self._val_grad_cache.items():
            # All-reduce to sum gradients across all ranks
            # Since each rank's gradient is already normalized by the global item count,
            # the sum gives us the correct global gradient (no averaging needed)
            dist.all_reduce(val_grad, op=dist.ReduceOp.SUM)
        self._merge_tied_val_grads()

        logger.debug(f"[Rank {rank}] Synchronized {len(self._val_grad_cache)} validation gradient layers across {world_size} ranks")

    def get_val_grad(self, layer_idx: int) -> Optional[Tensor]:
        """Get cached validation gradient for a layer."""
        return self._val_grad_cache.get(layer_idx)

    def clear_val_cache(self) -> None:
        """Clear validation gradient cache."""
        self._val_grad_cache.clear()
        self._val_total_tokens = None

    # Alias for API compatibility
    def clear_val_buffer(self) -> None:
        """Alias for clear_val_cache() for API compatibility."""
        self.clear_val_cache()

    def has_val_grads(self) -> bool:
        """Check if validation gradients are cached."""
        return len(self._val_grad_cache) > 0

    def get_num_val_layers_captured(self) -> int:
        """Get number of layers with captured validation gradients."""
        return len(self._val_grad_cache)

    # Property for API compatibility with base library
    @property
    def _val_cache(self) -> "_ValCacheWrapper":
        """Compatibility wrapper for _val_cache access."""
        return _ValCacheWrapper(self)

    def setup_selection_with_stored_val(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ) -> None:
        """
        Set up selection state using pre-captured validation gradients.

        This is an alias for setup_selection() that validates we have stored val grads.
        """
        if not self.has_val_grads():
            raise RuntimeError(
                "No validation gradients captured. Call start_val_capture(), "
                "run forward/backward on validation data, then end_val_capture() first."
            )
        self.setup_selection(
            train_batch_size=train_batch_size,
            selection_method=selection_method,
            frac=frac,
            lr=lr,
            use_second_order=use_second_order,
            selection_mode=selection_mode
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def remove_hooks(self) -> None:
        """Restore original forward methods."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.hooks_registered = False
        logger.info("Restored original forward methods")


# =============================================================================
# Autograd Functions for RLVR/Verl
# =============================================================================

TIED_STASH_MAX_BYTES = int(float(os.environ.get("DRPT_TIED_STASH_MAX_GB", "16")) * 1e9)


def _per_sample_weight_grads(grad_output: Tensor, input: Tensor, is_packed: bool, sample_ids: Optional[Tensor], state) -> Tensor:
    """Per-sample weight gradients [B, O, I] of a linear layer (raw token sums, no item scaling)."""
    if is_packed:
        go, inp = grad_output.squeeze(0), input.squeeze(0)
        B = int(state.cu_seqlens.shape[0] - 1)
        nbytes = B * go.shape[1] * inp.shape[1] * go.element_size()
        if nbytes > TIED_STASH_MAX_BYTES:
            raise RuntimeError(
                f"tie_embeddings: per-sample gradients of the tied output layer need {nbytes/1e9:.1f} GB "
                f"(micro-batch {B}); lower ppo_micro_batch_size_per_gpu or raise DRPT_TIED_STASH_MAX_GB"
            )
        if sample_ids is None:
            from .packed_ops import compute_sample_ids_from_cu_seqlens
            sample_ids = compute_sample_ids_from_cu_seqlens(go.shape[0], state.cu_seqlens, go.device)
        out = torch.zeros(B, go.shape[1], inp.shape[1], device=go.device, dtype=go.dtype)
        for b in range(B):
            m = sample_ids == b
            if bool(m.any()):
                out[b] = go[m].T @ inp[m]
        return out
    if grad_output.dim() == 3:
        out = torch.einsum('bso,bsi->boi', grad_output, input)
    else:
        out = torch.einsum('bo,bi->boi', grad_output, input)
    if out.numel() * out.element_size() > TIED_STASH_MAX_BYTES:
        raise RuntimeError("tie_embeddings: per-sample gradients of the tied output layer exceed DRPT_TIED_STASH_MAX_GB")
    return out


class ValCaptureLinearBackward(Function):
    """Capture validation gradients during backward pass."""

    @staticmethod
    def forward(ctx, input, weight, bias, hook_manager, layer_idx):
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        # Store total validation gradient [O, I]
        with torch.no_grad():
            if grad_output.dim() == 3:
                val_grad = torch.einsum('bso,bsi->oi', grad_output, input)
            else:
                val_grad = torch.einsum('bo,bi->oi', grad_output, input)

            if layer_idx in hook_manager._val_grad_cache:
                hook_manager._val_grad_cache[layer_idx] += val_grad
            else:
                hook_manager._val_grad_cache[layer_idx] = val_grad

        return grad_input, None, None, None, None


class LayerWiseSubsetLinearBackwardVerl(Function):
    """LayerWiseSubset backward with packed sequence support."""

    @staticmethod
    def forward(ctx, input, weight, bias, hook_manager, layer_idx):
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        state: LayerWiseSubsetStateVerl = hook_manager.selection_state
        if state is None:
            return grad_input, None, None, None, None

        val_grad_total = hook_manager.get_val_grad(layer_idx)
        if val_grad_total is None:
            return grad_input, None, None, None, None

        with torch.no_grad():
            # Check if packed sequences
            is_packed = (
                state.cu_seqlens is not None and
                grad_output.shape[0] == 1 and
                grad_output.dim() == 3
            )

            sample_ids = None
            if is_packed:
                # Returns sample_ids for reuse in gradient computation
                scores, similarity, sample_ids = compute_scores_packed_vectorized(
                    grad_output, input, val_grad_total,
                    state.cu_seqlens, state.use_second_order
                )
            else:
                scores, similarity = compute_scores_and_similarity_standard(
                    grad_output, input, None, None, val_grad_total,
                    state.use_second_order
                )

            if hook_manager.tie_embeddings and layer_idx in hook_manager.tied_pairs:
                # Tied output layer: keep its site scores and per-sample weight gradients; the partner
                # embedding's backward (last in the pass) selects ONE subset for the shared weight from
                # the summed scores and assembles both contributions.
                hook_manager._tied_stash[layer_idx] = {
                    'scores': scores.detach(),
                    'per_sample_grad': _per_sample_weight_grads(grad_output, input, is_packed, sample_ids, state),
                }
                return grad_input, None, None, None, None
            # Select samples
            selected_indices = state.select_for_layer(layer_idx, scores, similarity)
            selected_indices = selected_indices.sort()[0]
            state._last_selected_indices = selected_indices
            state.num_selected = selected_indices.shape[0]

            if hasattr(state, '_layer_selections'):
                state._layer_selections.append((layer_idx, state.num_selected))

            # Compute gradients
            scale_factor = state._compute_scale_factor(selected_indices)

            if is_packed:
                grad_weight, grad_bias = compute_selected_gradients_packed_vectorized(
                    grad_output, input, selected_indices,
                    state.cu_seqlens, bias is not None, scale_factor,
                    sample_ids=sample_ids  # Reuse sample_ids computed above
                )
            else:
                grad_weight, grad_bias = compute_selected_gradients_standard(
                    grad_output, input, selected_indices, bias is not None, scale_factor
                )

        return grad_input, grad_weight, grad_bias, None, None


class GlobalSubsetLinearBackwardVerl(Function):
    """GlobalSubset backward with packed sequence support."""

    @staticmethod
    def forward(ctx, input, weight, bias, hook_manager, layer_idx):
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        state: GlobalSubsetStateVerl = hook_manager.selection_state
        if state is None:
            return grad_input, None, None, None, None

        val_grad_total = hook_manager.get_val_grad(layer_idx)
        if val_grad_total is None:
            return grad_input, None, None, None, None

        with torch.no_grad():
            # Check if packed sequences
            is_packed = (
                state.cu_seqlens is not None and
                grad_output.shape[0] == 1 and
                grad_output.dim() == 3
            )

            if is_packed:
                # sample_ids not needed for GlobalSubset (no gradient computation in this pass)
                scores, similarity, _ = compute_scores_packed_vectorized(
                    grad_output, input, val_grad_total,
                    state.cu_seqlens, state.use_second_order
                )
            else:
                scores, similarity = compute_scores_and_similarity_standard(
                    grad_output, input, None, None, val_grad_total,
                    state.use_second_order
                )

            # Accumulate scores
            state.accumulate_precomputed_scores(scores, similarity)

        return grad_input, None, None, None, None


# =============================================================================
# Autograd Functions for Embedding layers
# =============================================================================
#
# Same roles as the Linear Functions above. The embedding has no input gradient (token ids),
# so backward returns only the weight gradient; the per-sample gradient is never materialized
# (see the lookup-based helpers in packed_ops).

def _embedding_forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx):
    ctx.save_for_backward(input_ids)
    ctx.hook_manager_ref = weakref.ref(hook_manager)
    ctx.layer_idx = layer_idx
    ctx.padding_idx = padding_idx
    ctx.num_embeddings = weight.shape[0]
    ctx.weight_dtype = weight.dtype
    return F.embedding(input_ids, weight, padding_idx=padding_idx if padding_idx >= 0 else None)


def _embedding_is_packed(state, grad_output) -> bool:
    """Same packed-layout test as the Linear Functions: [1, total_nnz, D] with cu_seqlens set."""
    return state.cu_seqlens is not None and grad_output.shape[0] == 1 and grad_output.dim() == 3


class ValCaptureEmbeddingBackward(Function):
    """Capture the validation embedding gradient [V, D] (the target of the lookup-based scores)."""

    @staticmethod
    def forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx):
        return _embedding_forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        (input_ids,) = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        with torch.no_grad():
            val_grad = compute_embedding_val_gradient(
                grad_output, input_ids, ctx.num_embeddings, ctx.padding_idx
            )
            if layer_idx in hook_manager._val_grad_cache:
                hook_manager._val_grad_cache[layer_idx] += val_grad
            else:
                hook_manager._val_grad_cache[layer_idx] = val_grad

        return None, None, None, None, None


class LayerWiseSubsetEmbeddingBackwardVerl(Function):
    """LayerWiseSubset backward for Embedding layers with packed sequence support."""

    @staticmethod
    def forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx):
        return _embedding_forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        (input_ids,) = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        state: LayerWiseSubsetStateVerl = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        val_grad_weight = hook_manager.get_val_grad(layer_idx)
        if val_grad_weight is None:
            return None, None, None, None, None

        with torch.no_grad():
            is_packed = _embedding_is_packed(state, grad_output)

            sample_ids = None
            if is_packed:
                scores, sample_ids = compute_embedding_scores_packed(
                    grad_output, input_ids, val_grad_weight, state.cu_seqlens
                )
            else:
                scores = compute_embedding_scores_standard(grad_output, input_ids, val_grad_weight)

            stash = None
            if hook_manager.tie_embeddings and layer_idx in hook_manager.tied_pairs:
                stash = hook_manager._tied_stash.pop(hook_manager.tied_pairs[layer_idx], None)
                if stash is None:
                    logger.warning("tie_embeddings: the tied output layer left no stash; the embedding selects alone this step")
                else:
                    scores = scores + stash['scores'].to(scores.dtype)
            # Select samples (no similarity matrix for embeddings; greedy falls back to top-k)
            selected_indices = state.select_for_layer(layer_idx, scores, None)
            selected_indices = selected_indices.sort()[0]
            state._last_selected_indices = selected_indices
            state.num_selected = selected_indices.shape[0]

            if hasattr(state, '_layer_selections'):
                state._layer_selections.append((layer_idx, state.num_selected))

            # Curated gradient
            scale_factor = state._compute_scale_factor(selected_indices)

            if is_packed:
                grad_weight = compute_embedding_selected_gradient_packed(
                    grad_output, input_ids, selected_indices, state.cu_seqlens, scale_factor,
                    ctx.num_embeddings, ctx.padding_idx, sample_ids=sample_ids
                )
            else:
                grad_weight = compute_embedding_selected_gradient_standard(
                    grad_output, input_ids, selected_indices, scale_factor,
                    ctx.num_embeddings, ctx.padding_idx
                )
            grad_weight = grad_weight.to(ctx.weight_dtype)
            if stash is not None:
                partner = (stash['per_sample_grad'][selected_indices].sum(dim=0) * scale_factor).to(grad_weight.dtype)
                grad_weight = grad_weight + partner
                if hasattr(state, '_layer_selections'):
                    state._layer_selections.append((hook_manager.tied_pairs[layer_idx], state.num_selected))
        return None, grad_weight, None, None, None


class GlobalSubsetEmbeddingBackwardVerl(Function):
    """GlobalSubset backward for Embedding layers (score accumulation pass) with packed support."""

    @staticmethod
    def forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx):
        return _embedding_forward(ctx, input_ids, weight, hook_manager, layer_idx, padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        (input_ids,) = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hook_manager = ctx.hook_manager_ref()

        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected")

        state: GlobalSubsetStateVerl = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        val_grad_weight = hook_manager.get_val_grad(layer_idx)
        if val_grad_weight is None:
            return None, None, None, None, None

        with torch.no_grad():
            if _embedding_is_packed(state, grad_output):
                scores, _ = compute_embedding_scores_packed(
                    grad_output, input_ids, val_grad_weight, state.cu_seqlens
                )
            else:
                scores = compute_embedding_scores_standard(grad_output, input_ids, val_grad_weight)

            # Accumulate scores (no similarity contribution from embeddings)
            state.accumulate_precomputed_scores(scores, None)

        return None, None, None, None, None
