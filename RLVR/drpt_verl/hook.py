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
import weakref
import logging

from torch.autograd import Function

import torch.distributed as dist

from .selection_state import SelectionStateVerl, LayerwiseStateVerl, SubsetStateVerl
from .packed_ops import (
    compute_scores_packed_vectorized,
    compute_selected_gradients_packed_vectorized,
    compute_scores_and_similarity_standard,
    compute_selected_gradients_standard,
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
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (only Linear layers supported)
            device: Device for tensors
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device

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

        logger.info(f"Initialized GradientHookVerl with {len(layer_names)} layers")

    def _register_hooks(self) -> None:
        """Monkey-patch Linear layers to use our custom Function."""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_name_to_module[name] = module

                if not isinstance(module, nn.Linear):
                    logger.warning(f"Layer {name} is not nn.Linear, skipping")
                    continue

                module._original_forward = module.forward
                wrapped_forward = functools.partial(self._custom_linear_forward, module, idx)
                module.forward = wrapped_forward

        self.hooks_registered = True
        logger.info(f"Successfully wrapped {len(self.layer_names)} layers")

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """Replacement forward method with custom backward."""
        # Track hook calls for debugging
        self._hook_call_count += 1
        layer_name = self.layer_names[idx]
        self._hook_call_count_by_layer[layer_name] = self._hook_call_count_by_layer.get(layer_name, 0) + 1

        if not self.hooks_enabled:
            return module._original_forward(input)

        state = self.selection_state

        if isinstance(state, SubsetStateVerl):
            return SubsetLinearBackwardVerl.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, LayerwiseStateVerl):
            return LayerwiseLinearBackwardVerl.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self._capturing_val:
            return ValCaptureLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            return module._original_forward(input)

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

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "Subset":
            self.selection_state = SubsetStateVerl(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode
            )
        elif selection_method == "Layerwise":
            self.selection_state = LayerwiseStateVerl(
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
        Set token counts for proper gradient scaling.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored
            train_batch_size: Number of training samples
            attention_mask: Attention mask for cu_seqlens computation
        """
        valid_mask = (labels != -100)
        tokens_per_sample = valid_mask.sum(dim=1)
        self.total_tokens = tokens_per_sample.sum()

        if train_batch_size is not None and self.selection_state is not None:
            train_tokens = tokens_per_sample[:train_batch_size]
            total_train_tokens = train_tokens.sum()

            # For packed sequences, use attention_mask for cu_seqlens
            if attention_mask is not None:
                packed_tokens = attention_mask[:train_batch_size].sum(dim=1)
            else:
                packed_tokens = train_tokens

            self.selection_state.set_token_counts(
                train_tokens, total_train_tokens, self.total_tokens, packed_tokens
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

        Since validation loss is normalized by GLOBAL token count (synced before
        capture), the all-reduce SUM directly gives us the correct global gradient.
        No division by world_size is needed.

        This should be called after end_val_capture() and before using the
        validation gradients for selection.
        """
        if not dist.is_initialized() or _get_world_size() == 1:
            return

        rank = _get_rank()
        world_size = _get_world_size()

        for layer_idx, val_grad in self._val_grad_cache.items():
            # All-reduce to sum gradients across all ranks
            # Since each rank's gradient is already normalized by global token count,
            # the sum gives us the correct global gradient (no averaging needed)
            dist.all_reduce(val_grad, op=dist.ReduceOp.SUM)

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


class LayerwiseLinearBackwardVerl(Function):
    """Layerwise backward with packed sequence support."""

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

        state: LayerwiseStateVerl = hook_manager.selection_state
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

            # Select samples
            selected_indices = state._select_indices(scores, similarity)
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


class SubsetLinearBackwardVerl(Function):
    """Subset backward with packed sequence support."""

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

        state: SubsetStateVerl = hook_manager.selection_state
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
                # sample_ids not needed for Subset (no gradient computation in this pass)
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
            state.accumulate_precomputed_scores(scores, similarity, None)

        return grad_input, None, None, None, None
