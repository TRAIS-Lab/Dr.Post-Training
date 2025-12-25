"""
Hook manager for on-the-fly gradient streaming.

This implementation uses monkey-patching with custom autograd Functions to:
1. Prevent full gradient materialization
2. Compute selection scores layer-by-layer during backward (streaming)
3. Reduce gradients on-the-fly to avoid OOM
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
import functools
import logging

from .compressor import Compressor
from .utils import greedy_selection, topk_selection, negative_filtering

logger = logging.getLogger(__name__)

# Global registry: maps a unique ID to hook manager
# CRITICAL: Used to avoid storing hook_manager in autograd context, which would cause memory leaks
_HOOK_MANAGER_REGISTRY: Dict[int, GradientHook] = {}


class StreamingState:
    """
    State manager for on-the-fly gradient streaming during backward pass.

    Uses per-layer selection: each layer independently selects samples based on
    that layer's gradient alignment with validation gradients. This enables
    unified data selection and model update in a single gradient stream.
    """

    def __init__(
        self,
        train_batch_size: int,
        num_layers: int,
        selection_method: str,
        frac: float,
        lr: float,
        device: str = 'cpu',
        compute_scores_only: bool = False,
        dtype: torch.dtype = torch.float32,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ):
        """
        Initialize streaming state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers (for tracking backward progress)
            selection_method: Selection method (Streaming, Regular, GREATS)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling (unused in per-layer mode)
            device: Device for tensors
            compute_scores_only: If True, only maintain running scores without aggregating gradients.
                                This is used for Streaming without MeSO (Case 2), where we compute
                                scores on-the-fly and then do a second forward/backward with full gradients.
            dtype: Data type for score tensors (should match model dtype for efficiency)
            use_second_order: If True, compute similarity matrix and use greedy selection with
                            second-order interactions. If False, use simple top-k selection
                            which is ~200x faster. Default False for efficiency.
            selection_mode: How to select samples based on scores:
                           - "topk": Select top frac samples by score (SFT style)
                           - "filtering": Keep all positive + drop bottom frac of negative (RLHF style)
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.selection_method = selection_method
        self.frac = frac
        self.lr = lr
        self.device = device
        self.compute_scores_only = compute_scores_only
        self.dtype = dtype
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode

        # Number of samples to select (for top-k mode)
        self.num_selected = max(1, int(train_batch_size * frac))

        # Token count tracking for proper gradient scaling
        # These are set by GradientHook.set_token_counts() before backward
        self.tokens_per_sample: Optional[Tensor] = None  # [train_batch_size]
        self.total_train_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None  # Total in batch (train + val for SFT)

        # Running scores only needed for compute_scores_only mode (Streaming without MeSO)
        # For per-layer selection with MeSO, we compute scores directly in backward()
        self.grad_dot_scores = None
        self.similarity_matrix = None
        if compute_scores_only:
            self.grad_dot_scores = torch.zeros(train_batch_size, device=device, dtype=dtype)
            if use_second_order:
                # Initialize similarity matrix for both Streaming and GREATS when second-order is enabled
                self.similarity_matrix = torch.zeros(train_batch_size, train_batch_size, device=device, dtype=dtype)

    def _update_scores(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
    ):
        """
        Update running scores with compressed gradients from current layer.

        Only used when compute_scores_only=True (global selection with compression).

        Args:
            train_grads: Per-sample compressed gradients [train_batch_size, k_l]
            val_grad: Mean compressed validation gradient [k_l]
        """
        if self.grad_dot_scores is None:
            return  # Per-layer mode: scores computed directly in backward()

        # Cast to score dtype if needed
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Streaming uses gradient alignment (dot product with validation gradient)
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, out=self.grad_dot_scores)

        if self.similarity_matrix is not None:
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), out=self.similarity_matrix)

    def _update_scores_direct(self, scores: Tensor, similarity: Tensor = None):
        """
        Update running scores directly with pre-computed score values.

        Used for global selection without compression (full gradient mode).

        Args:
            scores: Pre-computed scores [train_batch_size]
            similarity: Optional pairwise similarity matrix [train_batch_size, train_batch_size]
                       Required when use_second_order=True
        """
        if self.grad_dot_scores is None:
            return

        if scores.dtype != self.dtype:
            scores = scores.to(self.dtype)

        self.grad_dot_scores += scores

        # Update similarity matrix for second-order selection
        if self.similarity_matrix is not None and similarity is not None:
            if similarity.dtype != self.dtype:
                similarity = similarity.to(self.dtype)
            self.similarity_matrix += similarity

    def get_selected_indices(
        self,
        scores: Tensor | None = None,
        similarity: Tensor | None = None
    ) -> Tensor:
        """
        Select sample indices based on scores.

        Unified method for both GREATS (global selection) and Streaming (per-layer selection).

        Args:
            scores: Per-sample scores [train_batch_size].
                   If None, uses accumulated self.grad_dot_scores (GREATS mode).
            similarity: Per-sample similarity matrix [train_batch_size, train_batch_size]
                       for second-order selection. If None and using accumulated scores,
                       uses self.similarity_matrix.

        Returns:
            Selected sample indices tensor
        """
        # Determine whether using accumulated (GREATS) or explicit (Streaming) scores
        if scores is None:
            # GREATS mode: use accumulated scores with lr scaling
            if self.grad_dot_scores is None:
                raise RuntimeError("No accumulated scores and no explicit scores provided")
            scores = self.grad_dot_scores * self.lr
            # Use accumulated similarity matrix with lr^2 scaling
            if similarity is None and self.similarity_matrix is not None:
                similarity = self.similarity_matrix * (self.lr ** 2)
        else:
            # Streaming mode: use explicit scores with lr scaling for consistency with GREATS
            # Both modes should treat scores similarly: score represents lr * g_train @ g_val
            # and similarity represents lr^2 * g_train @ g_train
            scores = scores * self.lr
            if similarity is not None:
                similarity = similarity * (self.lr ** 2)

        # Common selection logic
        if self.selection_method in ('Streaming', 'GREATS'):
            if self.selection_mode == "filtering":
                # RLHF filtering mode: keep positive samples, drop bottom frac of negative
                selected_indices = negative_filtering(scores, self.frac)
            elif self.use_second_order and similarity is not None:
                # Second-order selection using greedy algorithm
                selected_indices = greedy_selection(scores, similarity, self.num_selected)
            else:
                # First-order selection using top-k
                selected_indices = topk_selection(scores, self.num_selected)
            self.num_selected = len(selected_indices)
        else:
            # Regular mode: no selection
            selected_indices = torch.arange(self.train_batch_size, device=scores.device)

        return selected_indices


class StreamingLinearBackward(Function):
    """
    Custom autograd Function for gradient streaming with on-the-fly data selection.

    Supports two modes based on whether compression is enabled:
    1. Compressed mode (compressor exists): Use compressed gradients for scoring,
       store compressed_grads for MeSO optimizer
    2. Full gradient mode (no compressor): Compute scores directly from full gradients,
       return grad_weight/grad_bias for standard optimizer
    """

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor | None, hook_manager_id: int, layer_idx: int) -> Tensor:
        """
        Forward pass: standard linear transformation.

        CRITICAL: We store hook_manager_id (an int) instead of the hook_manager object.
        Storing the object would keep it in the autograd graph, causing memory leaks.
        """
        # Cast input to weight dtype for computation (handles mixed precision)
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input

        # MEMORY OPTIMIZATION: Use PyTorch's built-in save_for_backward
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_id = hook_manager_id
        ctx.layer_idx = layer_idx

        # Standard forward pass (same as nn.Linear)
        output = F.linear(input_compute, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor | None, Tensor | None, None, None]:
        """
        Backward pass with on-the-fly data selection.

        Returns:
            grad_input: Gradient w.r.t. input (always computed)
            grad_weight: Gradient w.r.t. weight (only in full gradient mode with selection)
            grad_bias: Gradient w.r.t. bias (only in full gradient mode with selection)
            None, None: Placeholders for hook_manager_id and layer_idx
        """
        input, weight, bias = ctx.saved_tensors
        hook_manager_id: int = ctx.hook_manager_id
        layer_idx: int = ctx.layer_idx

        # Lookup hook manager from registry
        hook_manager: Optional[GradientHook] = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            raise RuntimeError(f"Hook manager {hook_manager_id} not found in registry")

        # Cast input to match grad_output dtype if needed
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Compute grad_input (always needed for backprop to previous layers)
        grad_input = grad_output @ weight.to(grad_output.dtype)

        # Initialize grad_weight and grad_bias (only set in full gradient mode)
        grad_weight = None
        grad_bias = None

        # Check if we have compression
        compressor = hook_manager.compressors[layer_idx]
        selection_state = hook_manager.selection_state

        with torch.no_grad():
            # Check for RLHF validation capture mode
            capture_val_mode = hook_manager.capture_val_mode
            use_stored_val = (
                selection_state is not None and
                hasattr(selection_state, '_use_stored_val') and
                selection_state._use_stored_val
            )

            if compressor is not None:
                # === COMPRESSED MODE ===
                # Use compressed gradients for scoring and store for MeSO optimizer
                #
                # SCALING PRINCIPLE:
                # - Baseline mode (no selection_state): Don't scale grad_output. The loss already
                #   has 1/num_tokens scaling. Just compress and sum to get token-averaged gradient.
                # - Selection mode (selection_state exists): Scale by total_tokens to recover
                #   per-sample gradients for scoring, then divide by selected_tokens at the end.
                batch_size = grad_output.shape[0]

                # Prepare input for compressor: augment with ones column for bias
                # This combines weight and bias gradients into a single compressed tensor:
                #   grad_W = grad_output^T @ input        -> weight gradient
                #   grad_b = grad_output^T @ ones = sum(grad_output)  -> bias gradient
                # The MeSO optimizer extracts weight/bias portions on decompression.
                input_for_compressor = input
                if bias is not None:
                    if input.dim() == 3:
                        _, seq_length, in_features = input.shape
                        ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
                    else:
                        ones = torch.ones(batch_size, 1, device=input.device, dtype=input.dtype)
                    input_for_compressor = torch.cat([input, ones], dim=-1)

                # Don't scale grad_output - preserve 1/num_tokens scaling from loss
                # This matches baseline mode and avoids scaling mismatch issues
                grad_output_for_compress = grad_output

                # Compute compressed gradients
                compressed_grad = compressor.forward((grad_output_for_compress, input_for_compressor))

                if capture_val_mode:
                    # RLHF val capture: Accumulate per-sample average for this layer
                    # mean() directly gives per-sample average since we already scaled
                    mean_grad = compressed_grad.mean(dim=0)  # [k_l]
                    if hook_manager.val_grad_buffer[layer_idx] is None:
                        hook_manager.val_grad_buffer[layer_idx] = mean_grad
                    else:
                        hook_manager.val_grad_buffer[layer_idx] = hook_manager.val_grad_buffer[layer_idx] + mean_grad

                elif selection_state is not None:
                    if use_stored_val:
                        # RLHF selection: Use stored val gradients instead of splitting
                        train_grads = compressed_grad  # All samples are train (per-sample magnitude)
                        val_grad = hook_manager.val_grad_buffer[layer_idx]  # Pre-captured (per-sample avg)

                        if val_grad is not None:
                            # Update running scores (per-sample @ per-sample_avg = correct alignment)
                            selection_state._update_scores(train_grads, val_grad)

                            if not selection_state.compute_scores_only:
                                # Per-layer selection and reduce (parallel structure to full gradient path)
                                # Step 1: Compute scores
                                scores = train_grads @ val_grad

                                # Step 2: Compute similarity for second-order selection if enabled
                                similarity = None
                                if selection_state.use_second_order:
                                    similarity = train_grads @ train_grads.T

                                # Step 3: Get selected indices (shared with full gradient path)
                                selected_indices = selection_state.get_selected_indices(scores, similarity)
                                num_selected = selected_indices.shape[0]

                                # Step 4: Sum selected samples (or zeros if none selected)
                                if num_selected == 0:
                                    reduced_grad = torch.zeros(1, train_grads.shape[1], device=train_grads.device, dtype=train_grads.dtype)
                                else:
                                    selected_grads = train_grads[selected_indices]
                                    reduced_grad = selected_grads.sum(dim=0, keepdim=True)

                                    # Step 5: Apply token-based scaling (same formula as full gradient path)
                                    if selection_state.tokens_per_sample is not None and selection_state.total_tokens is not None:
                                        selected_tokens = selection_state.tokens_per_sample[selected_indices].sum().item()
                                        if selected_tokens > 0:
                                            scale_factor = selection_state.total_tokens / selected_tokens
                                        else:
                                            scale_factor = 1.0
                                        reduced_grad = reduced_grad * scale_factor

                                hook_manager._store_compressed_grad(layer_idx, reduced_grad)
                        else:
                            # No val grad for this layer, just average
                            # Already per-sample, mean() gives correct per-sample average
                            hook_manager._store_compressed_grad(layer_idx, compressed_grad.mean(dim=0, keepdim=True))
                    else:
                        # SFT mode: Split merged batch into train/val
                        train_grads = compressed_grad[:selection_state.train_batch_size]  # per-sample
                        val_grads = compressed_grad[selection_state.train_batch_size:]    # per-sample
                        val_grad = val_grads.mean(dim=0)  # per-sample average

                        # Update running scores (per-sample @ per-sample_avg = correct alignment)
                        selection_state._update_scores(train_grads, val_grad)

                        if not selection_state.compute_scores_only:
                            # Per-layer selection and reduce (parallel structure to full gradient path)
                            # Step 1: Compute scores
                            scores = train_grads @ val_grad

                            # Step 2: Compute similarity for second-order selection if enabled
                            similarity = None
                            if selection_state.use_second_order:
                                similarity = train_grads @ train_grads.T

                            # Step 3: Get selected indices (shared with full gradient path)
                            selected_indices = selection_state.get_selected_indices(scores, similarity)
                            num_selected = selected_indices.shape[0]

                            # Step 4: Sum selected samples (or zeros if none selected)
                            if num_selected == 0:
                                reduced_grad = torch.zeros(1, train_grads.shape[1], device=train_grads.device, dtype=train_grads.dtype)
                            else:
                                selected_grads = train_grads[selected_indices]
                                reduced_grad = selected_grads.sum(dim=0, keepdim=True)

                                # Step 5: Apply token-based scaling (same formula as full gradient path)
                                if selection_state.tokens_per_sample is not None and selection_state.total_tokens is not None:
                                    selected_tokens = selection_state.tokens_per_sample[selected_indices].sum().item()
                                    if selected_tokens > 0:
                                        scale_factor = selection_state.total_tokens / selected_tokens
                                    else:
                                        scale_factor = 1.0
                                    reduced_grad = reduced_grad * scale_factor

                            hook_manager._store_compressed_grad(layer_idx, reduced_grad)
                else:
                    # Baseline mode (no selection): sum compressed gradients
                    # grad_output already has 1/num_tokens scaling from loss, so sum
                    # preserves the token-averaged gradient semantics
                    compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
                    hook_manager._store_compressed_grad(layer_idx, compressed_grad)

            else:
                # === FULL GRADIENT MODE ===
                # No compression: compute scores directly, return full gradients for standard optimizer

                if capture_val_mode:
                    # RLHF val capture: Accumulate mean full gradient for this layer
                    # Use += to properly accumulate across mini-batches
                    # The trainer scales loss by 1/full_batch_size, so gradients are pre-scaled
                    if grad_output.dim() == 3:
                        val_grad_full = torch.einsum('bso,bsi->oi', grad_output, input) / grad_output.shape[0]
                    else:
                        val_grad_full = torch.einsum('bo,bi->oi', grad_output, input) / grad_output.shape[0]
                    if hook_manager.val_grad_buffer[layer_idx] is None:
                        hook_manager.val_grad_buffer[layer_idx] = val_grad_full
                    else:
                        hook_manager.val_grad_buffer[layer_idx] = hook_manager.val_grad_buffer[layer_idx] + val_grad_full

                elif selection_state is not None:
                    if use_stored_val:
                        # RLHF selection: Use stored val gradients
                        train_grad_output = grad_output
                        train_input = input
                        train_batch_size = selection_state.train_batch_size
                        val_grad_full = hook_manager.val_grad_buffer[layer_idx]  # [O, I]

                        if val_grad_full is not None:
                            # Compute per-sample scores using ghost inner product
                            similarity = None
                            if train_grad_output.dim() == 3:
                                temp = train_input @ val_grad_full.T
                                scores = (train_grad_output * temp).sum(dim=(1, 2))

                                if selection_state.use_second_order:
                                    contracted = torch.bmm(
                                        train_grad_output.permute(0, 2, 1),
                                        train_input
                                    ).flatten(start_dim=1)
                                    similarity = torch.matmul(contracted, contracted.T)
                            else:
                                temp = train_input @ val_grad_full.T
                                scores = (train_grad_output * temp).sum(dim=1)

                                if selection_state.use_second_order:
                                    dot_g = torch.matmul(train_grad_output, train_grad_output.T)
                                    dot_x = torch.matmul(train_input, train_input.T)
                                    similarity = dot_g * dot_x

                            selection_state._update_scores_direct(scores, similarity)

                            if not selection_state.compute_scores_only:
                                # Use centralized selection logic
                                selected_indices = selection_state.get_selected_indices(scores, similarity)
                                num_selected = selected_indices.shape[0]

                                # Handle empty selection: set gradients to zero (no update)
                                if num_selected == 0:
                                    grad_weight = torch.zeros_like(weight)
                                    if bias is not None:
                                        grad_bias = torch.zeros_like(bias)
                                else:
                                    selected_grad_output = train_grad_output[selected_indices]
                                    selected_input = train_input[selected_indices]

                                    # GRADIENT SCALING FIX: Use token-based scaling
                                    # Same principle as SFT mode - scale by total_tokens/selected_tokens
                                    if selection_state.tokens_per_sample is not None and selection_state.total_tokens is not None:
                                        selected_tokens = selection_state.tokens_per_sample[selected_indices].sum().item()
                                        if selected_tokens > 0:
                                            scale_factor = selection_state.total_tokens / selected_tokens
                                        else:
                                            scale_factor = 1.0
                                    else:
                                        # Fall back to sample-based scaling if token counts not available
                                        scale_factor = train_batch_size / num_selected

                                    if selected_grad_output.dim() == 3:
                                        grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
                                        if bias is not None:
                                            grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor
                                    else:
                                        grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
                                        if bias is not None:
                                            grad_bias = selected_grad_output.sum(dim=0) * scale_factor
                    else:
                        # SFT mode: Split merged batch
                        train_batch_size = selection_state.train_batch_size
                        train_grad_output = grad_output[:train_batch_size]
                        val_grad_output = grad_output[train_batch_size:]
                        train_input = input[:train_batch_size]
                        val_input = input[train_batch_size:]

                        # Compute validation gradient [out_features, in_features]
                        if val_grad_output.dim() == 3:
                            val_grad_full = torch.einsum('vso,vsi->oi', val_grad_output, val_input)
                        else:
                            val_grad_full = torch.einsum('vo,vi->oi', val_grad_output, val_input)
                        val_grad_full = val_grad_full / val_grad_output.shape[0]

                        # Compute per-sample scores using ghost inner product
                        similarity = None
                        if train_grad_output.dim() == 3:
                            temp = train_input @ val_grad_full.T
                            scores = (train_grad_output * temp).sum(dim=(1, 2))

                            if selection_state.use_second_order:
                                contracted = torch.bmm(
                                    train_grad_output.permute(0, 2, 1),
                                    train_input
                                ).flatten(start_dim=1)
                                similarity = torch.matmul(contracted, contracted.T)
                        else:
                            temp = train_input @ val_grad_full.T
                            scores = (train_grad_output * temp).sum(dim=1)

                            if selection_state.use_second_order:
                                dot_g = torch.matmul(train_grad_output, train_grad_output.T)
                                dot_x = torch.matmul(train_input, train_input.T)
                                similarity = dot_g * dot_x

                        selection_state._update_scores_direct(scores, similarity)

                        if not selection_state.compute_scores_only:
                            # Use centralized selection logic
                            selected_indices = selection_state.get_selected_indices(scores, similarity)
                            num_selected = selected_indices.shape[0]

                            # Handle empty selection: set gradients to zero (no update)
                            if num_selected == 0:
                                grad_weight = torch.zeros_like(weight)
                                if bias is not None:
                                    grad_bias = torch.zeros_like(bias)
                            else:
                                selected_grad_output = train_grad_output[selected_indices]
                                selected_input = train_input[selected_indices]

                                # GRADIENT SCALING FIX: Use token-based scaling
                                # Loss is averaged over total_tokens (merged batch), but we want
                                # gradient as if training on selected train samples' tokens only.
                                # grad_output has 1/total_tokens scaling, so we multiply by
                                # total_tokens/selected_tokens to get 1/selected_tokens scaling.
                                if selection_state.tokens_per_sample is not None and selection_state.total_tokens is not None:
                                    selected_tokens = selection_state.tokens_per_sample[selected_indices].sum().item()
                                    if selected_tokens > 0:
                                        scale_factor = selection_state.total_tokens / selected_tokens
                                    else:
                                        scale_factor = 1.0
                                else:
                                    # Fall back to sample-based scaling if token counts not available
                                    scale_factor = train_batch_size / num_selected

                                if selected_grad_output.dim() == 3:
                                    grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
                                    if bias is not None:
                                        grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor
                                else:
                                    grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
                                    if bias is not None:
                                        grad_bias = selected_grad_output.sum(dim=0) * scale_factor

                # else: No selection and no compression = baseline, let PyTorch handle gradients normally

        return grad_input, grad_weight, grad_bias, None, None


class GradientHook:
    """
    Hook manager for gradient streaming with on-the-fly data selection.

    Features:
    1. StreamingState for managing streaming during backward pass
    2. On-the-fly score computation and gradient aggregation
    3. Memory-efficient processing without materializing all per-sample grads
    4. Unified data selection and model update in a single gradient stream
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cpu',
        register_hooks: bool = True
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (only Linear layers supported)
            device: Device for synchronization
            register_hooks: Whether to register hooks immediately
        """
        self.model: nn.Module = model
        self.layer_names: List[str] = layer_names
        self.device: str = device

        # Create mapping from layer name to index
        self.layer_name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(layer_names)}

        # Create mapping from layer name to module
        self.layer_name_to_module: Dict[str, nn.Module] = {}

        # Centralized storage arrays
        self.forward_hooks: List[Optional[Any]] = [None] * len(layer_names)
        # Note: Compressed gradients are now stored directly on weight parameters as
        # weight._compressed_grad instead of in a separate list. See _store_compressed_grad()
        # and _get_compressed_grad() for details.

        # Unified compressors
        self.compressors: List[Optional[Compressor]] = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True

        # Streaming state (set by trainer before forward/backward)
        self.selection_state: Optional[StreamingState] = None

        # RLHF-specific: Validation gradient buffer for separate val/train passes
        # In RLHF, val and train use different loss functions, so we can't merge batches
        # Instead, we capture val gradients first, then use them for selection during train
        self.val_grad_buffer: List[Optional[Tensor]] = [None] * len(layer_names)
        self.capture_val_mode: bool = False  # True when capturing validation gradients

        # Token count tracking for proper gradient scaling
        # The loss is averaged over valid tokens (where labels != -100), not samples.
        # To properly scale gradients, we need to track:
        # - total_tokens: total valid tokens in the batch
        # - tokens_per_sample: valid tokens per sample [batch_size]
        self.total_tokens: Optional[int] = None
        self.tokens_per_sample: Optional[Tensor] = None

        # Register in global registry
        self._hook_manager_id: int = id(self)
        _HOOK_MANAGER_REGISTRY[self._hook_manager_id] = self

        # Register hooks if requested
        if register_hooks:
            self._register_hooks()

        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    def _register_hooks(self):
        """
        Monkey-patch Linear layers to use our custom Function.
        """
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Cache the module
                self.layer_name_to_module[name] = module

                # Only support Linear layers
                if not isinstance(module, nn.Linear):
                    logger.warning(f"Layer {name} is not nn.Linear, skipping")
                    continue

                # Save original forward method
                module._original_forward = module.forward

                # Create wrapped forward that uses our custom Function
                wrapped_forward = functools.partial(
                    self._custom_linear_forward, module, idx
                )

                # Replace the forward method
                module.forward = wrapped_forward

        self.hooks_registered = True
        logger.info(f"Successfully wrapped {len(self.layer_names)} layers")

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """
        Replacement forward method that uses our custom streaming Function.
        """
        if self.hooks_enabled:
            return StreamingLinearBackward.apply(
                input, module.weight, module.bias, self._hook_manager_id, idx
            )
        else:
            if hasattr(module, '_original_forward'):
                return module._original_forward(input)
            else:
                return F.linear(input, module.weight, module.bias)

    def set_compressors(self, compressors: List[Compressor]) -> None:
        """Set unified compressor objects for each layer."""
        self.compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute compressed gradients."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    def setup_selection(
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
        Set up streaming state for on-the-fly gradient streaming.

        This should be called by the trainer before each forward/backward pass
        that requires gradient streaming with data selection.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method (Streaming, Regular, GREATS)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores without aggregating gradients.
                                Used for Streaming without MeSO (Case 2).
            use_second_order: If True, compute similarity matrix and use greedy selection
                            with second-order interactions. If False (default), use simple
                            top-k selection which is ~200x faster.
            selection_mode: How to select samples based on scores:
                           - "topk": Select top frac samples by score (SFT style)
                           - "filtering": Keep all positive + drop bottom frac of negative (RLHF style)
        """
        # Infer dtype from model parameters (use first parameter's dtype)
        dtype = next(self.model.parameters()).dtype

        num_layers = len(self.layer_names)
        self.selection_state = StreamingState(
            train_batch_size=train_batch_size,
            num_layers=num_layers,
            selection_method=selection_method,
            frac=frac,
            lr=lr,
            device=self.device,
            compute_scores_only=compute_scores_only,
            dtype=dtype,
            use_second_order=use_second_order,
            selection_mode=selection_mode
        )
        logger.debug(f"Set up streaming state: {train_batch_size} train, scores_only={compute_scores_only}, use_second_order={use_second_order}, selection_mode={selection_mode}, frac={frac}, dtype={dtype}")

    def clear_selection(self) -> None:
        """Clear selection state after forward/backward."""
        self.selection_state = None

    def set_token_counts(self, labels: Tensor, train_batch_size: Optional[int] = None) -> None:
        """
        Set token counts for proper gradient scaling.

        The loss is averaged over valid tokens (where labels != -100), not samples.
        This method computes and stores the token counts needed for proper scaling.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored positions
            train_batch_size: If provided, only count tokens for first train_batch_size samples
                            (for SFT mode where batch is merged train+val)
        """
        # Count valid tokens per sample
        valid_mask = (labels != -100)  # [batch_size, seq_length]
        tokens_per_sample = valid_mask.sum(dim=1)  # [batch_size]

        # Total tokens in the batch (used for gradient scaling)
        self.total_tokens = tokens_per_sample.sum().item()
        self.tokens_per_sample = tokens_per_sample

        # If train_batch_size is provided, also track train-specific counts
        if train_batch_size is not None and self.selection_state is not None:
            train_tokens = tokens_per_sample[:train_batch_size]
            self.selection_state.tokens_per_sample = train_tokens
            self.selection_state.total_train_tokens = train_tokens.sum().item()
            self.selection_state.total_tokens = self.total_tokens

        logger.debug(f"Set token counts: total={self.total_tokens}, per_sample={tokens_per_sample.tolist()}")

    def clear_token_counts(self) -> None:
        """Clear token counts after forward/backward."""
        self.total_tokens = None
        self.tokens_per_sample = None

    # ========================================
    # RLHF-specific methods for separate val/train passes
    # ========================================

    def start_val_capture(self) -> None:
        """
        Start capturing validation gradients (RLHF mode).

        In RLHF, validation and training use different loss functions:
        - Validation: sequence-level reward-weighted log probs
        - Training: PPO loss (policy + value + KL)

        This mode captures compressed gradients during backward and stores them
        in val_grad_buffer for later use during training selection.
        """
        self.capture_val_mode = True
        # Clear the buffer for fresh capture
        self.val_grad_buffer = [None] * len(self.layer_names)
        logger.debug("Started validation gradient capture mode")

    def end_val_capture(self) -> None:
        """
        End validation gradient capture mode.

        After calling this, the val_grad_buffer contains the mean compressed
        validation gradients per layer, ready for use in selection.
        """
        self.capture_val_mode = False
        logger.debug(f"Ended validation gradient capture, captured {sum(1 for g in self.val_grad_buffer if g is not None)} layers")

    def clear_val_buffer(self) -> None:
        """Clear the validation gradient buffer."""
        self.val_grad_buffer = [None] * len(self.layer_names)
        self.capture_val_mode = False

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
        Set up selection state using pre-captured validation gradients (RLHF mode).

        This is like setup_selection but uses the stored val_grad_buffer instead
        of expecting merged batches. The backward pass will only process training
        samples and compute scores against stored validation gradients.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method (Streaming, GREATS)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (for GREATS two-pass)
            use_second_order: If True, use greedy selection with second-order
            selection_mode: How to select samples based on scores:
                           - "topk": Select top frac samples by score (SFT style)
                           - "filtering": Keep all positive + drop bottom frac of negative (RLHF style)
        """
        # Verify we have captured validation gradients
        num_captured = sum(1 for g in self.val_grad_buffer if g is not None)
        if num_captured == 0:
            raise RuntimeError(
                "No validation gradients captured. Call start_val_capture(), "
                "run forward/backward on validation data, then end_val_capture() first."
            )

        # Infer dtype from model parameters
        dtype = next(self.model.parameters()).dtype

        num_layers = len(self.layer_names)
        self.selection_state = StreamingState(
            train_batch_size=train_batch_size,
            num_layers=num_layers,
            selection_method=selection_method,
            frac=frac,
            lr=lr,
            device=self.device,
            compute_scores_only=compute_scores_only,
            dtype=dtype,
            use_second_order=use_second_order,
            selection_mode=selection_mode
        )
        # Mark that we're using stored validation gradients (train_batch_size = actual batch size)
        self.selection_state._use_stored_val = True
        logger.debug(
            f"Set up selection with stored val gradients: {train_batch_size} train samples, "
            f"{num_captured} layers with val gradients, selection_mode={selection_mode}, frac={frac}"
        )

    def _get_layer_name_from_idx(self, layer_idx: int) -> str:
        """Get layer name from layer index."""
        return self.layer_names[layer_idx]

    def _get_module_from_idx(self, layer_idx: int) -> nn.Module:
        """Get module from layer index."""
        layer_name = self._get_layer_name_from_idx(layer_idx)
        return self.layer_name_to_module[layer_name]

    def _store_compressed_grad(self, layer_idx: int, compressed_grad: Tensor) -> None:
        """
        Store compressed gradient on the weight parameter.

        Also decompresses and stores in param.grad for gradient norm computation.
        Stored on weight (not bias) because compressed grad is one tensor per layer.
        """
        module = self._get_module_from_idx(layer_idx)
        module.weight._compressed_grad = compressed_grad

        # Also decompress and store in param.grad for HF Trainer's grad_norm
        compressor = self.compressors[layer_idx] if self.compressors else None
        if compressor is not None:
            # Decompress: [1, k] -> [1, out_features * (in_features + has_bias)]
            full_grad = compressor.transpose(compressed_grad)  # [1, d]

            # Reshape to [out_features, in_features + has_bias]
            out_features = module.out_features
            in_features = module.in_features
            has_bias = module.bias is not None

            if has_bias:
                # Last column is bias gradient
                full_grad_2d = full_grad.view(out_features, in_features + 1)
                weight_grad = full_grad_2d[:, :in_features]
                bias_grad = full_grad_2d[:, -1]
                module.weight.grad = weight_grad
                module.bias.grad = bias_grad
            else:
                full_grad_2d = full_grad.view(out_features, in_features)
                module.weight.grad = full_grad_2d

    def _get_compressed_grad(self, layer_idx: int) -> Optional[Tensor]:
        """
        Get compressed gradient from the weight parameter of the layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Compressed gradient tensor or None if not set
        """
        module = self._get_module_from_idx(layer_idx)
        return getattr(module.weight, '_compressed_grad', None)

    def _clear_compressed_grad(self, layer_idx: int) -> None:
        """
        Clear compressed gradient from the weight parameter of the layer.

        Args:
            layer_idx: Index of the layer
        """
        module = self._get_module_from_idx(layer_idx)
        if hasattr(module.weight, '_compressed_grad'):
            module.weight._compressed_grad = None

    def clear_all_compressed_grads(self) -> None:
        """Clear all compressed gradients from all layer weight parameters."""
        for layer_idx in range(len(self.layer_names)):
            self._clear_compressed_grad(layer_idx)

    def get_compressed_grads(self) -> List[Optional[Tensor]]:
        """
        Get all captured compressed gradients.

        Returns a list for backward compatibility, but the gradients are actually
        stored on weight._compressed_grad attributes.
        """
        return [self._get_compressed_grad(idx) for idx in range(len(self.layer_names))]

    def refresh_compressors(self, step: int) -> Tuple[int, List[Optional[Any]]]:
        """
        Refresh all compressors if needed.

        Args:
            step: Current training step

        Returns:
            Tuple of (num_refreshed, old_compressors)
        """
        num_refreshed = 0
        old_compressors = []

        for idx, compressor in enumerate(self.compressors):
            if compressor is not None:
                old_container = compressor.refresh(step)
                old_compressors.append(old_container)
                if old_container is not None:
                    num_refreshed += 1
            else:
                old_compressors.append(None)

        return num_refreshed, old_compressors

    def remove_hooks(self) -> None:
        """Restore original forward methods and cleanup registry."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        # Remove from registry
        if self._hook_manager_id in _HOOK_MANAGER_REGISTRY:
            del _HOOK_MANAGER_REGISTRY[self._hook_manager_id]

        logger.info("Restored original forward methods for all layers")

    def __del__(self):
        """Cleanup when hook manager is deleted."""
        if hasattr(self, '_hook_manager_id') and self._hook_manager_id in _HOOK_MANAGER_REGISTRY:
            del _HOOK_MANAGER_REGISTRY[self._hook_manager_id]
