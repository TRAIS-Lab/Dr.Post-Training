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
from .utils import greedy_selection, topk_selection

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
        selection_frac: float,
        lr: float,
        device: str = 'cpu',
        compute_scores_only: bool = False,
        dtype: torch.dtype = torch.float32,
        use_second_order: bool = False
    ):
        """
        Initialize streaming state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers (for tracking backward progress)
            selection_method: Selection method (Streaming, Regular)
            selection_frac: Fraction of samples to select
            lr: Learning rate for score scaling (unused in per-layer mode)
            device: Device for tensors
            compute_scores_only: If True, only maintain running scores without aggregating gradients.
                                This is used for Streaming without MeSO (Case 2), where we compute
                                scores on-the-fly and then do a second forward/backward with full gradients.
            dtype: Data type for score tensors (should match model dtype for efficiency)
            use_second_order: If True, compute similarity matrix and use greedy selection with
                            second-order interactions. If False, use simple top-k selection
                            which is ~200x faster. Default False for efficiency.
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.selection_method = selection_method
        self.selection_frac = selection_frac
        self.lr = lr
        self.device = device
        self.compute_scores_only = compute_scores_only
        self.dtype = dtype
        self.use_second_order = use_second_order

        # Number of samples to select
        self.num_selected = int(train_batch_size * selection_frac)

        # Running scores only needed for compute_scores_only mode (Streaming without MeSO)
        # For per-layer selection with MeSO, we compute scores directly in select_and_reduce
        self.grad_dot_scores = None
        self.similarity_matrix = None
        if compute_scores_only:
            self.grad_dot_scores = torch.zeros(train_batch_size, device=device, dtype=dtype)
            if use_second_order and selection_method == 'Streaming':
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
            return  # Per-layer mode: scores computed in select_and_reduce

        # Cast to score dtype if needed
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Streaming uses gradient alignment (dot product with validation gradient)
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, out=self.grad_dot_scores)

        if self.similarity_matrix is not None:
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), out=self.similarity_matrix)

    def _update_scores_direct(self, scores: Tensor):
        """
        Update running scores directly with pre-computed score values.

        Used for global selection without compression (full gradient mode).

        Args:
            scores: Pre-computed scores [train_batch_size]
        """
        if self.grad_dot_scores is None:
            return

        if scores.dtype != self.dtype:
            scores = scores.to(self.dtype)

        self.grad_dot_scores += scores

    def get_selected_indices(self) -> Tensor:
        """
        Get selected sample indices based on accumulated scores.

        Only used when compute_scores_only=True (Streaming without MeSO).

        Returns:
            Tensor of selected training sample indices
        """
        if self.grad_dot_scores is None:
            raise RuntimeError("get_selected_indices called but running scores not enabled")

        if self.selection_method == 'Streaming':
            scores = self.grad_dot_scores * self.lr
            if self.use_second_order:
                similarity = self.similarity_matrix * (self.lr ** 2)
                selected_indices = greedy_selection(scores, similarity, self.num_selected)
            else:
                selected_indices = topk_selection(scores, self.num_selected)
        else:
            # Regular mode: no selection
            selected_indices = torch.arange(self.train_batch_size, device=self.device)

        return selected_indices

    def select_and_reduce(self, train_grads: Tensor, val_grad: Tensor) -> Tensor:
        """
        Per-layer streaming: select samples based on this layer's scores and reduce.

        Each layer independently selects based on gradient alignment with validation.
        This is the core of gradient streaming - unified selection and aggregation.

        Args:
            train_grads: Per-sample compressed gradients [train_batch_size, k_l]
            val_grad: Mean compressed validation gradient [k_l]

        Returns:
            Reduced gradient [1, k_l] for selected samples
        """
        if self.selection_method == 'Streaming':
            # Per-layer scores: dot product of each sample's gradient with val_grad
            # [train_batch_size, k_l] @ [k_l] -> [train_batch_size]
            scores = train_grads @ val_grad
            selected_indices = topk_selection(scores, self.num_selected)
        else:
            # Regular mode: no selection
            selected_indices = torch.arange(self.train_batch_size, device=self.device)

        selected_grads = train_grads[selected_indices]  # [num_selected, k_l]
        reduced = selected_grads.mean(dim=0, keepdim=True)  # [1, k_l]
        return reduced


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
            if compressor is not None:
                # === COMPRESSED MODE ===
                # Use compressed gradients for scoring and store for MeSO optimizer

                # Prepare input for compressor (add bias column)
                input_for_compressor = input
                if bias is not None:
                    if input.dim() == 3:
                        batch_size, seq_length, in_features = input.shape
                        ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
                    else:
                        ones = torch.ones(input.size(0), 1, device=input.device, dtype=input.dtype)
                    input_for_compressor = torch.cat([input, ones], dim=-1)

                # Compute compressed gradients
                compressed_grad = compressor.forward((grad_output, input_for_compressor))

                if selection_state is not None:
                    # Data selection mode with compression
                    train_grads = compressed_grad[:selection_state.train_batch_size]
                    val_grads = compressed_grad[selection_state.train_batch_size:]
                    val_grad = val_grads.mean(dim=0)

                    # Update running scores (for global selection)
                    selection_state._update_scores(train_grads, val_grad)

                    if not selection_state.compute_scores_only:
                        # Per-layer selection and reduce (Streaming mode)
                        reduced_grad = selection_state.select_and_reduce(train_grads, val_grad)
                        hook_manager.compressed_grads[layer_idx] = reduced_grad
                else:
                    # No selection, just average compressed gradients
                    compressed_grad = compressed_grad.mean(dim=0, keepdim=True)
                    hook_manager.compressed_grads[layer_idx] = compressed_grad

            else:
                # === FULL GRADIENT MODE ===
                # No compression: compute scores directly, return full gradients for standard optimizer

                if selection_state is not None:
                    # Data selection mode with full gradients
                    train_batch_size = selection_state.train_batch_size

                    # Split into train/val
                    train_grad_output = grad_output[:train_batch_size]
                    val_grad_output = grad_output[train_batch_size:]
                    train_input = input[:train_batch_size]
                    val_input = input[train_batch_size:]

                    # Compute validation gradient [out_features, in_features]
                    if val_grad_output.dim() == 3:
                        # 3D: [val_batch, seq, out] x [val_batch, seq, in] -> [out, in]
                        val_grad_full = torch.einsum('vso,vsi->oi', val_grad_output, val_input)
                    else:
                        # 2D: [val_batch, out] x [val_batch, in] -> [out, in]
                        val_grad_full = torch.einsum('vo,vi->oi', val_grad_output, val_input)
                    val_grad_full = val_grad_full / val_grad_output.shape[0]

                    # Compute per-sample scores: score[i] = <grad[i], val_grad>
                    if train_grad_output.dim() == 3:
                        scores = torch.einsum('bso,bsi,oi->b', train_grad_output, train_input, val_grad_full)
                    else:
                        scores = torch.einsum('bo,bi,oi->b', train_grad_output, train_input, val_grad_full)

                    # Update running scores (for global selection mode)
                    selection_state._update_scores_direct(scores)

                    if not selection_state.compute_scores_only:
                        # Per-layer selection (Streaming mode)
                        if selection_state.selection_method == 'Streaming':
                            selected_indices = topk_selection(scores, selection_state.num_selected)
                        else:
                            selected_indices = torch.arange(train_batch_size, device=input.device)

                        # Compute full gradient for selected samples
                        selected_grad_output = train_grad_output[selected_indices]
                        selected_input = train_input[selected_indices]
                        num_selected = selected_grad_output.shape[0]

                        if selected_grad_output.dim() == 3:
                            # 3D: [k, seq, out] x [k, seq, in] -> [out, in]
                            grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) / num_selected
                            if bias is not None:
                                grad_bias = selected_grad_output.sum(dim=(0, 1)) / num_selected
                        else:
                            # 2D: [k, out] x [k, in] -> [out, in]
                            grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) / num_selected
                            if bias is not None:
                                grad_bias = selected_grad_output.sum(dim=0) / num_selected

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
        self.compressed_grads: List[Optional[Tensor]] = [None] * len(layer_names)

        # Unified compressors
        self.compressors: List[Optional[Compressor]] = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True

        # Streaming state (set by trainer before forward/backward)
        self.selection_state: Optional[StreamingState] = None

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
        selection_frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False
    ) -> None:
        """
        Set up streaming state for on-the-fly gradient streaming.

        This should be called by the trainer before each forward/backward pass
        that requires gradient streaming with data selection.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method (Streaming, Regular)
            selection_frac: Fraction of samples to select
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores without aggregating gradients.
                                Used for Streaming without MeSO (Case 2).
            use_second_order: If True, compute similarity matrix and use greedy selection
                            with second-order interactions. If False (default), use simple
                            top-k selection which is ~200x faster.
        """
        # Infer dtype from model parameters (use first parameter's dtype)
        dtype = next(self.model.parameters()).dtype

        num_layers = len(self.layer_names)
        self.selection_state = StreamingState(
            train_batch_size=train_batch_size,
            num_layers=num_layers,
            selection_method=selection_method,
            selection_frac=selection_frac,
            lr=lr,
            device=self.device,
            compute_scores_only=compute_scores_only,
            dtype=dtype,
            use_second_order=use_second_order
        )
        logger.debug(f"Set up streaming state: {train_batch_size} train, scores_only={compute_scores_only}, use_second_order={use_second_order}, dtype={dtype}")

    def clear_selection(self) -> None:
        """Clear selection state after forward/backward."""
        self.selection_state = None

    def get_compressed_grads(self) -> List[Optional[Tensor]]:
        """Get all captured compressed gradients."""
        return self.compressed_grads

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
