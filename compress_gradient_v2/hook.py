"""
Hook manager for efficient gradient compression with on-the-fly data selection.

This implementation uses monkey-patching with custom autograd Functions to:
1. Prevent full gradient materialization
2. Compute selection scores layer-by-layer during backward
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
from .utils import greedy_selection

logger = logging.getLogger(__name__)

# Global registry: maps a unique ID to hook manager
# CRITICAL: Used to avoid storing hook_manager in autograd context, which would cause memory leaks
_HOOK_MANAGER_REGISTRY: Dict[int, 'GradientHook'] = {}


class SelectionState:
    """
    State manager for on-the-fly data selection during backward pass.

    This class maintains running statistics (grad_dot_scores, similarity_matrix)
    that are updated layer-by-layer during backward. This avoids materializing
    per-sample gradients for all layers at once, preventing OOM for large batches.
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
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize selection state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers (for tracking backward progress)
            selection_method: Selection method (GREATS, GradNorm, etc.)
            selection_frac: Fraction of samples to select
            lr: Learning rate for score scaling
            device: Device for tensors
            compute_scores_only: If True, only maintain running scores without aggregating gradients.
                                This is used for GREATS without MeSO (Case 2), where we compute
                                scores on-the-fly and then do a second forward/backward with full gradients.
            dtype: Data type for score tensors (should match model dtype for efficiency)
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.selection_method = selection_method
        self.selection_frac = selection_frac
        self.lr = lr
        self.device = device
        self.compute_scores_only = compute_scores_only
        self.dtype = dtype

        # Running statistics (updated layer-by-layer during backward)
        # Use model dtype for efficiency (bfloat16/float16 on modern GPUs)
        self.grad_dot_scores = torch.zeros(train_batch_size, device=device, dtype=dtype)
        self.similarity_matrix = None
        if selection_method in ['GREATS', 'GradNorm']:
            self.similarity_matrix = torch.zeros(train_batch_size, train_batch_size, device=device, dtype=dtype)

        # Number of samples to select
        self.num_selected = int(train_batch_size * selection_frac)

    def _update_scores(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
    ):
        """
        Update running scores with gradients from current layer.

        This is called during backward for each layer.

        Args:
            train_grads: Per-sample compressed gradients [train_batch_size, k_l]
            val_grad: Mean compressed validation gradient [k_l]
            layer_idx: Current layer index
        """
        # Cast to score dtype if needed (typically matches model dtype for efficiency)
        # For bfloat16/float16, this keeps computation in lower precision
        # For float32, this is a no-op
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Update grad_dot_scores: scores += train_grads @ val_grad
        # [train_batch_size, k_l] @ [k_l] -> [train_batch_size]
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, out=self.grad_dot_scores)

        # Update similarity matrix if needed
        if self.similarity_matrix is not None:
            # similarity += train_grads @ train_grads.T
            # [train_batch_size, k_l] @ [k_l, train_batch_size] -> [train_batch_size, train_batch_size]
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), out=self.similarity_matrix)

    def get_selected_indices(self) -> Tensor:
        """
        Get selected sample indices based on final accumulated scores.

        This is called after the full forward/backward pass when compute_scores_only=True
        (Case 2: GREATS without MeSO). The scores have been accumulated across all layers
        during the backward pass.

        Returns:
            Tensor of selected training sample indices
        """
        if self.selection_method == 'GREATS':
            scores = self.grad_dot_scores * self.lr
            similarity = self.similarity_matrix * (self.lr ** 2)
            selected_indices = greedy_selection(scores, similarity, self.num_selected)

        elif self.selection_method == 'GradNorm':
            scores = torch.diag(self.similarity_matrix)
            selected_indices = greedy_selection(scores, self.similarity_matrix * 0, self.num_selected)

        else:
            # Default: select all (no selection)
            selected_indices = torch.arange(self.train_batch_size, device=self.device)

        return selected_indices

    def select_and_reduce(self, train_grads: Tensor) -> Tensor:
        """
        Progressive selection: select samples based on current scores and reduce.

        Args:
            train_grads: Per-sample compressed gradients [train_batch_size, k_l]
            layer_idx: Current layer index

        Returns:
            Reduced gradient [k_l] for selected samples
        """
        selected_indices = self.get_selected_indices()
        selected_grads = train_grads[selected_indices]  # [num_selected, k_l]
        reduced = selected_grads.mean(dim=0, keepdim=True)  # [1, k_l]
        return reduced


class CompressedLinearBackward(Function):
    """
    Custom autograd Function with on-the-fly data selection.
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
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        """
        Backward pass with progressive (approximate online) selection.
        """
        input, weight, bias = ctx.saved_tensors
        hook_manager_id: int = ctx.hook_manager_id
        layer_idx: int = ctx.layer_idx

        # Lookup hook manager from registry
        hook_manager: Optional['GradientHook'] = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            raise RuntimeError(f"Hook manager {hook_manager_id} not found in registry")

        # Cast input to match grad_output dtype if needed
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Compute grad_input (needed for backprop to previous layers)
        grad_input = grad_output @ weight.to(grad_output.dtype)

        # Compute compressed gradient directly (without full gradient)
        with torch.no_grad():
            # Gradient-specific preprocessing: Add bias column to input if needed
            if bias is not None:
                # Determine if input is 3D (batch, seq, features) or 2D (batch, features)
                if input.dim() == 3:
                    batch_size, seq_length, in_features = input.shape
                    ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
                else:
                    ones = torch.ones(input.size(0), 1, device=input.device, dtype=input.dtype)

                input = torch.cat([input, ones], dim=-1)

            # Delegate to compressor: it handles component-based compression
            compressed_grad = hook_manager.compressors[layer_idx].forward((grad_output, input))

            # Check if we have selection state (data selection mode)
            selection_state = hook_manager.selection_state

            if selection_state is not None:
                # Assumes merged batch is [train_samples, val_samples] in that order
                train_grads = compressed_grad[:selection_state.train_batch_size]  # [train_batch_size, k_l]
                val_grads = compressed_grad[selection_state.train_batch_size:]  # [val_batch_size, k_l]

                # Compute mean validation gradient
                val_grad = val_grads.mean(dim=0)  # [k_l]

                # Step 1: Update running scores
                selection_state._update_scores(train_grads, val_grad)

                # Check whether aggregation is needed
                if not selection_state.compute_scores_only:
                    # Select and reduce based on current scores
                    reduced_grad = selection_state.select_and_reduce(train_grads)

                    # Store reduced gradient
                    hook_manager.compressed_grads[layer_idx] = reduced_grad

                else:
                    # We only maintain running scores without aggregating gradients.
                    # After the full backward pass, the trainer will:
                    # 1. Get selected indices from selection_state.get_selected_indices()
                    # 2. Do a second forward/backward with full gradients on selected samples
                    pass
            else:
                # No data selection
                # Average compressed gradients across batch dimension
                compressed_grad = compressed_grad.mean(dim=0, keepdim=True) # [1, k_l]

                # Store reduced compressed gradient
                hook_manager.compressed_grads[layer_idx] = compressed_grad

        # Return gradients
        return grad_input, None, None, None, None


class GradientHook:
    """
    Hook manager with on-the-fly data selection support.

    New features:
    1. SelectionState for managing selection during backward
    2. On-the-fly score computation and gradient aggregation
    3. Memory-efficient processing without materializing all per-sample grads
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

        # Selection state (set by trainer before forward/backward)
        self.selection_state: Optional[SelectionState] = None

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
        Replacement forward method that uses our custom Function.
        """
        if self.hooks_enabled:
            return CompressedLinearBackward.apply(
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
        compute_scores_only: bool = False
    ) -> None:
        """
        Set up selection state for on-the-fly data selection.

        This should be called by the trainer before each forward/backward pass
        that requires data selection.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method (GREATS, GradNorm, etc.)
            selection_frac: Fraction of samples to select
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores without aggregating gradients.
                                Used for GREATS without MeSO (Case 2).
        """
        # Infer dtype from model parameters (use first parameter's dtype)
        dtype = next(self.model.parameters()).dtype

        num_layers = len(self.layer_names)
        self.selection_state = SelectionState(
            train_batch_size=train_batch_size,
            num_layers=num_layers,
            selection_method=selection_method,
            selection_frac=selection_frac,
            lr=lr,
            device=self.device,
            compute_scores_only=compute_scores_only,
            dtype=dtype
        )
        logger.debug(f"Set up selection state: {train_batch_size} train, scores_only={compute_scores_only}, dtype={dtype}")

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
