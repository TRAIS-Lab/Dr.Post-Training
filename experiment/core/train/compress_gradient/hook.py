"""
Hook manager for efficient gradient compression with prevented materialization.

This implementation uses monkey-patching with custom autograd Functions to prevent
full gradient materialization. Key technique:

1. **Monkey-Patching**: Replace module.forward with custom function
2. **Custom Autograd Function**: Control backward pass to compute ONLY compressed gradients
3. **Return None for weight.grad**: Tells PyTorch to skip full gradient computation
4. **Centralized Storage**: All data stored in hook manager (memory efficient)
5. **Global Registry**: Prevents memory leaks in autograd graph

This prevents PyTorch from computing full gradients that we don't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
import functools
import logging

from .compressor import SparsifierContainer, ProjectorContainer

logger = logging.getLogger(__name__)

# Global registry: maps a unique ID to hook manager
# CRITICAL: Used to avoid storing hook_manager in autograd context, which would cause memory leaks
_HOOK_MANAGER_REGISTRY = {}


class CompressedLinearBackward(Function):
    """
    Custom autograd Function that prevents full gradient materialization.

    Key mechanism: When backward() returns None for a parameter, PyTorch skips
    computing that parameter's gradient entirely. This is how we avoid materializing
    the full weight gradient.
    """

    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor | None,
                hook_manager_id: int, layer_idx: int) -> Tensor:
        """
        Forward pass: standard linear transformation.

        CRITICAL: We store hook_manager_id (an int) instead of the hook_manager object.
        Storing the object would keep it in the autograd graph, causing memory leaks.
        """
        ctx.save_for_backward(weight, bias)
        ctx.hook_manager_id = hook_manager_id
        ctx.layer_idx = layer_idx

        # Lookup hook manager from global registry
        hook_manager = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            raise RuntimeError(f"Hook manager {hook_manager_id} not found in registry")

        # Store input for backward pass (centralized storage)
        # Store at original dtype - will be cast during backward if needed
        hook_manager.inputs[layer_idx] = input.detach()

        # Cast input to weight dtype for computation (handles mixed precision)
        # This mimics PyTorch's autocast behavior
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input

        # Standard forward pass (same as nn.Linear)
        output = F.linear(input_compute, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass: compute ONLY compressed gradient, not the full gradient.

        By returning None for weight and bias, we tell PyTorch: "don't compute
        param.grad for these, I handled it myself". This is the key mechanism
        that prevents full gradient materialization.
        """
        weight, bias = ctx.saved_tensors
        hook_manager_id = ctx.hook_manager_id
        layer_idx = ctx.layer_idx

        # Lookup hook manager from registry
        hook_manager = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            raise RuntimeError(f"Hook manager {hook_manager_id} not found in registry")

        # Retrieve stored input
        input = hook_manager.inputs[layer_idx]

        # Cast input to match grad_output dtype for mixed precision training
        # This ensures all operations in _compute_compressed_grad work correctly
        input = input.to(grad_output.dtype)

        # Compute grad_input (needed for backprop to previous layers)
        # Cast weight to match grad_output dtype (important for mixed precision training)
        # This mimics PyTorch's internal behavior in linear backward
        grad_input = grad_output @ weight.to(grad_output.dtype)

        # Compute compressed gradient directly (without full gradient)
        with torch.no_grad():
            compressed_grad = _compute_compressed_grad(
                grad_output=grad_output,
                input=input,
                has_bias=(bias is not None),
                sparsifier=hook_manager.sparsifiers[layer_idx],
                projector=hook_manager.projectors[layer_idx],
            )

            # Store compressed gradient in hook manager
            hook_manager.compressed_grads[layer_idx] = compressed_grad

            # Free input immediately
            hook_manager.inputs[layer_idx] = None

        # Return gradients:
        # - grad_input: needed for backprop
        # - None for weight: tells PyTorch NOT to compute weight.grad
        # - None for bias: tells PyTorch NOT to compute bias.grad
        # - None for hook_manager_id, layer_idx: not tensors
        return grad_input, None, None, None, None


def _compute_compressed_grad(
    grad_output: Tensor,
    input: Tensor,
    has_bias: bool,
    sparsifier: Any,
    projector: Any,
) -> Tensor:
    """
    Compute compressed gradient without materializing full gradient.

    Process:
    1. Add bias term to input (if needed)
    2. Apply sparsification BEFORE outer product (reduces dimensions early)
    3. Compute outer product in compressed space
    4. Apply projection to get final compressed gradient
    """
    # Determine dimensions
    is_3d = input.dim() == 3

    if is_3d:
        batch_size, seq_length, in_features = input.shape
        out_features = grad_output.shape[2]
    else:
        batch_size = input.shape[0]
        in_features = input.shape[-1]
        out_features = grad_output.shape[-1]

    # Reshape to 2D if needed
    if is_3d:
        input_2d = input.reshape(-1, in_features)
        grad_output_2d = grad_output.reshape(-1, out_features)
    else:
        input_2d = input
        grad_output_2d = grad_output

    # Add bias term BEFORE sparsification
    if has_bias:
        ones = torch.ones(input_2d.size(0), 1,
                         device=input_2d.device,
                         dtype=input_2d.dtype)
        input_2d = torch.cat([input_2d, ones], dim=1)

    # Apply sparsification BEFORE outer product (if available)
    if sparsifier and hasattr(sparsifier, 'sparsifier_comp') and sparsifier.sparsifier_comp != (None, None):
        sparsifier_output, sparsifier_input = sparsifier.sparsifier_comp

        # Apply sparsifiers
        grad_output_sparse = sparsifier_output(grad_output_2d)
        input_sparse = sparsifier_input(input_2d)

        # Compute per-sample gradients with scaling
        if is_3d:
            grad_sparse_3d = grad_output_sparse.reshape(batch_size, seq_length, -1)
            input_sparse_3d = input_sparse.reshape(batch_size, seq_length, -1)

            grad_tensor = torch.einsum('bsi,bsj->bij',
                                      grad_sparse_3d * batch_size,
                                      input_sparse_3d)
        else:
            grad_tensor = torch.einsum('bi,bj->bij',
                                      grad_output_sparse * batch_size,
                                      input_sparse)

        grad = grad_tensor.reshape(batch_size, -1)

    else:
        # No sparsification: compute outer product directly
        if is_3d:
            input_3d = input_2d.reshape(batch_size, seq_length, -1)
            grad_output_3d = grad_output_2d.reshape(batch_size, seq_length, -1)

            grad_tensor = torch.einsum('bsi,bsj->bij',
                                      grad_output_3d * batch_size,
                                      input_3d)
        else:
            grad_tensor = torch.einsum('bi,bj->bij',
                                      grad_output_2d * batch_size,
                                      input_2d)

        grad = grad_tensor.reshape(batch_size, -1)

    # Apply projection (operates AFTER outer product)
    if projector and hasattr(projector, 'projector') and projector.projector is not None:
        grad = projector.projector(grad)

    return grad


class GradientHook:
    """
    Hook manager that prevents full gradient materialization through monkey-patching.

    How it works:
    1. Replaces module.forward with a custom function that uses CompressedLinearBackward
    2. CompressedLinearBackward.backward() computes ONLY compressed gradients
    3. Returns None for weight/bias gradients, telling PyTorch to skip full computation
    4. Stores everything centrally in hook manager (memory efficient)
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
            register_hooks: Whether to register hooks immediately (monkey-patch forward methods)
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device

        # Create mapping from layer name to index
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        # Create mapping from layer name to module
        self.layer_name_to_module = {}

        # Centralized storage arrays
        self.forward_hooks = [None] * len(layer_names)
        self.compressed_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)

        # Compression components
        self.sparsifiers = [None] * len(layer_names)
        self.projectors = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered = False

        # Register in global registry
        # CRITICAL: Store ID, not self, to avoid memory leaks in autograd graph
        self._hook_manager_id = id(self)
        _HOOK_MANAGER_REGISTRY[self._hook_manager_id] = self

        # Register hooks if requested
        if register_hooks:
            self._register_hooks()

        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    def _register_hooks(self):
        """
        Monkey-patch Linear layers to use our custom Function.

        This replaces module.forward with our custom function that prevents
        full gradient materialization.
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

                # Save original forward method (so we can restore it later)
                module._original_forward = module.forward

                # Create wrapped forward that uses our custom Function
                wrapped_forward = functools.partial(
                    self._custom_linear_forward, module, idx
                )

                # Replace the forward method (monkey-patching)
                module.forward = wrapped_forward

        self.hooks_registered = True
        logger.info(f"Successfully wrapped {len(self.layer_names)} layers")

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """
        Replacement forward method that uses our custom Function.

        During training: Uses CompressedLinearBackward to prevent full gradient
        During eval: Uses standard F.linear (no overhead)
        """
        if module.training and input.requires_grad:
            # Use our custom backward that computes only compressed gradients
            # Pass hook_manager_id (not self) to avoid keeping hook manager in autograd graph
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self._hook_manager_id, idx
            )
        else:
            # Use standard forward during eval
            return F.linear(input, module.weight, module.bias)

    def set_sparsifiers(self, sparsifiers: List[SparsifierContainer]) -> None:
        """Set sparsifier objects for each layer."""
        self.sparsifiers = sparsifiers

    def set_projectors(self, projectors: List[ProjectorContainer]) -> None:
        """Set projector objects for each layer."""
        self.projectors = projectors

    def get_compressed_grads(self) -> List[Tensor]:
        """Get all captured compressed gradients."""
        return self.compressed_grads

    def get_decompressed_grads(self, selected_indices: List[int]) -> dict:
        """
        Apply transpose operations to recover full gradients from compressed gradients.

        Args:
            selected_indices: List of sample indices to aggregate

        Returns:
            Dictionary mapping layer names to full gradients
        """
        full_grads = {}

        for idx, layer_name in enumerate(self.layer_names):
            compressed_grad = self.compressed_grads[idx]

            if compressed_grad is None:
                continue

            # Aggregate selected samples
            if selected_indices is not None and len(selected_indices) > 0:
                selected_compressed = compressed_grad[selected_indices]
                aggregated_compressed = selected_compressed.mean(dim=0, keepdim=True)
            else:
                aggregated_compressed = compressed_grad.mean(dim=0, keepdim=True)

            # Get compressors
            projector = self.projectors[idx] if idx < len(self.projectors) else None
            sparsifier = self.sparsifiers[idx] if idx < len(self.sparsifiers) else None

            # Apply transpose in reverse order
            if projector and hasattr(projector, 'transpose'):
                g_intermediate = projector.transpose(aggregated_compressed)
            else:
                g_intermediate = aggregated_compressed

            if sparsifier and hasattr(sparsifier, 'transpose'):
                g_full = sparsifier.transpose(g_intermediate)
            else:
                g_full = g_intermediate

            if g_full is not None:
                full_grads[layer_name] = g_full

        return full_grads

    def refresh_compressors(self, step: int) -> int:
        """
        Refresh all projectors and sparsifiers if needed.

        Args:
            step: Current training step

        Returns:
            Number of compressors refreshed
        """
        num_refreshed = 0

        for idx, sparsifier in enumerate(self.sparsifiers):
            if sparsifier is not None and sparsifier.refresh(step):
                num_refreshed += 1

        for idx, projector in enumerate(self.projectors):
            if projector is not None and projector.refresh(step):
                num_refreshed += 1

        return num_refreshed

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
