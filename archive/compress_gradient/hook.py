"""
Hook manager for efficient gradient compression with prevented materialization.

This implementation uses monkey-patching with custom autograd Functions to prevent
full gradient materialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
import functools
import logging

from .compressor import Compressor

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
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor | None, hook_manager_id: int, layer_idx: int) -> Tensor:
        """
        Forward pass: standard linear transformation.

        CRITICAL: We store hook_manager_id (an int) instead of the hook_manager object.
        Storing the object would keep it in the autograd graph, causing memory leaks.
        """
        # Cast input to weight dtype for computation (handles mixed precision)
        # This mimics PyTorch's autocast behavior
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input

        # MEMORY OPTIMIZATION: Use PyTorch's built-in save_for_backward instead of separate copy
        # PyTorch manages memory efficiently and frees tensors automatically after backward
        # This saves ~1.8 GB for 113 layers with batch_size=8, seq_len=512
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_id = hook_manager_id
        ctx.layer_idx = layer_idx

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
        input, weight, bias = ctx.saved_tensors
        hook_manager_id = ctx.hook_manager_id
        layer_idx = ctx.layer_idx

        # Lookup hook manager from registry
        hook_manager = _HOOK_MANAGER_REGISTRY.get(hook_manager_id)
        if hook_manager is None:
            raise RuntimeError(f"Hook manager {hook_manager_id} not found in registry")

        # Input is already at weight dtype from forward pass
        # Only cast if truly necessary (rare case where dtypes differ)
        if input.dtype != grad_output.dtype:
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
                compressor=hook_manager.compressors[layer_idx],
            )

            # MEMORY OPTIMIZATION: Optionally aggregate across batch dimension
            # - If aggregate_grads=True: saves ~1.1 GB (for standard training)
            # - If aggregate_grads=False: keeps per-sample grads (needed for GREATS)
            # if hook_manager.aggregate_grads:
            #     # Aggregate: [batch, k] → [1, k]
            #     compressed_grad = compressed_grad.mean(dim=0, keepdim=True)

            # Store compressed gradient in hook manager
            hook_manager.compressed_grads[layer_idx] = compressed_grad

        # Return gradients:
        # - grad_input: needed for backprop
        # - None for weight: tells PyTorch NOT to compute weight.grad
        # - None for bias: tells PyTorch NOT to compute bias.grad
        # - None for hook_manager_id, layer_idx: not tensors
        # Note: PyTorch automatically frees saved tensors (including input) after backward
        return grad_input, None, None, None, None


def _compute_compressed_grad(grad_output: Tensor, input: Tensor, has_bias: bool, compressor: Compressor) -> Tensor:
    """
    Compute compressed gradient without materializing full gradient.

    Process:
    1. Add bias term to input if needed (gradient-specific preprocessing)
    2. Pass components to compressor which handles:
    """
    # Gradient-specific preprocessing: Add bias column if needed
    if has_bias:
        # Determine if input is 3D (batch, seq, features) or 2D (batch, features)
        if input.dim() == 3:
            batch_size, seq_length, in_features = input.shape
            ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
        else:
            ones = torch.ones(input.size(0), 1, device=input.device, dtype=input.dtype)

        input = torch.cat([input, ones], dim=-1)

    # Delegate to compressor: it handles component-based compression
    compressed_grad = compressor.forward((grad_output, input))

    return compressed_grad


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
        # Storage for compressed gradients (computed during backward)
        # Note: inputs are now stored in PyTorch's autograd graph (via save_for_backward)
        # which is more memory-efficient than storing separate copies
        self.compressed_grads = [None] * len(layer_names)

        # Unified compressors (handles all compression/decompression operations)
        self.compressors = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered = False

        # Track whether hooks are enabled (can disable without unregistering)
        self.hooks_enabled = True

        # Memory optimization flag: aggregate gradients across batch dimension
        # Set to False when using GREATS (needs per-sample gradients)
        # Set to True for standard training (saves ~1.1 GB memory)
        self.aggregate_grads = True  # Default: aggregate for memory efficiency

        # Register in global registry: Store ID, not self, to avoid memory leaks in autograd graph
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

                # Only support Linear layers (note that lora adapters from peft package are also Linear)
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

        When hooks are enabled: Uses CompressedLinearBackward to compute compressed gradients
        When hooks are disabled: Uses original forward method for proper gradient computation

        Note: Hooks are attached to the actual trainable layers:
        - For LoRA: lora_A and lora_B modules (the trainable adapters)
        - For full fine-tuning: the Linear layer weights
        """
        if self.hooks_enabled:
            # Use our custom backward that computes only compressed gradients
            # Pass hook_manager_id (not self) to avoid keeping hook manager in autograd graph
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self._hook_manager_id, idx
            )
        else:
            # Use original forward method when hooks are disabled
            # This ensures proper gradient computation for LoRA and other special layers
            if hasattr(module, '_original_forward'):
                return module._original_forward(input)
            else:
                # Fallback to F.linear if original forward not available
                return F.linear(input, module.weight, module.bias)

    def set_compressors(self, compressors: List[Compressor]) -> None:
        """Set unified compressor objects for each layer."""
        self.compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute compressed gradients (default)."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    def get_compressed_grads(self) -> List[Tensor]:
        """Get all captured compressed gradients."""
        return self.compressed_grads

    def refresh_compressors(self, step: int):
        """
        Refresh all compressors if needed.

        Args:
            step: Current training step

        Returns:
            Tuple of (num_refreshed, old_compressors) where:
                - num_refreshed: Number of compressors refreshed
                - old_compressors: List of old Compressor (or None if not refreshed)
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
