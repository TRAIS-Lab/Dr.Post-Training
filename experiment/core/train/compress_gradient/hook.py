"""
Hook manager for efficient gradient component capture and projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools
import logging

import torch.jit as jit
from .compressor import SparsifierContainer, ProjectorContainer

# Configure logger
logger = logging.getLogger(__name__)

@jit.script
def linear_grad_2d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 2D tensors.

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, output_dim]
        input_features: Input features with shape [batch_size, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    output_dim = grad_pre_activation.shape[1]
    input_dim = input_features.shape[1]

    grad_tensor = torch.einsum('bi,bj->bij', grad_pre_activation, input_features)
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

@jit.script
def linear_grad_3d(grad_pre_activation: Tensor, input_features: Tensor) -> Tensor:
    """
    Compute weight gradients using outer products for 3D tensors (sequence data).

    Args:
        grad_pre_activation: Gradient of pre-activation with shape [batch_size, seq_length, output_dim]
        input_features: Input features with shape [batch_size, seq_length, input_dim]

    Returns:
        Tensor of shape [batch_size, output_dim * input_dim] containing per-sample gradients
    """
    batch_size = input_features.shape[0]
    output_dim = grad_pre_activation.shape[2]
    input_dim = input_features.shape[2]

    grad_tensor = torch.einsum('bsi,bsj->bij', grad_pre_activation, input_features)
    return grad_tensor.reshape(batch_size, output_dim * input_dim)

class GradientHook:
    """
    Manages hooks for efficient gradient component capturing and projection
    without requiring custom layer implementations.
    """
    def __init__(
            self,
            model: nn.Module,
            layer_names: List[str],
            device: str = 'cpu',
            register_hooks: bool = True
        ) -> None:
        """
        Initialize the hook manager

        Args:
            model: The model to hook
            layer_names: Names of layers to hook
            device: Device to use for profiling synchronization
            register_hooks: Whether to register hooks immediately (default: True)
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device

        # Create mapping from layer name to index for O(1) lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(layer_names)}

        # Create mapping from layer name to module for O(1) lookups
        self.layer_name_to_module = {}

        self.forward_hooks = [None] * len(layer_names)
        self.backward_hooks = [None] * len(layer_names)
        self.compressed_grads = [None] * len(layer_names)
        self.inputs = [None] * len(layer_names)
        self.pre_activations = [None] * len(layer_names)
        self.normalized = [None] * len(layer_names)

        # Initialize sparsifiers and projectors (to avoid AttributeError when not set)
        self.sparsifiers = [None] * len(layer_names)
        self.projectors = [None] * len(layer_names)

        # Flag to track if hooks are currently registered
        self.hooks_registered = False

        # Register hooks if requested
        if register_hooks:
            self._register_hooks()

        logger.info(f"Initialized GradientHook with {len(layer_names)} layers (hooks {'registered' if register_hooks else 'NOT registered'})")

    def _register_hooks(self):
        """Register forward hooks to target layers (backward hooks are registered dynamically via tensors)"""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping registration")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]

                # Cache the module for efficient lookup later
                self.layer_name_to_module[name] = module

                forward_hook = functools.partial(self._forward_hook_fn, name)

                # Register only forward hook - backward hooks are registered dynamically on tensors
                # to avoid BackwardHookFunction wrapper issues with in-place operations
                self.forward_hooks[idx] = module.register_forward_hook(forward_hook)

        self.hooks_registered = True
        logger.info(f"Successfully registered hooks for {len(self.layer_names)} layers")

    def set_sparsifiers(self, sparsifiers: List[SparsifierContainer]) -> None:
        """
        Set specifier objects for each layer

        Args:
            sparsifiers: List of projector objects, ordered by layer_names
        """
        self.sparsifiers = sparsifiers

    def set_projectors(self, projectors: List[ProjectorContainer]) -> None:
        """
        Set projector objects for each layer

        Args:
            projectors: List of projector objects, ordered by layer_names
        """
        self.projectors = projectors

    def refresh_compressors(self, step: int) -> int:
        """
        Refresh all projectors and sparsifiers if needed based on current training step.

        Args:
            step: Current training step

        Returns:
            Number of compressors (projectors + sparsifiers) that were refreshed
        """
        num_refreshed = 0

        # Refresh sparsifiers
        for idx, sparsifier in enumerate(self.sparsifiers):
            if sparsifier is not None and sparsifier.refresh(step):
                num_refreshed += 1

        # Refresh projectors
        for idx, projector in enumerate(self.projectors):
            if projector is not None and projector.refresh(step):
                num_refreshed += 1

        return num_refreshed

    def get_compressed_grads(self) -> List[Tensor]:
        """
        Get all captured projected gradients

        Returns:
            List of projected gradient tensors, ordered by layer_names
        """
        return self.compressed_grads

    def get_decompressed_grads(self, selected_indices: List[int]) -> dict:
        """
        Apply GraSS transpose to recover full gradients from compressed gradients.

        This method aggregates the selected per-sample compressed gradients and then
        applies the transpose operation to recover full gradients suitable for parameter updates.

        Args:
            selected_indices: List of sample indices to aggregate (for data selection)

        Returns:
            Dictionary mapping layer names to full gradients [1, p_l] ready for parameter updates
        """
        full_grads = {}

        for idx, layer_name in enumerate(self.layer_names):
            compressed_grad = self.compressed_grads[idx]

            if compressed_grad is None:
                continue

            # Aggregate selected samples
            if selected_indices is not None and len(selected_indices) > 0:
                selected_compressed = compressed_grad[selected_indices]  # [|S|, k_l]
                aggregated_compressed = selected_compressed.mean(dim=0, keepdim=True)  # [1, k_l]
            else:
                # If no selection, use mean of all samples
                aggregated_compressed = compressed_grad.mean(dim=0, keepdim=True)  # [1, k_l]

            # Get projector and sparsifier for this layer
            projector = self.projectors[idx] if idx < len(self.projectors) else None
            sparsifier = self.sparsifiers[idx] if idx < len(self.sparsifiers) else None

            # Apply transpose in REVERSE order of compression:
            # Forward compression: g → (sparsify) → g' → (project) → ĝ
            # Backward transpose:  ĝ → (project^T) → g' → (sparsify^T) → ḡ

            # Step 1: Apply projection transpose (stage 2) if available
            if projector and hasattr(projector, 'transpose'):
                g_intermediate = projector.transpose(aggregated_compressed)  # ĝ → g'
            else:
                g_intermediate = aggregated_compressed

            # Step 2: Apply sparsification transpose (stage 1) if available
            if sparsifier and hasattr(sparsifier, 'transpose'):
                g_full = sparsifier.transpose(g_intermediate)  # g' → ḡ
            else:
                g_full = g_intermediate

            if g_full is not None:
                full_grads[layer_name] = g_full

        return full_grads


    def _forward_hook_fn(self, name: str, mod: nn.Module, inp: Any, out: Any) -> Any:
        """
        Forward hook function that captures inputs and pre-activations

        Args:
            name: Layer name
            mod: Module instance
            inp: Input tensors
            out: Output tensors

        Returns:
            The output (potentially cloned to avoid in-place modification issues)
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Store input
        if isinstance(inp, tuple) and len(inp) > 0:
            self.inputs[idx] = inp[0].detach()
        else:
            self.inputs[idx] = inp.detach()

        # Store pre-activation (output) - clone to avoid view issues
        if isinstance(out, tuple):
            self.pre_activations[idx] = tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in out)
            # Register tensor hook on the first tensor output to avoid BackwardHookFunction issues
            if len(out) > 0 and isinstance(out[0], torch.Tensor) and out[0].requires_grad:
                out[0].register_hook(lambda grad, n=name: self._tensor_backward_hook(n, grad))
        else:
            self.pre_activations[idx] = out.detach()
            # Register tensor hook to avoid BackwardHookFunction issues with in-place ops
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(lambda grad, n=name: self._tensor_backward_hook(n, grad))

        # For LayerNorm, also capture the normalized tensor
        if isinstance(mod, nn.LayerNorm):
            x = inp[0] if isinstance(inp, tuple) else inp
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            normalized = (x - mean) / torch.sqrt(var + mod.eps)
            self.normalized[idx] = normalized.detach()

        # Return None to not modify the output
        return None

    def _tensor_backward_hook(self, name: str, grad_output: Tensor) -> None:
        """
        Tensor-level backward hook that computes projected gradients.
        This is called during backward pass on the output tensor of a layer.

        Args:
            name: Layer name
            grad_output: Gradient w.r.t the output tensor
        """
        # Get the module and index for this layer
        idx = self.layer_name_to_idx[name]
        mod = self.layer_name_to_module.get(name)

        if mod is None:
            return

        # Calculate the projected gradient based on layer type
        with torch.no_grad():
            if isinstance(mod, nn.Linear):
                grad = self._linear_grad(
                    mod, idx, grad_output, per_sample=True
                )
            elif isinstance(mod, nn.LayerNorm):
                grad = self._layernorm_grad(
                    mod, idx, grad_output, per_sample=True
                )
            elif isinstance(mod, nn.Embedding):
                # Embeddings would need their own implementation
                grad = None
            else:
                # Fallback for other layer types
                grad = None

            if grad is not None:
                # Store the projected gradient
                self.compressed_grads[idx] = grad.detach()

    def _backward_hook_fn(self, name: str, mod: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """
        Backward hook function that computes projected gradients

        Args:
            name: Layer name
            mod: Module instance
            grad_input: Gradient w.r.t inputs
            grad_output: Gradient w.r.t outputs
        """
        # Get the index for this layer
        idx = self.layer_name_to_idx[name]

        # Get the gradient of the pre-activation
        grad_pre_activation = grad_output[0]

        # Calculate the projected gradient based on layer type
        with torch.no_grad():
            if isinstance(mod, nn.Linear):
                grad = self._linear_grad(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.LayerNorm):
                grad = self._layernorm_grad(
                    mod, idx, grad_pre_activation, per_sample=True
                )
            elif isinstance(mod, nn.Embedding):
                # Embeddings would need their own implementation
                grad = None
            else:
                # Fallback for other layer types
                grad = None

            if grad is not None:
                # Store the projected gradient
                self.compressed_grads[idx] = grad.detach()

    def _linear_grad(
        self,
        layer: nn.Linear,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for Linear layers with strict two-step compression:

        STRICT CONVENTION:
        - Stage 1 (Sparsification): ALWAYS factorized, operates BEFORE outer product
          Applied component-wise to (grad_output, input_features)
        - Stage 2 (Projection): NEVER factorized, operates AFTER outer product
          Applied to flattened gradient

        Forward compression: (g_out, g_in) → (g_out', g_in') → ĝ
                            [sparsify components] [outer product] [project]

        Args:
            layer: Linear layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Compressed gradient tensor [batch, k_l] or [batch, p_l] if no compression
        """
        input_features = self.inputs[idx]
        is_3d = input_features.dim() == 3

        # Get sparsifier and projector for this layer
        sparsifier = self.sparsifiers[idx] if hasattr(self, 'sparsifiers') and idx < len(self.sparsifiers) else None
        projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None

        # Process tensors for gradient computation
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features_flat = input_features.reshape(-1, hidden_size)
            grad_pre_activation_flat = grad_pre_activation.reshape(-1, layer.out_features)
        else:
            batch_size = input_features.shape[0]
            input_features_flat = input_features
            grad_pre_activation_flat = grad_pre_activation

        # Scale gradient for per-sample computation
        if per_sample:
            grad_pre_activation_flat = grad_pre_activation_flat * batch_size

        # Add bias term by augmenting input with ones
        if layer.bias is not None:
            ones = torch.ones(
                input_features_flat.size(0), 1,
                device=input_features_flat.device,
                dtype=input_features_flat.dtype
            )
            input_features_flat = torch.cat([input_features_flat, ones], dim=1)

        # STRICT TWO-STEP COMPRESSION:
        # Stage 1: Apply sparsification (ALWAYS factorized, operates BEFORE outer product)
        if sparsifier and hasattr(sparsifier, 'sparsifier_comp') and sparsifier.sparsifier_comp != (None, None):
            sparsifier_output, sparsifier_input = sparsifier.sparsifier_comp

            if is_3d:
                # Apply sparsification to components
                grad_pre_activation_sparse = sparsifier_output(grad_pre_activation_flat).reshape(
                    batch_size, seq_length, -1
                )
                input_features_sparse = sparsifier_input(input_features_flat).reshape(
                    batch_size, seq_length, -1
                )

                # Compute outer product with sparsified components
                grad_tensor = linear_grad_3d(grad_pre_activation_sparse, input_features_sparse)
            else:
                # Apply sparsification to components
                grad_pre_activation_sparse = sparsifier_output(grad_pre_activation_flat)
                input_features_sparse = sparsifier_input(input_features_flat)

                # Compute outer product with sparsified components
                grad_tensor = linear_grad_2d(grad_pre_activation_sparse, input_features_sparse)

            grad = grad_tensor.reshape(batch_size, -1)
        else:
            # No sparsification: compute outer product directly
            if is_3d:
                input_features_3d = input_features_flat.reshape(batch_size, seq_length, -1)
                grad_pre_activation_3d = grad_pre_activation_flat.reshape(batch_size, seq_length, -1)
                grad_tensor = linear_grad_3d(grad_pre_activation_3d, input_features_3d)
            else:
                grad_tensor = linear_grad_2d(grad_pre_activation_flat, input_features_flat)

            grad = grad_tensor.reshape(batch_size, -1)

        # Stage 2: Apply projection (NEVER factorized, operates AFTER outer product)
        if projector and hasattr(projector, 'projector') and projector.projector is not None:
            grad = projector.projector(grad)

        return grad

    def _layernorm_grad(
        self,
        layer: nn.LayerNorm,
        idx: int,
        grad_pre_activation: Tensor,
        per_sample: bool = True
    ) -> Tensor:
        """
        Compute the gradient for LayerNorm layers with strict two-step compression:

        STRICT CONVENTION:
        - Stage 1 (Sparsification): ALWAYS factorized, operates on (grad_weight, grad_bias) separately
        - Stage 2 (Projection): NEVER factorized, operates on concatenated gradient

        Forward compression: (g_weight, g_bias) → (g_weight', g_bias') → concat → ĝ
                            [sparsify components]              [project]

        Args:
            layer: LayerNorm layer
            idx: Layer index
            grad_pre_activation: Gradient of the pre-activation
            per_sample: Whether to compute per-sample gradients

        Returns:
            Compressed gradient tensor [batch, k_l] or [batch, 2*d] if no compression
        """
        if not layer.elementwise_affine:
            return None

        normalized = self.normalized[idx]
        if normalized is None:
            return None

        is_3d = normalized.dim() == 3

        # Get sparsifier and projector for this layer
        sparsifier = self.sparsifiers[idx] if hasattr(self, 'sparsifiers') and idx < len(self.sparsifiers) else None
        projector = self.projectors[idx] if hasattr(self, 'projectors') and idx < len(self.projectors) else None

        # Check if we have sparsification (ALWAYS factorized)
        has_sparsifier = (
            sparsifier and
            hasattr(sparsifier, 'sparsifier_comp') and
            sparsifier.sparsifier_comp != (None, None)
        )

        # Check if we have projection (NEVER factorized)
        has_projector = (
            projector and
            hasattr(projector, 'projector') and
            projector.projector is not None
        )

        # Compute weight and bias gradients
        if per_sample:
            grad_pre_activation = grad_pre_activation * normalized.shape[0]
            if is_3d:
                grad_weight = torch.einsum("ijk,ijk->ik", grad_pre_activation, normalized)
                grad_bias = torch.sum(grad_pre_activation, dim=1)
            else:
                grad_weight = grad_pre_activation * normalized
                grad_bias = grad_pre_activation
        else:
            if is_3d:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=(0, 1))
                grad_bias = torch.sum(grad_pre_activation, dim=(0, 1))
            else:
                grad_weight = torch.sum(grad_pre_activation * normalized, dim=0)
                grad_bias = torch.sum(grad_pre_activation, dim=0)

        # STRICT TWO-STEP COMPRESSION:
        # Stage 1: Apply sparsification (ALWAYS factorized, operates on components)
        if has_sparsifier:
            sparsifier_weight, sparsifier_bias = sparsifier.sparsifier_comp
            grad_weight = sparsifier_weight(grad_weight)
            grad_bias = sparsifier_bias(grad_bias)

        # Concatenate weight and bias gradients
        grad = torch.cat((grad_weight, grad_bias), dim=1)

        # Stage 2: Apply projection (NEVER factorized, operates on concatenated gradient)
        if has_projector:
            grad = projector.projector(grad)

        return grad

    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.forward_hooks:
            if hook is not None:
                hook.remove()
        for hook in self.backward_hooks:
            if hook is not None:
                hook.remove()
        self.forward_hooks = [None] * len(self.layer_names)
        self.backward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False