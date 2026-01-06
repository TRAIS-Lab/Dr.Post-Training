"""
Activation hook manager for capturing layer activations during forward pass.

Used for RLVR-style difficulty prediction based on activation similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Callable
    from torch import Tensor

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ActivationHook:
    """
    Hook manager for capturing layer activations during forward pass.

    Unlike GradientHook which captures gradients during backward,
    this captures activations (outputs) during forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cpu',
        pooling: str = 'last',  # 'last', 'mean', 'first'
    ) -> None:
        """
        Initialize the activation hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (typically Linear or attention layers)
            device: Device for tensor operations
            pooling: How to pool sequence dimension:
                - 'last': Use last token activation (common for causal LMs)
                - 'mean': Average over sequence
                - 'first': Use first token (CLS-style)
        """
        self.model: nn.Module = model
        self.layer_names: List[str] = layer_names
        self.device: str = device
        self.pooling: str = pooling

        # Create mapping from layer name to index
        self.layer_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(layer_names)
        }

        # Create mapping from layer name to module
        self.layer_name_to_module: Dict[str, nn.Module] = {}

        # Storage for forward hooks
        self.forward_hooks: List[Optional[Any]] = [None] * len(layer_names)

        # Storage for captured activations: [batch_size, hidden_dim] per layer
        self.activations: List[Optional[Tensor]] = [None] * len(layer_names)

        # Reference activations for difficulty prediction
        # Shape: [num_ref, hidden_dim] per layer
        self.ref_activations: List[Optional[Tensor]] = [None] * len(layer_names)

        # Reference difficulties (ground-truth from rollouts)
        # Shape: [num_ref] - shared across all layers
        self.ref_difficulties: Optional[Tensor] = None

        # Track hook registration status
        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True
        self.capture_mode: bool = False  # True when capturing activations

        # Attention mask for proper pooling
        self._current_attention_mask: Optional[Tensor] = None

        # Register hooks
        self._register_hooks()

        logger.info(f"Initialized ActivationHook with {len(layer_names)} layers, pooling={pooling}")

    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_name_to_module[name] = module

                # Register forward hook
                hook = module.register_forward_hook(
                    self._create_forward_hook(idx)
                )
                self.forward_hooks[idx] = hook

        self.hooks_registered = True
        logger.info(f"Successfully registered hooks on {len(self.layer_names)} layers")

    def _create_forward_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            if not self.hooks_enabled or not self.capture_mode:
                return

            # output shape: [batch_size, seq_length, hidden_dim]
            # Pool to [batch_size, hidden_dim]
            pooled = self._pool_activations(output)
            self.activations[layer_idx] = pooled.detach()

        return hook

    def _pool_activations(self, activations: Tensor) -> Tensor:
        """
        Pool sequence dimension based on pooling strategy.

        Args:
            activations: [batch_size, seq_length, hidden_dim]

        Returns:
            Pooled activations: [batch_size, hidden_dim]
        """
        if activations.dim() == 2:
            # Already pooled or no sequence dimension
            return activations

        if self.pooling == 'last':
            # Use last valid token (before padding)
            if self._current_attention_mask is not None:
                # Find last non-padding position
                # attention_mask: [batch_size, seq_length], 1 = valid, 0 = padding
                seq_lengths = self._current_attention_mask.sum(dim=1) - 1  # [batch_size]
                seq_lengths = seq_lengths.clamp(min=0)
                batch_indices = torch.arange(activations.size(0), device=activations.device)
                return activations[batch_indices, seq_lengths.long()]
            else:
                # No mask, use actual last position
                return activations[:, -1, :]

        elif self.pooling == 'mean':
            if self._current_attention_mask is not None:
                # Masked mean
                mask = self._current_attention_mask.unsqueeze(-1)  # [B, S, 1]
                masked_activations = activations * mask
                return masked_activations.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                return activations.mean(dim=1)

        elif self.pooling == 'first':
            return activations[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def set_attention_mask(self, attention_mask: Tensor) -> None:
        """Set attention mask for proper sequence pooling."""
        self._current_attention_mask = attention_mask

    def clear_attention_mask(self) -> None:
        """Clear attention mask."""
        self._current_attention_mask = None

    def enable_hooks(self) -> None:
        """Enable activation capture."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable activation capture."""
        self.hooks_enabled = False

    def start_capture(self) -> None:
        """Start capturing activations."""
        self.capture_mode = True
        self.activations = [None] * len(self.layer_names)

    def end_capture(self) -> None:
        """End capturing activations."""
        self.capture_mode = False

    def get_activations(self, layer_idx: Optional[int] = None) -> List[Optional[Tensor]]:
        """
        Get captured activations.

        Args:
            layer_idx: If provided, return activation for specific layer.
                      Otherwise return all activations.

        Returns:
            List of activation tensors [batch_size, hidden_dim] per layer,
            or single tensor if layer_idx provided.
        """
        if layer_idx is not None:
            return self.activations[layer_idx]
        return self.activations

    def clear_activations(self) -> None:
        """Clear all captured activations."""
        self.activations = [None] * len(self.layer_names)

    # ========================================
    # Reference Set Management
    # ========================================

    def set_reference(
        self,
        ref_activations: List[Tensor],
        ref_difficulties: Tensor
    ) -> None:
        """
        Set reference activations and difficulties for similarity computation.

        Args:
            ref_activations: List of tensors [num_ref, hidden_dim] per layer
            ref_difficulties: Tensor [num_ref] of ground-truth difficulties
        """
        if len(ref_activations) != len(self.layer_names):
            raise ValueError(
                f"Expected {len(self.layer_names)} activation tensors, "
                f"got {len(ref_activations)}"
            )

        self.ref_activations = ref_activations
        self.ref_difficulties = ref_difficulties

        logger.debug(
            f"Set reference: {ref_difficulties.shape[0]} samples, "
            f"{len(ref_activations)} layers"
        )

    def clear_reference(self) -> None:
        """Clear reference activations and difficulties."""
        self.ref_activations = [None] * len(self.layer_names)
        self.ref_difficulties = None

    def has_reference(self) -> bool:
        """Check if reference is set."""
        return (
            self.ref_difficulties is not None and
            any(a is not None for a in self.ref_activations)
        )

    # ========================================
    # Activation Capture Workflow
    # ========================================

    def capture_activations_for_batch(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> List[Tensor]:
        """
        Capture activations for a batch by running forward pass.

        Args:
            model: The model to run forward on
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]

        Returns:
            List of activation tensors [batch_size, hidden_dim] per layer
        """
        self.set_attention_mask(attention_mask)
        self.start_capture()

        with torch.no_grad():
            # Run forward pass to trigger hooks
            model(input_ids=input_ids, attention_mask=attention_mask)

        self.end_capture()
        self.clear_attention_mask()

        # Return captured activations
        activations = [a.clone() if a is not None else None for a in self.activations]
        self.clear_activations()

        return activations

    def capture_reference_activations(
        self,
        model: nn.Module,
        ref_input_ids: Tensor,
        ref_attention_mask: Tensor,
        ref_difficulties: Tensor,
    ) -> None:
        """
        Capture and store reference activations.

        Args:
            model: The model to run forward on
            ref_input_ids: [num_ref, seq_length]
            ref_attention_mask: [num_ref, seq_length]
            ref_difficulties: [num_ref] ground-truth difficulties
        """
        activations = self.capture_activations_for_batch(
            model, ref_input_ids, ref_attention_mask
        )
        self.set_reference(activations, ref_difficulties)

    # ========================================
    # Cleanup
    # ========================================

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self.forward_hooks:
            if hook is not None:
                hook.remove()

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False
        logger.info("Removed all forward hooks")

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
