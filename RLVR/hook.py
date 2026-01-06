"""
Unified RLVR Hook for activation-based data selection.

This module provides a unified hook manager that handles:
1. Activation capture during forward pass
2. Reference set management (activations + difficulties from rollouts)
3. Difficulty prediction using activation similarity
4. Selection computation (GREATS: global, Streaming: per-layer)
5. Integration with backward pass (for Streaming mode)

Selection Methods:
- GREATS: Online per-batch selection, filter batch before loss
- Streaming: Per-layer selection, filter gradients during backward
- Regular: No selection (baseline)
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional
    from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from RLVR.difficulty import (
    SimpleCosineSimilarityPredictor,
    DifficultyPredictor,
    RLVRState,
    RLVRStreamingState,
    RLVRStreamingLinearBackward,
)

logger = logging.getLogger(__name__)


class RLVRHook:
    """
    Unified hook manager for RLVR activation-based selection.

    Responsibilities:
    1. Capture activations during forward pass (via forward hooks)
    2. Manage reference set (activations + difficulties from rollouts)
    3. Compute per-sample/per-layer difficulties using activation similarity
    4. Compute and store selection indices (after forward, before backward)
    5. Route to RLVR backward functions (for Streaming mode)

    Usage:
        # Initialize
        hook = RLVRHook(model, layer_names, device='cuda')

        # Set reference from rollouts (per rollout generation)
        hook.capture_reference(ref_input_ids, ref_attention_mask, ref_difficulties)

        # GREATS mode: filter batch before loss
        hook.setup_selection('GREATS', batch_size, alpha=0.5, tau=0.1)
        hook.start_capture(attention_mask)
        outputs = model(**batch)
        hook.end_capture()
        hook.compute_selections()
        selected_indices = hook.get_global_selection()
        loss = compute_loss(outputs[selected_indices], labels[selected_indices])

        # Streaming mode: per-layer selection during backward
        hook.setup_selection('Streaming', batch_size, alpha=0.5, tau=0.1)
        hook.start_capture(attention_mask)
        outputs = model(**batch)
        hook.end_capture()
        hook.compute_selections()  # Stores per-layer selections
        loss = compute_loss(outputs, labels)
        loss.backward()  # RLVRStreamingLinearBackward uses stored selections
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        pooling: str = 'last',
        predictor_type: str = 'simple',
        predictor_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the RLVR hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to capture activations from
                        (typically all MLP/trainable layers)
            device: Device for tensor operations
            dtype: Data type for tensors
            pooling: How to pool sequence dimension:
                - 'last': Use last valid token (common for causal LMs)
                - 'mean': Average over sequence
                - 'first': Use first token (CLS-style)
            predictor_type: Type of difficulty predictor:
                - 'simple': Cosine similarity (no parameters)
                - 'learned': MLP projection + attention
            predictor_kwargs: Additional kwargs for difficulty predictor
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.dtype = dtype
        self.pooling = pooling
        self.num_layers = len(layer_names)

        # Layer mappings
        self.layer_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(layer_names)
        }
        self.layer_name_to_module: Dict[str, nn.Module] = {}

        # Forward hooks for activation capture
        self.forward_hooks: List[Any] = []

        # Activation storage: [batch_size, hidden_dim] per layer
        self.activations: List[Optional[Tensor]] = [None] * self.num_layers

        # Reference set (updated per rollout generation)
        self.ref_activations: List[Optional[Tensor]] = [None] * self.num_layers
        self.ref_difficulties: Optional[Tensor] = None

        # RLVR selection storage (computed after forward, used in backward)
        self.rlvr_layer_selections: List[Optional[Tensor]] = [None] * self.num_layers
        self.rlvr_global_selection: Optional[Tensor] = None
        self.rlvr_selection_stats: Dict = {}

        # Selection state
        self.selection_state: Optional[RLVRState] = None
        self.selection_method: str = 'Regular'  # 'Streaming', 'GREATS', 'Regular'

        # Difficulty predictor
        predictor_kwargs = predictor_kwargs or {}
        if predictor_type == 'simple':
            self.difficulty_predictor = SimpleCosineSimilarityPredictor(
                tau=predictor_kwargs.get('predictor_tau', 1.0)
            )
        elif predictor_type == 'learned':
            hidden_dim = self._get_hidden_dim()
            self.difficulty_predictor = DifficultyPredictor(
                hidden_dim=hidden_dim,
                proj_dim=predictor_kwargs.get('proj_dim', 256),
                tau=predictor_kwargs.get('predictor_tau', 1.0),
            ).to(device)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        # Capture mode
        self.capture_mode = False
        self._attention_mask: Optional[Tensor] = None

        # Token counts for gradient scaling (Streaming mode)
        self.total_tokens: Optional[int] = None
        self.tokens_per_sample: Optional[Tensor] = None

        # Hook registration status
        self.hooks_registered = False
        self.hooks_enabled = True

        # Register forward hooks
        self._register_forward_hooks()

        logger.info(
            f"Initialized RLVRHook with {self.num_layers} layers, "
            f"predictor={predictor_type}, pooling={pooling}"
        )

    def _get_hidden_dim(self) -> int:
        """Get hidden dimension from first Linear layer."""
        for name in self.layer_names:
            for n, module in self.model.named_modules():
                if n == name and isinstance(module, nn.Linear):
                    return module.in_features
        raise ValueError("Could not determine hidden dimension from layer names")

    # ==========================================
    # Forward Hook Registration
    # ==========================================

    def _register_forward_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        if self.hooks_registered:
            logger.warning("Hooks already registered, skipping")
            return

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_name_to_module[name] = module

                # Register forward hook for activation capture
                hook = module.register_forward_hook(
                    self._create_capture_hook(idx)
                )
                self.forward_hooks.append(hook)

        self.hooks_registered = True
        logger.info(f"Registered forward hooks on {len(self.forward_hooks)} layers")

    def _create_capture_hook(self, layer_idx: int) -> Callable:
        """Create forward hook for activation capture."""
        def hook(module: nn.Module, input: tuple, output: Tensor) -> None:
            if not self.hooks_enabled or not self.capture_mode:
                return

            # Pool sequence dimension: [B, S, H] -> [B, H]
            pooled = self._pool_activations(output)
            self.activations[layer_idx] = pooled.detach()

        return hook

    def _pool_activations(self, activations: Tensor) -> Tensor:
        """
        Pool sequence dimension based on pooling strategy.

        Args:
            activations: [batch_size, seq_length, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            [batch_size, hidden_dim]
        """
        if activations.dim() == 2:
            return activations

        if self.pooling == 'last':
            if self._attention_mask is not None:
                # Use last valid token (before padding)
                seq_lengths = self._attention_mask.sum(dim=1) - 1
                seq_lengths = seq_lengths.clamp(min=0)
                batch_indices = torch.arange(activations.size(0), device=activations.device)
                return activations[batch_indices, seq_lengths.long()]
            return activations[:, -1, :]

        elif self.pooling == 'mean':
            if self._attention_mask is not None:
                mask = self._attention_mask.unsqueeze(-1)  # [B, S, 1]
                return (activations * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return activations.mean(dim=1)

        elif self.pooling == 'first':
            return activations[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    # ==========================================
    # Activation Capture Control
    # ==========================================

    def start_capture(self, attention_mask: Optional[Tensor] = None) -> None:
        """
        Start activation capture mode.

        Args:
            attention_mask: [batch_size, seq_length] for proper pooling
        """
        self.capture_mode = True
        self._attention_mask = attention_mask
        self.activations = [None] * self.num_layers

    def end_capture(self) -> None:
        """End activation capture mode."""
        self.capture_mode = False
        self._attention_mask = None

    def enable_hooks(self) -> None:
        """Enable activation capture hooks."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable activation capture hooks."""
        self.hooks_enabled = False

    def get_activations(self, layer_idx: Optional[int] = None) -> List[Optional[Tensor]]:
        """Get captured activations."""
        if layer_idx is not None:
            return self.activations[layer_idx]
        return self.activations

    def clear_activations(self) -> None:
        """Clear captured activations."""
        self.activations = [None] * self.num_layers

    # ==========================================
    # Reference Set Management
    # ==========================================

    def set_reference(
        self,
        ref_activations: List[Tensor],
        ref_difficulties: Tensor,
    ) -> None:
        """
        Set reference activations and difficulties directly.

        Args:
            ref_activations: List of [num_ref, hidden_dim] per layer
            ref_difficulties: [num_ref] ground-truth difficulties (0-1)
        """
        self.ref_activations = [
            a.to(self.device) if a is not None else None
            for a in ref_activations
        ]
        self.ref_difficulties = ref_difficulties.to(self.device)

        logger.debug(
            f"Set reference: {ref_difficulties.shape[0]} samples, "
            f"{len(ref_activations)} layers"
        )

    def capture_reference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        difficulties: Tensor,
    ) -> None:
        """
        Capture reference activations via forward pass.

        Called per rollout generation to update reference set.

        Args:
            input_ids: [num_ref, seq_length]
            attention_mask: [num_ref, seq_length]
            difficulties: [num_ref] ground-truth difficulties (e.g., rollout rewards)
        """
        self.start_capture(attention_mask)
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.end_capture()

        # Store reference activations
        self.ref_activations = [
            a.clone() if a is not None else None
            for a in self.activations
        ]

        # Normalize difficulties to [0, 1] if needed
        if difficulties.min() < 0 or difficulties.max() > 1:
            difficulties = (difficulties - difficulties.min()) / (
                difficulties.max() - difficulties.min() + 1e-8
            )
        self.ref_difficulties = difficulties.to(self.device)

        self.activations = [None] * self.num_layers

        logger.debug(
            f"Captured reference: {difficulties.shape[0]} samples, "
            f"{sum(1 for a in self.ref_activations if a is not None)} layers"
        )

    def has_reference(self) -> bool:
        """Check if reference is set."""
        return (
            self.ref_difficulties is not None and
            any(a is not None for a in self.ref_activations)
        )

    def clear_reference(self) -> None:
        """Clear reference set."""
        self.ref_activations = [None] * self.num_layers
        self.ref_difficulties = None

    # ==========================================
    # Selection Setup
    # ==========================================

    def setup_selection(
        self,
        selection_method: str,
        train_batch_size: int,
        alpha: float = 0.5,
        tau: float = 0.1,
        selection_ratio: float = 0.5,
        selection_mode: str = 'soft',
    ) -> None:
        """
        Set up selection state for RLVR.

        Args:
            selection_method: 'Streaming', 'GREATS', or 'Regular'
            train_batch_size: Number of training samples in batch
            alpha: Target difficulty (0-1, default 0.5 = medium, Goldilocks zone)
            tau: Selection temperature (lower = greedier)
            selection_ratio: Fraction of samples to select
            selection_mode: 'soft' (probabilistic) or 'hard' (top-k)
        """
        self.selection_method = selection_method

        if selection_method == 'Regular':
            self.selection_state = None
            return

        if selection_method == 'GREATS':
            self.selection_state = RLVRState(
                train_batch_size=train_batch_size,
                alpha=alpha,
                tau=tau,
                selection_ratio=selection_ratio,
                device=self.device,
                dtype=self.dtype,
                selection_mode=selection_mode,
            )
        elif selection_method == 'Streaming':
            self.selection_state = RLVRStreamingState(
                train_batch_size=train_batch_size,
                num_layers=self.num_layers,
                alpha=alpha,
                tau=tau,
                selection_ratio=selection_ratio,
                device=self.device,
                dtype=self.dtype,
                selection_mode=selection_mode,
            )
        else:
            raise ValueError(
                f"Unknown selection_method: {selection_method}. "
                "Use 'Streaming', 'GREATS', or 'Regular'."
            )

        logger.debug(
            f"Set up {selection_method} selection: batch_size={train_batch_size}, "
            f"alpha={alpha}, tau={tau}, ratio={selection_ratio}"
        )

    def clear_selection(self) -> None:
        """Clear selection state after training step."""
        self.selection_state = None
        self.selection_method = 'Regular'
        self.rlvr_layer_selections = [None] * self.num_layers
        self.rlvr_global_selection = None
        self.rlvr_selection_stats = {}

    # ==========================================
    # Difficulty Computation
    # ==========================================

    def compute_layer_difficulties(self) -> List[Optional[Tensor]]:
        """
        Compute per-layer difficulty predictions using activation similarity.

        Returns:
            List of [batch_size] tensors, one per layer (None if layer has no activations)
        """
        if not self.has_reference():
            raise RuntimeError("No reference set. Call capture_reference() first.")

        difficulties = []

        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                train_act = self.activations[layer_idx]
                ref_act = self.ref_activations[layer_idx]

                if train_act is None or ref_act is None:
                    difficulties.append(None)
                    continue

                # Predict difficulty using activation similarity
                diff = self.difficulty_predictor(
                    train_act, ref_act, self.ref_difficulties
                )
                difficulties.append(diff)

        return difficulties

    def compute_aggregated_difficulty(
        self,
        aggregation: str = 'mean',
    ) -> Tensor:
        """
        Compute aggregated difficulty across layers.

        Args:
            aggregation: 'mean', 'max', or 'min'

        Returns:
            [batch_size] aggregated difficulties
        """
        layer_diffs = self.compute_layer_difficulties()
        valid = [d for d in layer_diffs if d is not None]

        if not valid:
            raise RuntimeError("No valid layer difficulties computed")

        stacked = torch.stack(valid, dim=0)  # [num_layers, batch_size]

        if aggregation == 'mean':
            return stacked.mean(dim=0)
        elif aggregation == 'max':
            return stacked.max(dim=0)[0]
        elif aggregation == 'min':
            return stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    # ==========================================
    # Selection Computation
    # ==========================================

    def compute_selections(self, aggregation: str = 'mean') -> None:
        """
        Compute and store selection indices AFTER forward pass.

        For GREATS: Computes global selection (stored in rlvr_global_selection)
        For Streaming: Computes per-layer selections (stored in rlvr_layer_selections)

        Must be called after forward pass and before backward pass (for Streaming).

        Args:
            aggregation: How to aggregate layer difficulties for GREATS ('mean', 'max', 'min')
        """
        if self.selection_state is None:
            return

        if isinstance(self.selection_state, RLVRStreamingState):
            # Streaming: per-layer selections
            layer_diffs = self.compute_layer_difficulties()
            for layer_idx, diff in enumerate(layer_diffs):
                if diff is not None:
                    indices = self.selection_state.process_layer(diff, layer_idx)
                    self.rlvr_layer_selections[layer_idx] = indices
            self.rlvr_selection_stats = {
                **self.selection_state.get_selection_summary(),
                **self.selection_state.get_difficulty_summary(),
            }
        else:
            # GREATS: global selection
            agg_diff = self.compute_aggregated_difficulty(aggregation)
            self.rlvr_global_selection = self.selection_state.process_batch(agg_diff)
            self.rlvr_selection_stats = self.selection_state.get_stats()

        logger.debug(f"Computed {self.selection_method} selections: {self.rlvr_selection_stats}")

    def get_global_selection(self) -> Optional[Tensor]:
        """Get global selection indices (for GREATS mode)."""
        return self.rlvr_global_selection

    def get_layer_selection(self, layer_idx: int) -> Optional[Tensor]:
        """Get selection indices for a specific layer (for Streaming mode)."""
        return self.rlvr_layer_selections[layer_idx]

    def get_selection_stats(self) -> Dict:
        """Get selection statistics."""
        return self.rlvr_selection_stats

    # ==========================================
    # Token Counts (for gradient scaling in Streaming)
    # ==========================================

    def set_token_counts(self, labels: Tensor) -> None:
        """
        Set token counts for proper gradient scaling.

        For Streaming mode, gradients need to be scaled when using
        a subset of samples.

        Args:
            labels: [batch_size, seq_length] with -100 for ignored positions
        """
        valid_mask = (labels != -100)
        self.tokens_per_sample = valid_mask.sum(dim=1)
        self.total_tokens = self.tokens_per_sample.sum().item()

    def clear_token_counts(self) -> None:
        """Clear token counts."""
        self.total_tokens = None
        self.tokens_per_sample = None

    # ==========================================
    # Streaming Mode: Linear Layer Patching
    # ==========================================

    def patch_linear_layers(self) -> None:
        """
        Monkey-patch Linear layers to use RLVRStreamingLinearBackward.

        For Streaming mode, this enables per-layer gradient selection
        during backward pass.
        """
        for name in self.layer_names:
            if name not in self.layer_name_to_module:
                continue

            module = self.layer_name_to_module[name]
            if not isinstance(module, nn.Linear):
                continue

            idx = self.layer_name_to_idx[name]

            # Save original forward
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward

            # Create wrapped forward
            module.forward = functools.partial(
                self._streaming_linear_forward, module, idx
            )

        logger.info(f"Patched {len(self.layer_name_to_module)} Linear layers for Streaming mode")

    def _streaming_linear_forward(
        self,
        module: nn.Linear,
        layer_idx: int,
        input: Tensor,
    ) -> Tensor:
        """
        Custom forward for Streaming mode.

        Routes to RLVRStreamingLinearBackward when Streaming selection is active.
        """
        if self.selection_method == 'Streaming' and self.selection_state is not None:
            return RLVRStreamingLinearBackward.apply(
                input, module.weight, module.bias, self, layer_idx
            )
        return F.linear(input, module.weight, module.bias)

    def unpatch_linear_layers(self) -> None:
        """Restore original forward methods."""
        for name in self.layer_names:
            if name not in self.layer_name_to_module:
                continue

            module = self.layer_name_to_module[name]
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        logger.info("Restored original Linear layer forwards")

    # ==========================================
    # Cleanup
    # ==========================================

    def remove_hooks(self) -> None:
        """Remove all forward hooks."""
        for hook in self.forward_hooks:
            hook.remove()
        self.forward_hooks = []
        self.hooks_registered = False
        logger.info("Removed all forward hooks")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.remove_hooks()
            self.unpatch_linear_layers()
        except Exception:
            pass


def get_trainable_layer_names(
    model: nn.Module,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Get names of trainable layers (typically MLP/Linear layers).

    For LoRA models, this returns all LoRA-adapted layers.

    Args:
        model: The model to inspect
        include_patterns: Patterns to include (e.g., ['lora_', 'mlp'])
        exclude_patterns: Patterns to exclude (e.g., ['embed', 'lm_head'])

    Returns:
        List of layer names suitable for RLVRHook
    """
    include_patterns = include_patterns or ['lora_', 'mlp', 'Linear']
    exclude_patterns = exclude_patterns or ['embed', 'lm_head', 'norm']

    layer_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check include patterns
        include = any(p in name for p in include_patterns)
        # Check exclude patterns
        exclude = any(p in name for p in exclude_patterns)

        if include and not exclude:
            # Check if module has trainable parameters
            if any(p.requires_grad for p in module.parameters()):
                layer_names.append(name)

    return layer_names


def create_rlvr_hook(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    device: str = 'cuda',
    **kwargs,
) -> RLVRHook:
    """
    Factory function to create RLVR hook.

    Args:
        model: The model to hook
        layer_names: Layer names to hook (auto-detect if None)
        device: Device for tensors
        **kwargs: Additional arguments for RLVRHook

    Returns:
        RLVRHook instance
    """
    if layer_names is None:
        layer_names = get_trainable_layer_names(model)
        logger.info(f"Auto-detected {len(layer_names)} trainable layers")

    return RLVRHook(
        model=model,
        layer_names=layer_names,
        device=device,
        **kwargs,
    )
