"""
Hook manager with monkey-patching and custom autograd Functions.

The hook supports two distinct selection methods via the selection module:
- Layerwise: Per-layer selection, single-pass (LayerwiseLinearBackward)
- Subset: Global selection, two-pass (SubsetLinearBackward)

Compression is configured independently for two purposes:
- Score compression: compresses gradients for influence score computation
- Update compression: compresses gradients for MeSO optimizer updates

See CompressionMode for the full mode matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple
    from torch import Tensor
    from .compressor import Compressor

import torch.nn as nn
import torch.nn.functional as F
import functools
import logging

# Compression mode configuration
from .compression_mode import CompressionMode

# Validation gradient cache
from .validation_cache import ValidationCache

# Selection module
from .selection.state import SelectionState, LayerwiseState, SubsetState
from .selection.backward import (
    CompressedLinearBackward,
    LayerwiseLinearBackward,
    SubsetLinearBackward
)

logger = logging.getLogger(__name__)


class GradientHook:
    """
    Hook manager for custom gradient computation.

    This class manages:
    1. Monkey-patching Linear layers for custom backward passes
    2. Compression configuration (via CompressionMode)
    3. Selection state management (Layerwise/Subset)
    4. Validation gradient caching (for separate-batch strategies)
    5. Token count tracking for proper gradient scaling
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cpu',
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to hook (only Linear layers supported)
            device: Device for synchronization
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

        # Separate compressor lists for score computation and optimizer updates.
        # When both use the same config, they share the same Compressor objects.
        self.score_compressors: List[Optional[Compressor]] = [None] * len(layer_names)
        self.update_compressors: List[Optional[Compressor]] = [None] * len(layer_names)

        # Track hook registration status
        self.hooks_registered: bool = False
        self.hooks_enabled: bool = True

        # Selection state (set by trainer before forward/backward)
        self.selection_state: Optional[SelectionState] = None

        # Validation gradient cache (consolidated from three separate buffers)
        self._val_cache: ValidationCache = ValidationCache(len(layer_names))

        # Token count tracking for proper gradient scaling
        # Kept as Tensors to avoid D2H memory copies during backward
        self.total_tokens: Optional[Tensor] = None
        self.tokens_per_sample: Optional[Tensor] = None

        # Register hooks
        self._register_hooks()

        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    # =========================================================================
    # Compression Mode Configuration
    # =========================================================================

    @property
    def compression_mode(self) -> CompressionMode:
        """Derive compression mode from which compressor lists are populated."""
        has_score = any(c is not None for c in self.score_compressors)
        has_update = any(c is not None for c in self.update_compressors)
        if has_score and has_update:
            return CompressionMode.FULL
        elif has_score:
            return CompressionMode.SCORE_ONLY
        elif has_update:
            return CompressionMode.UPDATE_ONLY
        else:
            return CompressionMode.NONE

    # =========================================================================
    # Hook Registration
    # =========================================================================

    def _register_hooks(self):
        """Monkey-patch Linear layers to use our custom Function."""
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
        Replacement forward method that uses our custom autograd Function.

        Routing logic:
        1. If hooks disabled -> call original forward method
        2. If Subset state -> SubsetLinearBackward (score accumulation)
        3. If Layerwise state -> LayerwiseLinearBackward (per-layer selection)
        4. If capture_val_mode -> LayerwiseLinearBackward (val gradient capture)
        5. If compressor present -> CompressedLinearBackward (compression only)
        6. Otherwise -> call original forward method
        """
        if not self.hooks_enabled:
            # Use original forward method to preserve any layer-specific behavior
            # This is important for LoRA layers where we hook lora_A/lora_B Linear modules
            return module._original_forward(input)

        # Route based on selection state type
        state = self.selection_state

        if isinstance(state, SubsetState):
            return SubsetLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, LayerwiseState):
            return LayerwiseLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.capture_val_mode:
            # Val capture mode: use LayerwiseLinearBackward
            return LayerwiseLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.update_compressors[idx] is not None:
            # Update compression only, no data selection (MeSO without selection)
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            # No selection and no compression: use original forward
            return module._original_forward(input)

    def set_score_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set compressors for influence score computation."""
        self.score_compressors = compressors

    def set_update_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set compressors for MeSO optimizer updates."""
        self.update_compressors = compressors

    def set_compressors(self, compressors: List[Optional[Compressor]]) -> None:
        """Set same compressors for both scoring and updates (shared objects)."""
        self.score_compressors = compressors
        self.update_compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute custom gradients."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    # =========================================================================
    # Selection State Management
    # =========================================================================

    def setup_selection(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        scoring_method: str = "ghost",
    ) -> None:
        """
        Set up selection state for gradient streaming.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method ("Layerwise", "Subset", or "Regular")
            frac: Selection fraction (topk) or filter fraction (filtering)
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (Subset pass 1)
            use_second_order: If True, use greedy selection with similarity matrix
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            scoring_method: For Subset: "ghost" (factored inner product) or "direct"
                           (explicit per-sample gradient materialization, Algorithm 4.4)
        """
        if selection_method == "Regular":
            self.selection_state = None
            logger.debug("Set up baseline mode (no selection)")
            return

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "Subset":
            self.selection_state = SubsetState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                scoring_method=scoring_method,
            )
        elif selection_method == "Layerwise":
            self.selection_state = LayerwiseState(
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
            raise ValueError(
                f"Unknown selection_method: {selection_method}. "
                f"Use 'Layerwise', 'Subset', or 'Regular'."
            )

        logger.debug(
            f"Set up {selection_method} state: {train_batch_size} train, "
            f"scores_only={compute_scores_only}, use_second_order={use_second_order}, "
            f"selection_mode={selection_mode}, frac={frac}"
        )

    def clear_selection(self) -> None:
        """Clear selection state after forward/backward."""
        self.selection_state = None

    # =========================================================================
    # Token Count Tracking
    # =========================================================================

    def set_token_counts(
        self,
        labels: Tensor,
        train_batch_size: Optional[int] = None,
    ) -> None:
        """
        Set token counts for proper gradient scaling.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored positions.
                    Used for gradient scaling (only response tokens count).
            train_batch_size: If provided, only count tokens for first train_batch_size samples
        """
        valid_mask = (labels != -100)
        tokens_per_sample = valid_mask.sum(dim=1)

        # Keep as Tensor to avoid D2H memory copy
        self.total_tokens = tokens_per_sample.sum()
        self.tokens_per_sample = tokens_per_sample

        if train_batch_size is not None and self.selection_state is not None:
            train_tokens = tokens_per_sample[:train_batch_size]
            total_train_tokens = train_tokens.sum()
            self.selection_state.set_token_counts(
                train_tokens, total_train_tokens, self.total_tokens
            )

        logger.debug(f"Set token counts: total={self.total_tokens}")

    def clear_token_counts(self) -> None:
        """Clear token counts after forward/backward."""
        self.total_tokens = None
        self.tokens_per_sample = None

    # =========================================================================
    # Validation Gradient Management
    # =========================================================================

    @property
    def val_cache(self) -> ValidationCache:
        """Get the validation gradient cache."""
        return self._val_cache

    @property
    def capture_val_mode(self) -> bool:
        """Check if in validation gradient capture mode."""
        return self._val_cache.capturing

    @property
    def use_factorized_val(self) -> bool:
        """Check if using factorized validation gradient storage."""
        return self._val_cache.is_factorized

    @property
    def val_total_tokens(self) -> Optional[int]:
        """Get total tokens in validation batch."""
        return self._val_cache.total_tokens

    def start_val_capture(self, use_factorized: bool = True) -> None:
        """
        Start capturing validation gradients.

        Args:
            use_factorized: If True, store (grad_output, input) components.
                           If False, store total gradient [O, I] per layer.
        """
        mode = "factorized" if use_factorized else "full"

        # Use compressed storage if score compressors are available
        if self.compression_mode.uses_compressed_scoring:
            mode = "compressed"

        self._val_cache.start_capture(mode=mode)
        logger.debug(f"Started validation gradient capture mode (mode={mode})")

    def end_val_capture(self, val_total_tokens: Optional[int] = None) -> None:
        """
        End validation gradient capture mode.

        Args:
            val_total_tokens: Total valid tokens in validation batch.
        """
        if val_total_tokens is None:
            val_total_tokens = self.total_tokens

        self._val_cache.end_capture(total_tokens=val_total_tokens)

        num_captured = self._val_cache.get_num_captured()
        logger.debug(
            f"Ended validation gradient capture, captured {num_captured} layers, "
            f"val_tokens={self._val_cache.total_tokens}"
        )

    def clear_val_buffer(self) -> None:
        """Clear all validation gradient buffers."""
        self._val_cache.clear()

    def setup_selection_with_stored_val(
        self,
        train_batch_size: int,
        selection_method: str,
        frac: float,
        lr: float,
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "ghost",
    ) -> None:
        """
        Set up selection state using pre-captured validation gradients.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method ("Layerwise" or "Subset")
            frac: Selection/filter fraction
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (Subset pass 1)
            use_second_order: If True, use greedy selection
            selection_mode: "topk" or "filtering"
            record_selections: If True, record selected indices/scores for case study
            scoring_method: For Subset: "ghost" (factored inner product) or "direct"
                           (explicit per-sample gradient materialization, Algorithm 4.4)
        """
        num_captured = self._val_cache.get_num_captured()
        if num_captured == 0:
            raise RuntimeError(
                "No validation gradients captured. Call start_val_capture(), "
                "run forward/backward on validation data, then end_val_capture() first."
            )

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "Subset":
            self.selection_state = SubsetState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                record_selections=record_selections,
                scoring_method=scoring_method,
            )
        elif selection_method == "Layerwise":
            self.selection_state = LayerwiseState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode,
                record_selections=record_selections,
            )
        else:
            raise ValueError(
                f"Unknown selection_method: {selection_method}. "
                f"Use 'Layerwise' or 'Subset'."
            )

        # Mark that we're using stored validation gradients
        self.selection_state._use_stored_val = True

        logger.debug(
            f"Set up selection with stored val gradients: {train_batch_size} train samples, "
            f"{num_captured} layers with val gradients, selection_mode={selection_mode}, frac={frac}"
        )

    # =========================================================================
    # Compressed Gradient Storage (for MeSO optimizer)
    # =========================================================================

    def _get_layer_name_from_idx(self, layer_idx: int) -> str:
        """Get layer name from layer index."""
        return self.layer_names[layer_idx]

    def _get_module_from_idx(self, layer_idx: int) -> nn.Module:
        """Get module from layer index."""
        layer_name = self._get_layer_name_from_idx(layer_idx)
        return self.layer_name_to_module[layer_name]

    def _store_compressed_grad(self, layer_idx: int, compressed_grad: Tensor) -> None:
        """Store compressed gradient on the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        module.weight._compressed_grad = compressed_grad

    def _get_compressed_grad(self, layer_idx: int) -> Optional[Tensor]:
        """Get compressed gradient from the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        return getattr(module.weight, '_compressed_grad', None)

    def _clear_compressed_grad(self, layer_idx: int) -> None:
        """Clear compressed gradient from the weight parameter."""
        module = self._get_module_from_idx(layer_idx)
        if hasattr(module.weight, '_compressed_grad'):
            module.weight._compressed_grad = None

    def clear_all_compressed_grads(self) -> None:
        """Clear all compressed gradients from all layer weight parameters."""
        for layer_idx in range(len(self.layer_names)):
            self._clear_compressed_grad(layer_idx)

    def get_compressed_grads(self) -> List[Optional[Tensor]]:
        """Get all captured compressed gradients."""
        return [self._get_compressed_grad(idx) for idx in range(len(self.layer_names))]

    # =========================================================================
    # Compressor Management
    # =========================================================================

    def refresh_compressors(self, step: int) -> Tuple[int, List[Optional[Any]]]:
        """
        Refresh all compressors if needed.

        Refreshes update_compressors (used by MeSO optimizer for state transformation).
        Score compressors that share the same objects are refreshed implicitly.
        Score-only compressors (not shared with update) are refreshed separately.

        Args:
            step: Current training step

        Returns:
            Tuple of (num_refreshed, old_update_compressors) for optimizer state transform
        """
        num_refreshed = 0
        old_update_compressors = []

        # Refresh update compressors (MeSO optimizer needs old state for transform)
        for idx, compressor in enumerate(self.update_compressors):
            if compressor is not None:
                old_container = compressor.refresh(step)
                old_update_compressors.append(old_container)
                if old_container is not None:
                    num_refreshed += 1
            else:
                old_update_compressors.append(None)

        # Refresh score-only compressors (those not shared with update)
        for idx, compressor in enumerate(self.score_compressors):
            if compressor is not None and compressor is not self.update_compressors[idx]:
                compressor.refresh(step)

        return num_refreshed, old_update_compressors

    def remove_hooks(self) -> None:
        """Restore original forward methods."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        logger.info("Restored original forward methods for all layers")
