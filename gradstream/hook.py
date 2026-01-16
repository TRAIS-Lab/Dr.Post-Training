"""
Hook manager with monkey-patching and custom autograd Functions.

The hook supports two distinct selection methods via the selection module:
- Streaming: Per-layer selection, single-pass (StreamingLinearBackward)
- GREATS: Global selection, two-pass (GREATSLinearBackward)

Compression modes (via CompressionMode enum):
- NONE: No compression, full gradients everywhere
- SCORE_ONLY: Compressed scoring, full gradient updates
- FULL: Compressed scoring and gradient updates (MeSO optimizer)
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
from .selection.state import SelectionState, StreamingState, GREATSState
from .selection.backward import (
    CompressedLinearBackward,
    StreamingLinearBackward,
    GREATSLinearBackward
)

logger = logging.getLogger(__name__)


class GradientHook:
    """
    Hook manager for custom gradient computation.

    This class manages:
    1. Monkey-patching Linear layers for custom backward passes
    2. Compression configuration (via CompressionMode)
    3. Selection state management (Streaming/GREATS)
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

        # Unified compressors (one per layer)
        self.compressors: List[Optional[Compressor]] = [None] * len(layer_names)

        # Compression mode configuration
        self._compression_mode: CompressionMode = CompressionMode.NONE

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
        """Get the current compression mode."""
        return self._compression_mode

    @compression_mode.setter
    def compression_mode(self, mode: CompressionMode) -> None:
        """
        Set the compression mode.

        Args:
            mode: CompressionMode enum value

        Raises:
            ValueError: If FULL mode is set but no compressors are configured
        """
        if mode == CompressionMode.FULL:
            if not any(c is not None for c in self.compressors):
                raise ValueError(
                    "Cannot set FULL compression mode without compressors. "
                    "Call set_compressors() first."
                )
        self._compression_mode = mode
        logger.debug(f"Set compression mode to {mode.value}")

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

        # Counter to verify hooks are being called
        self._hook_call_count = 0
        self._hook_call_count_by_layer = {}

    def _custom_linear_forward(self, module: nn.Linear, idx: int, input: Tensor) -> Tensor:
        """
        Replacement forward method that uses our custom autograd Function.

        Routing logic:
        1. If hooks disabled -> call original forward method
        2. If GREATS state -> GREATSLinearBackward (score accumulation)
        3. If Streaming state -> StreamingLinearBackward (per-layer selection)
        4. If capture_val_mode -> StreamingLinearBackward (val gradient capture)
        5. If compressor present -> CompressedLinearBackward (compression only)
        6. Otherwise -> call original forward method
        """
        # Track hook calls
        self._hook_call_count += 1
        layer_name = self.layer_names[idx]
        self._hook_call_count_by_layer[layer_name] = self._hook_call_count_by_layer.get(layer_name, 0) + 1

        if not self.hooks_enabled:
            # Use original forward method to preserve any layer-specific behavior
            # This is important for LoRA layers where we hook lora_A/lora_B Linear modules
            return module._original_forward(input)

        # Route based on selection state type
        state = self.selection_state

        if isinstance(state, GREATSState):
            return GREATSLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, StreamingState):
            return StreamingLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.capture_val_mode:
            # Val capture mode: use StreamingLinearBackward
            return StreamingLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.compressors[idx] is not None:
            # Compression only (no data selection)
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            # No selection and no compression: use original forward
            return module._original_forward(input)

    def set_compressors(self, compressors: List[Compressor]) -> None:
        """Set unified compressor objects for each layer."""
        self.compressors = compressors

    def enable_hooks(self) -> None:
        """Enable hooks to compute custom gradients."""
        self.hooks_enabled = True

    def disable_hooks(self) -> None:
        """Disable hooks to allow standard gradient computation."""
        self.hooks_enabled = False

    def get_hook_stats(self) -> dict:
        """Get hook call statistics for verification."""
        return {
            'total_calls': getattr(self, '_hook_call_count', 0),
            'unique_layers_called': len(getattr(self, '_hook_call_count_by_layer', {})),
            'total_layers': len(self.layer_names),
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
        compute_scores_only: bool = False,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ) -> None:
        """
        Set up selection state for gradient streaming.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method ("Streaming", "GREATS", or "Regular")
            frac: Selection fraction (topk) or filter fraction (filtering)
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (GREATS pass 1)
            use_second_order: If True, use greedy selection with similarity matrix
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        """
        if selection_method == "Regular":
            self.selection_state = None
            logger.debug("Set up baseline mode (no selection)")
            return

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "GREATS":
            self.selection_state = GREATSState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode
            )
        elif selection_method == "Streaming":
            self.selection_state = StreamingState(
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
                f"Use 'Streaming', 'GREATS', or 'Regular'."
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

    def set_token_counts(self, labels: Tensor, train_batch_size: Optional[int] = None) -> None:
        """
        Set token counts for proper gradient scaling.

        Args:
            labels: Label tensor [batch_size, seq_length] with -100 for ignored positions
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
            self.selection_state.set_token_counts(train_tokens, total_train_tokens, self.total_tokens)

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

        # Check if compression is enabled - use compressed storage
        if self._compression_mode.uses_compression:
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
        selection_mode: str = "topk"
    ) -> None:
        """
        Set up selection state using pre-captured validation gradients.

        Args:
            train_batch_size: Number of training samples
            selection_method: Selection method ("Streaming" or "GREATS")
            frac: Selection/filter fraction
            lr: Learning rate for score scaling
            compute_scores_only: If True, only compute scores (GREATS pass 1)
            use_second_order: If True, use greedy selection
            selection_mode: "topk" or "filtering"
        """
        num_captured = self._val_cache.get_num_captured()
        if num_captured == 0:
            raise RuntimeError(
                "No validation gradients captured. Call start_val_capture(), "
                "run forward/backward on validation data, then end_val_capture() first."
            )

        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "GREATS":
            self.selection_state = GREATSState(
                train_batch_size=train_batch_size,
                num_layers=num_layers,
                frac=frac,
                lr=lr,
                device=self.device,
                dtype=dtype,
                use_second_order=use_second_order,
                selection_mode=selection_mode
            )
        elif selection_method == "Streaming":
            self.selection_state = StreamingState(
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
                f"Use 'Streaming' or 'GREATS'."
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
        """Restore original forward methods."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        logger.info("Restored original forward methods for all layers")
