"""
Hook manager with monkey-patching and custom autograd Functions.

The hook supports two distinct selection methods via the selection module:
- Streaming: Per-layer selection, single-pass (StreamingLinearBackward)
- GREATS: Global selection, two-pass (GREATSLinearBackward)
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

        # Selection state (set by trainer before forward/backward)
        self.selection_state: Optional[SelectionState] = None

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

        Routes to appropriate autograd function based on selection state type:
        - GREATSState -> GREATSLinearBackward (score accumulation only)
        - StreamingState -> StreamingLinearBackward (per-layer selection)
        - None -> original forward (no interception needed)
        """
        if not self.hooks_enabled:
            return F.linear(input, module.weight, module.bias)

        # Route based on selection state type
        state = self.selection_state

        if isinstance(state, GREATSState):
            # GREATS: score accumulation across layers
            return GREATSLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif isinstance(state, StreamingState):
            # Streaming: per-layer selection
            return StreamingLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        elif self.compressors[idx] is not None:
            # Compression only (no data selection):
            # Use dedicated CompressedLinearBackward for clean separation
            return CompressedLinearBackward.apply(
                input, module.weight, module.bias, self, idx
            )
        else:
            # No selection state and no compression: use standard PyTorch forward
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
                                Used for GREATS first pass (score accumulation).
            use_second_order: If True, compute similarity matrix and use greedy selection
                            with second-order interactions. If False (default), use simple
                            top-k selection which is ~200x faster.
            selection_mode: How to select samples based on scores:
                           - "topk": Select top frac samples by score (SFT style)
                           - "filtering": Keep all positive + drop bottom frac of negative (RLHF style)
        """
        # Regular mode: no selection, use baseline
        if selection_method == "Regular":
            self.selection_state = None
            logger.debug("Set up baseline mode (no selection)")
            return

        # Infer dtype from model parameters
        dtype = next(self.model.parameters()).dtype
        num_layers = len(self.layer_names)

        if selection_method == "GREATS":
            # GREATS: score accumulation across layers
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
            # Streaming: per-layer selection
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
            raise ValueError(f"Unknown selection_method: {selection_method}. Use 'Streaming', 'GREATS', or 'Regular'.")

        logger.debug(f"Set up {selection_method} state: {train_batch_size} train, scores_only={compute_scores_only}, use_second_order={use_second_order}, selection_mode={selection_mode}, frac={frac}, dtype={dtype}")

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
            compute_scores_only: If True, only compute scores (for GREATS first pass)
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

        if selection_method == "GREATS":
            # GREATS: score accumulation across layers
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
            # Streaming: per-layer selection
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
            raise ValueError(f"Unknown selection_method: {selection_method}. Use 'Streaming' or 'GREATS'.")

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

        Stored on weight (not bias) because compressed grad is one tensor per layer.
        Gradient norm should be computed in compressed space using _compressed_grad directly.
        """
        module = self._get_module_from_idx(layer_idx)
        module.weight._compressed_grad = compressed_grad

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
        """Restore original forward methods."""
        for name, module in self.layer_name_to_module.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

        self.forward_hooks = [None] * len(self.layer_names)
        self.hooks_registered = False

        logger.info("Restored original forward methods for all layers")
