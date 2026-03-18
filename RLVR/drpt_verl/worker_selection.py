"""
Selection-aware FSDP worker for verl.

This module provides a modified ActorRolloutRefWorker that uses
DataParallelPPOActorWithSelection for Layerwise/Subset gradient-based selection.

Usage:
    In your training script, use SelectionActorRolloutRefWorker instead of
    the standard ActorRolloutRefWorker. After init_workers(), call
    setup_selection() with the selection config dict.

Selection config keys (passed via setup_selection):
    - method: "Layerwise", "Subset", or "NA" (default)
    - frac: Fraction of training samples to select (default 0.5)
    - use_second_order: Use similarity matrix for greedy selection (default False)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Optional
    from omegaconf import DictConfig

import torch
from omegaconf import open_dict

from verl import DataProto
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.single_controller.base.decorator import register, make_nd_compute_dataproto_dispatch_fn, Dispatch
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id

logger = logging.getLogger(__name__)


class SelectionActorRolloutRefWorker(BaseActorRolloutRefWorker):
    """
    Extended ActorRolloutRefWorker with gradient-based data selection.

    This worker uses DataParallelPPOActorWithSelection instead of
    DataParallelPPOActor when selection is enabled. The selection-aware
    actor handles Layerwise/Subset selection internally in update_policy().

    Usage:
    1. Create worker as usual
    2. Call setup_selection() with selection config dict after init
    3. Call capture_validation_gradients() before each training epoch
    4. Training with selection happens automatically in update_policy()
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        """Initialize the worker (selection setup happens later via setup_selection)."""
        # Selection will be set up later via setup_selection()
        self._selection_enabled = False
        self._selection_config = None

        # Call parent constructor
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def setup_selection(self, selection_config: Dict[str, Any]) -> None:
        """
        Set up gradient-based selection on the actor.

        This should be called after init_workers() and before training.

        Args:
            selection_config: Dict with keys:
                - method: "Layerwise", "Subset", or "NA"
                - frac: Selection fraction (0.0-1.0)
                - use_second_order: Use similarity matrix for greedy selection
        """
        if not self._is_actor:
            return

        self._selection_config = selection_config
        method = selection_config.get("method", "NA")

        if method not in ["Layerwise", "Subset"]:
            self._selection_enabled = False
            return

        self._selection_enabled = True

        # Upgrade actor to selection-aware version
        self._upgrade_to_selection_actor(selection_config)

    def _upgrade_to_selection_actor(self, selection_config: Dict[str, Any]) -> None:
        """Replace the actor with DataParallelPPOActorWithSelection."""
        try:
            from .dp_actor_selection import DataParallelPPOActorWithSelection
        except ImportError as e:
            logger.error(f"Failed to import DataParallelPPOActorWithSelection: {e}")
            logger.warning("Falling back to standard actor without selection")
            self._selection_enabled = False
            return

        # Get actor config
        actor_cfg = omega_conf_to_dataclass(self.config.actor)

        # Create selection-aware actor, reusing existing optimizer
        self.actor = DataParallelPPOActorWithSelection(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
            selection_method=selection_config.get("method", "NA"),
            train_selection_ratio=selection_config.get("frac", 0.5),
            use_second_order=selection_config.get("use_second_order", False),
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def capture_validation_gradients(self, val_batch: DataProto) -> DataProto:
        """
        Capture validation gradients from external validation batch.

        This should be called once per epoch with a held-out validation set.
        The validation gradients are stored and used for all training batches.

        Args:
            val_batch: Validation batch DataProto containing:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - position_ids: [batch, seq_len] (optional, will be computed if missing)
                - responses: [batch, response_len]
                - rewards: [batch]

        Returns:
            DataProto with validation statistics
        """
        if not self._is_actor or not self._selection_enabled:
            return DataProto(meta_info={"val_loss": 0.0})

        if not hasattr(self.actor, 'capture_validation_gradients_external'):
            logger.warning("Actor does not support external validation gradient capture")
            return DataProto(meta_info={"val_loss": 0.0})

        # Move data to device
        device = get_device_id()
        val_batch = val_batch.to(device)

        # Prepare position_ids if not present
        if 'position_ids' not in val_batch.batch:
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(val_batch.batch['attention_mask'])
            val_batch.batch['position_ids'] = position_ids

        # Get temperature from config (default 1.0)
        temperature = self.config.actor.get('temperature', 1.0)

        # Get GRPO normalization parameters from meta_info (matching training)
        n_responses = val_batch.meta_info.get('n_responses', 1)
        norm_adv_by_std = val_batch.meta_info.get('norm_adv_by_std', True)
        val_loss_type = val_batch.meta_info.get('val_loss_type', 'reward')

        # Call actor's validation gradient capture
        stats = self.actor.capture_validation_gradients_external(
            val_input_ids=val_batch.batch['input_ids'],
            val_attention_mask=val_batch.batch['attention_mask'],
            val_position_ids=val_batch.batch['position_ids'],
            val_responses=val_batch.batch['responses'],
            val_rewards=val_batch.batch['rewards'],
            temperature=temperature,
            n_responses=n_responses,
            norm_adv_by_std=norm_adv_by_std,
            val_loss_type=val_loss_type,
        )

        # Stats are now synchronized across ranks via all-reduce in actor
        # Return only rank-invariant stats to avoid DataProto.concat conflicts
        return DataProto(meta_info={
            'val/loss': stats.get('val/loss', 0.0),
            'val/num_samples': stats.get('val/num_samples', 0),
            'val/num_layers_captured': stats.get('val/num_layers_captured', 0),
            'val/world_size': stats.get('val/world_size', 1),
            'val/loss_type': stats.get('val/loss_type', 'reward'),
            'val/num_correct': stats.get('val/num_correct', -1),
            'val/num_samples_used': stats.get('val/num_samples_used', 0),
        })

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_validation_gradients(self) -> None:
        """Clear cached validation gradients."""
        if self._is_actor and hasattr(self.actor, 'clear_validation_gradients'):
            self.actor.clear_validation_gradients()

    # Forward methods from parent that use @register with DIRECT_ROLLOUT_METHOD
    # These need to be re-registered in subclass for Ray to find them

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        """Forward to parent's get_zeromq_address for vLLM hybrid mode."""
        return self.rollout.get_zeromq_address()
