"""
Streaming trainer with unified data selection and model update.
"""

import json
import math
import os
import time
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from typing import Dict
from torch import Tensor

from transformers import Trainer


def compute_train_loss_from_merged(logits, labels, train_batch_size):
    """
    Compute loss for train samples only from merged batch forward.

    This allows reporting the train-only loss without an extra forward pass
    when using merged batches (train + val) for data selection.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Labels tensor [batch_size, seq_len]
        train_batch_size: Number of train samples (first N in batch)

    Returns:
        Loss computed only on the train portion of the batch
    """
    # Extract train portion
    train_logits = logits[:train_batch_size]  # [train_bs, seq_len, vocab_size]
    train_labels = labels[:train_batch_size]  # [train_bs, seq_len]

    # Shift for causal LM (predict next token)
    shift_logits = train_logits[..., :-1, :].contiguous()
    shift_labels = train_labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute loss (mean over valid tokens only)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss

from gradstream.optimizer import MeSOAdamW
from gradstream.hook import GradientHook
from gradstream.selection import create_selection_strategy

logger = logging.getLogger(__name__)


class StreamingTrainer(Trainer):
    """
    Trainer supporting gradient-based data selection with optional compression.

    Supports 6 training modes:

    Data Selection Modes (method in ['Streaming', 'GREATS']):
        - Streaming: Single-pass, per-layer selection during backward.
          Validation gradients guide sample selection at each layer independently.
        - GREATS: Two-pass, global selection across all layers.
          First pass computes selection scores, second pass computes gradients
          on the globally selected samples.

    Compression Options (MeSO optimizer):
        - With compression: Uses MeSOAdamW for compressed gradient storage.
        - Without compression: Uses standard AdamW with full gradients.

    Baseline Modes (method == 'NA'):
        - MeSO only: Compressed gradients without data selection.
        - Baseline: Standard training with full gradients.
    """

    def __init__(self, grad_hook: GradientHook, val_dataset, *args, **kwargs):
        """
        Initialize the trainer.

        Args:
            grad_hook: GradientHook instance for gradient capture and compression.
            val_dataset: Small validation set used for data selection during training.
                Batches from this set are merged with training batches to compute
                selection scores that guide which training samples to use.
            *args, **kwargs: Same as transformers.Trainer.
                Must include eval_dataset: Large held-out set for generalization testing.
                This is separate from val_dataset and used only during evaluation.
        """
        # Extract eval_dataset from kwargs before passing to parent
        eval_dataset = kwargs.get('eval_dataset', None)
        if eval_dataset is None:
            raise ValueError("StreamingTrainer requires an eval_dataset to be passed in kwargs")

        # Pass eval_dataset to parent Trainer
        super().__init__(*args, **kwargs)

        # Store our custom datasets
        self.grad_hook = grad_hook
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset

        # Set selection_frac to 1.0 when no selection method is specified (consistency)
        # This means we "select" all samples when not doing data selection
        if not hasattr(self.args, 'selection_frac') or self.args.selection_frac is None:
            if self.args.method == 'NA':
                self.args.selection_frac = 1.0

        # Initialize results tracking
        self.evaluation_results = []

        # Track wall time using CUDA events for accurate GPU timing
        if torch.cuda.is_available():
            self.training_start_event = torch.cuda.Event(enable_timing=True)
            self.training_start_event.record()
        else:
            self.training_start_event = None
            self.training_start_time = time.time()  # Fallback for CPU

        # Determine validation batch size for data selection
        # Use val_batch_size_for_selection if provided, otherwise default to per_device_train_batch_size
        self.val_batch_size_for_selection = (
            self.args.val_batch_size_for_selection
            if hasattr(self.args, 'val_batch_size_for_selection') and self.args.val_batch_size_for_selection is not None
            else self.args.per_device_train_batch_size
        )

        # Create validation dataloader iterator for efficient batch sampling during training
        # Only needed if we're doing gradient streaming (data selection)
        if val_dataset is not None:
            self.val_dataloader_iter = iter(
                self.get_val_dataloader(self.val_dataset, batch_size=self.val_batch_size_for_selection, shuffle=True)
            )
        else:
            self.val_dataloader_iter = None

        # Determine if we have compression (which implies MeSO)
        self.has_compression = (
            self.grad_hook is not None and
            any(c is not None for c in self.grad_hook.compressors)
        )

        # Create selection strategy for clean separation of selection methods
        self.selection_strategy = create_selection_strategy(
            method=self.args.method,
            grad_hook=self.grad_hook,
            frac=getattr(self.args, 'selection_frac', 0.5),
            use_second_order=getattr(self.args, 'use_second_order', False),
            selection_mode="topk",  # SFT uses top-k selection
        )

        logger.info("="*60)
        logger.info("Initialized StreamingTrainer")
        selection_frac = getattr(self.args, 'selection_frac', None)
        logger.info(f"  Method: {self.args.method} (selection fraction: {selection_frac})")
        logger.info(f"  Compression: {self.has_compression}")
        logger.info(f"  Validation set size: {len(val_dataset) if val_dataset is not None else 0}")
        logger.info(f"  Evaluation set size: {len(eval_dataset) if eval_dataset is not None else 0}")
        logger.info(f"  Training batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"  Validation batch size (for selection): {self.val_batch_size_for_selection}")

        # Log the training mode based on configuration
        # Naming convention: {selection}-{compression}-{training_type}
        if self.args.method in ('Streaming', 'GREATS'):
            if self.has_compression:
                logger.info(f"  Mode: {self.args.method} with compression (MeSO optimizer)")
            else:
                logger.info(f"  Mode: {self.args.method} without compression (standard optimizer)")
        elif self.has_compression:
            logger.info(f"  Mode: MeSO only (compressed gradients, no selection)")
        else:
            logger.info(f"  Mode: Baseline (full gradients, no selection)")

        logger.info("="*60)

    def _get_unwrapped_optimizer(self):
        """
        Get the underlying optimizer, unwrapping AcceleratedOptimizer if present.

        Returns:
            The underlying optimizer (e.g., MeSOAdamW or AdamW)
        """
        optimizer = self.optimizer
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        return optimizer

    def create_optimizer(self):
        """
        Setup the optimizer based on compression setting.

        - With compression: Use MeSOAdamW (compressed optimizer states)
        - Without compression: Use standard AdamW
        """
        if self.has_compression:
            logger.info("Using MeSOAdamW optimizer (compression enabled)")

            # Create compressed optimizer
            self.optimizer = MeSOAdamW(
                params=self.model.parameters(),
                grad_hook=self.grad_hook,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        else:
            # Use standard Hugging Face optimizer creation
            logger.info("Using standard AdamW optimizer (no compression)")
            super().create_optimizer()

        return self.optimizer

    def get_val_dataloader(self, val_dataset, batch_size=4, shuffle=True):
        """Create validation dataloader for data selection."""
        if shuffle:
            sampler = RandomSampler(val_dataset)
        else:
            sampler = SequentialSampler(val_dataset)

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _merge_batches(self, batch_train: Dict[str, Tensor], batch_val: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Merge training and validation batches along batch dimension.

        Handles the case where batches have different sequence lengths by padding
        to the maximum length across both batches.

        Args:
            batch_train: Training batch
            batch_val: Validation batch

        Returns:
            Merged batch with train samples first, then val samples
        """
        merged_batch = {}

        # Find max sequence length for padding
        max_seq_len = max(
            batch_train.get('input_ids', batch_train.get('attention_mask')).shape[1],
            batch_val.get('input_ids', batch_val.get('attention_mask')).shape[1]
        )

        for key in batch_train.keys():
            if key not in batch_val:
                # If key not in val batch, just use train batch value
                merged_batch[key] = batch_train[key]
                continue

            train_tensor = batch_train[key]
            val_tensor = batch_val[key]

            # Pad to max sequence length if needed (for 2D tensors like input_ids, attention_mask, labels)
            if train_tensor.dim() == 2:
                train_seq_len = train_tensor.shape[1]
                val_seq_len = val_tensor.shape[1]

                # Determine padding value based on key
                if key == 'attention_mask':
                    pad_value = 0
                elif key == 'labels':
                    pad_value = -100  # Standard ignore index for CrossEntropyLoss
                else:  # input_ids and other token-based fields
                    # Use processing_class's pad_token_id if available, otherwise 0
                    processing_class = getattr(self, 'processing_class', None)
                    pad_value = processing_class.pad_token_id if processing_class is not None and hasattr(processing_class, 'pad_token_id') and processing_class.pad_token_id is not None else 0

                # Pad train batch if needed
                if train_seq_len < max_seq_len:
                    padding = torch.full(
                        (train_tensor.shape[0], max_seq_len - train_seq_len),
                        pad_value,
                        dtype=train_tensor.dtype,
                        device=train_tensor.device
                    )
                    train_tensor = torch.cat([train_tensor, padding], dim=1)

                # Pad val batch if needed
                if val_seq_len < max_seq_len:
                    padding = torch.full(
                        (val_tensor.shape[0], max_seq_len - val_seq_len),
                        pad_value,
                        dtype=val_tensor.dtype,
                        device=val_tensor.device
                    )
                    val_tensor = torch.cat([val_tensor, padding], dim=1)

            # Concatenate along batch dimension (dim=0)
            merged_batch[key] = torch.cat([train_tensor, val_tensor], dim=0)

        return merged_batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Training step using selection strategy pattern.

        The selection strategy handles the difference between:
        - Streaming: Single-pass, per-layer selection
        - GREATS: Two-pass, global selection
        - NA: Baseline (no selection)

        With or without compression (MeSO).
        """
        model.train()
        args = self.args

        # Refresh compressors if needed (only for MeSO)
        if self.has_compression:
            unwrapped_optimizer = self._get_unwrapped_optimizer()
            if isinstance(unwrapped_optimizer, MeSOAdamW):
                unwrapped_optimizer.refresh_compressors_if_needed()

        # === DATA SELECTION MODE (Streaming or GREATS) ===
        if args.method in ('Streaming', 'GREATS'):
            # Get validation batch for selection
            try:
                val_batch = next(self.val_dataloader_iter)
            except StopIteration:
                self.val_dataloader_iter = iter(
                    self.get_val_dataloader(self.val_dataset, batch_size=self.val_batch_size_for_selection, shuffle=True)
                )
                val_batch = next(self.val_dataloader_iter)

            # Prepare inputs
            batch_train = self._prepare_inputs(inputs)
            batch_val = self._prepare_inputs(val_batch)
            train_batch_size = batch_train['input_ids'].shape[0]

            # Merge batches: [train_samples, val_samples]
            merged_batch = self._merge_batches(batch_train, batch_val)

            # Get current learning rate
            lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, 'optimizer') and self.optimizer else args.learning_rate
            if lr is None or lr == 0:
                lr = args.learning_rate

            # Execute selection strategy
            loss = self.selection_strategy.execute_training_step(
                model=model,
                merged_batch=merged_batch,
                train_batch_size=train_batch_size,
                compute_loss_fn=self._compute_loss_for_selection,
                lr=lr,
                batch_train=batch_train  # For GREATS pass 2
            )

            return loss

        # === BASELINE MODE (no data selection) ===
        else:
            return self._training_step_baseline(model, inputs)

    def _compute_loss_for_selection(self, model, batch):
        """Compute loss for selection (handles multi-GPU and grad accumulation)."""
        with self.compute_loss_context_manager():
            outputs = model(**batch)
            loss = outputs.loss

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def _training_step_baseline(self, model, inputs):
        """Baseline training step without selection."""
        args = self.args

        model.zero_grad()

        # Disable hooks if no compression (baseline training)
        if not self.has_compression and self.grad_hook is not None and self.grad_hook.hooks_registered:
            self.grad_hook.disable_hooks()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        # Re-enable hooks if they were disabled
        if not self.has_compression and self.grad_hook is not None and self.grad_hook.hooks_registered:
            self.grad_hook.enable_hooks()

        return loss.detach()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute both validation and evaluation perplexity.

        Computes:
        - Validation perplexity: on the small validation set (used for data selection)
        - Evaluation perplexity: on the large held-out evaluation set (for generalization)

        Args:
            eval_dataset: Dataset to evaluate on (defaults to self.eval_dataset)
            ignore_keys: Keys to ignore in the output
            metric_key_prefix: Prefix for metric keys

        Returns:
            Dict of metrics including val_perplexity and eval_perplexity
        """
        # Disable gradient hooks during evaluation to avoid overhead
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        # Evaluate on validation dataset (small set for data selection)
        val_loss, val_perplexity = self._evaluate_on_dataset(
            self.val_dataset,
            description="Validation"
        )

        # Evaluate on evaluation dataset (large held-out set for generalization)
        eval_loss, eval_perplexity = self._evaluate_on_dataset(
            self.eval_dataset_custom,
            description="Evaluation"
        )

        # Calculate elapsed wall time
        if self.training_start_event is not None:
            # Use CUDA event timing for accurate GPU time measurement
            eval_event = torch.cuda.Event(enable_timing=True)
            eval_event.record()
            torch.cuda.synchronize()
            wall_time = self.training_start_event.elapsed_time(eval_event) / 1000.0  # Convert ms to seconds
        else:
            # Fallback to CPU timing
            wall_time = time.time() - self.training_start_time

        # Create metrics dictionary
        eval_metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_perplexity": eval_perplexity,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "wall_time": wall_time,
        }

        # Save results
        result_entry = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "wall_time": wall_time,
        }
        self.evaluation_results.append(result_entry)
        self._save_evaluation_results()

        logger.info(
            f"Step {self.state.global_step}: "
            f"val_perplexity={val_perplexity:.4f}, "
            f"eval_perplexity={eval_perplexity:.4f}, "
            f"wall_time={wall_time:.2f}s"
        )

        # Re-enable hooks after evaluation if selection method is active OR compression is used
        # Hooks are needed for both: (1) streaming data selection, (2) MeSO compressed gradients
        if self.grad_hook is not None and (self.args.method != "NA" or self.has_compression):
            self.grad_hook.enable_hooks()

        return eval_metrics

    def _evaluate_on_dataset(self, dataset, description="Evaluation"):
        """
        Evaluate model on a given dataset and compute loss and perplexity.

        Args:
            dataset: Dataset to evaluate on
            description: Description for logging

        Returns:
            tuple: (average_loss, perplexity)
        """
        model = self.model
        model.eval()

        logger.debug(f"{description}: Dataset size = {len(dataset)}, Batch size = {self.args.per_device_train_batch_size}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        total_loss = 0.0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Debug: Check for NaN in loss
                if torch.isnan(loss):
                    logger.warning(f"{description}: NaN loss detected in batch {num_batches}!")
                    logger.warning(f"  Batch size: {batch['input_ids'].shape[0]}")
                    logger.warning(f"  Input shape: {batch['input_ids'].shape}")
                    # Check labels
                    if 'labels' in batch:
                        labels = batch['labels']
                        non_ignore_mask = (labels != -100)
                        num_valid_tokens = non_ignore_mask.sum().item()
                        logger.warning(f"  Labels shape: {labels.shape}")
                        logger.warning(f"  Valid tokens (not -100): {num_valid_tokens}")
                        logger.warning(f"  All labels ignored: {num_valid_tokens == 0}")
                        if num_valid_tokens > 0:
                            logger.warning(f"  Sample label values: {labels[non_ignore_mask][:10].tolist()}")
                    # Check outputs
                    logger.warning(f"  Model output loss: {loss}")
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        logger.warning(f"  Logits shape: {logits.shape}")
                        logger.warning(f"  Logits contains NaN: {torch.isnan(logits).any().item()}")
                        logger.warning(f"  Logits contains Inf: {torch.isinf(logits).any().item()}")

                # Accumulate loss
                batch_size = batch["input_ids"].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                num_batches += 1

        logger.debug(f"{description}: Processed {num_batches} batches, {total_samples} samples")

        # Compute average loss and perplexity
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss != float("inf") else float("inf")

        logger.debug(f"{description}: avg_loss={avg_loss:.4f}, perplexity={perplexity:.4f}")

        model.train()

        return avg_loss, perplexity

    def _save_evaluation_results(self):
        """Save evaluation results to JSON file."""
        output_file = os.path.join(self.args.output_dir, "evaluation_results.json")

        # Ensure output directory exists
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

    def on_train_end(self):
        """Called at the end of training to save final results."""
        self._save_evaluation_results()
        logger.info(f"Training completed. Final results saved to {self.args.output_dir}/evaluation_results.json")
