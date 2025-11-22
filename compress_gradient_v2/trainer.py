"""
Hook-based trainer with on-the-fly data selection.
"""

import json
import math
import os
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from typing import Dict
from torch import Tensor

from transformers import Trainer

from .optimizer import MeSOAdamW
from .hook import GradientHook

logger = logging.getLogger(__name__)


class CompGradTrainer(Trainer):
    """
    Trainer with on-the-fly data selection and gradient compression.
    - Single merged forward/backward pass (train + val batches)
    - Selection scores computed incrementally during backward
    - Gradients aggregated on-the-fly for memory efficiency
    """

    def __init__(self, grad_hook: GradientHook, val_dataset, *args, **kwargs):
        """
        Initialize the hook-based trainer.

        Args:
            grad_hook: hook.GradientHook instance
            val_dataset: Validation dataset (for data selection, small)
            *args, **kwargs: Same as transformers.Trainer
                - Must include eval_dataset: Evaluation dataset (for generalization testing, large)
        """
        # Extract eval_dataset from kwargs before passing to parent
        eval_dataset = kwargs.get('eval_dataset', None)
        if eval_dataset is None:
            raise ValueError("CompGradTrainer requires an eval_dataset to be passed in kwargs")

        # Pass eval_dataset to parent Trainer
        super().__init__(*args, **kwargs)

        # Store our custom datasets
        self.grad_hook = grad_hook
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset

        # Set selection_frac to 1.0 when no selection method is specified (consistency)
        # This means we "select" all samples when not doing data selection
        if not hasattr(self.args, 'selection_frac') or self.args.selection_frac is None:
            if self.args.method not in ['GREATS', 'GradNorm', 'MaxLoss']:
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

        # Create validation dataloader iterator for efficient batch sampling during training
        # Only needed if we're doing data selection (GREATS/GradNorm)
        if val_dataset is not None:
            val_batch_size = self.args.per_device_train_batch_size
            self.val_dataloader_iter = iter(
                self.get_val_dataloader(self.val_dataset, batch_size=val_batch_size, shuffle=True)
            )
        else:
            self.val_dataloader_iter = None

        logger.info("="*60)
        logger.info("Initialized CompGradTrainer (v2 - On-the-fly selection)")
        selection_frac = getattr(self.args, 'selection_frac', None)
        logger.info(f"  Selection method: {self.args.method} (fraction: {selection_frac})")
        logger.info(f"  Validation set size: {len(val_dataset) if val_dataset is not None else 0}")
        logger.info(f"  Evaluation set size: {len(eval_dataset) if eval_dataset is not None else 0}")
        logger.info(f"  Compressed optimizer: {self.args.use_compressed_optimizer}")

        # Log the training mode based on configuration
        if val_dataset is not None and self.args.method in ['GREATS', 'GradNorm']:
            if self.args.use_compressed_optimizer:
                logger.info(f"  Mode: Case 3 - GREATS with MeSO (on-the-fly selection + compressed gradients)")
            else:
                logger.info(f"  Mode: Case 2 - GREATS without MeSO (on-the-fly scoring + full gradients)")
        elif self.args.use_compressed_optimizer:
            logger.info(f"  Mode: Case 1 - MeSO without GREATS (compressed gradients, no selection)")
        else:
            logger.info(f"  Mode: Standard training (full gradients, no selection)")

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
        Setup the optimizer with optional compressed state storage.

        If use_compressed_optimizer is True and gradient compression is enabled,
        uses MeSOAdamW which maintains optimizer states in compressed space.
        """
        # Check if we should use compressed optimizer
        use_compressed = (
            self.args.use_compressed_optimizer and
            self.grad_hook is not None and
            len(self.grad_hook.compressors) > 0
        )

        if use_compressed:
            # Check that compressors are actually set up
            has_compressors = any(c is not None for c in self.grad_hook.compressors)

            if not has_compressors:
                logger.warning(
                    "use_compressed_optimizer=True but no compressors found! "
                    "Falling back to standard optimizer. "
                    "Make sure to set sparsification and/or projection arguments."
                )
                use_compressed = False

        if use_compressed:
            logger.info("Using MeSOAdamW optimizer with compressed state storage")

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
            logger.info("Using standard AdamW optimizer")
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
                    # Use tokenizer's pad_token_id if available, otherwise 0
                    processing_class = getattr(self, 'processing_class', getattr(self, 'tokenizer', None))
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
        Training step with on-the-fly data selection.
            1. Merge validation with training batch
            2. Setup selection state in hook
            3. Single forward/backward pass on merged batch
            4. Hook computes scores and aggregates gradients during backward
            5. Retrieve aggregated gradients from hook
            6. Return loss
        """
        model.train()
        args = self.args

        # Refresh compressors if needed
        unwrapped_optimizer = self._get_unwrapped_optimizer()
        using_compressed_optimizer = isinstance(unwrapped_optimizer, MeSOAdamW)

        if using_compressed_optimizer:
            unwrapped_optimizer.refresh_compressors_if_needed()

        # Perform selection based on method
        if args.method in ['GREATS', 'GradNorm']:
            # Get validation batch for selection
            try:
                val_batch = next(self.val_dataloader_iter)
            except StopIteration:
                # Restart iterator when exhausted
                val_batch_size = args.per_device_train_batch_size
                self.val_dataloader_iter = iter(
                    self.get_val_dataloader(self.val_dataset, batch_size=val_batch_size, shuffle=True)
                )
                val_batch = next(self.val_dataloader_iter)

            # Prepare inputs
            batch_train = self._prepare_inputs(inputs)
            batch_val = self._prepare_inputs(val_batch)

            # Get batch sizes
            train_batch_size = batch_train['input_ids'].shape[0]

            # Merge batches: [train_samples, val_samples]
            merged_batch = self._merge_batches(batch_train, batch_val)

            # Get current learning rate from optimizer (handles scheduling)
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                lr = self.optimizer.param_groups[0]["lr"]
                # Handle case where lr might be None (e.g., warmup with 0 initial lr)
                if lr is None or lr == 0:
                    lr = args.learning_rate  # Fall back to base learning rate
            else:
                # Optimizer not created yet, use base learning rate
                lr = args.learning_rate

            if using_compressed_optimizer:
                # On-the-fly selection and aggregation during backward
                # Reuse compressed gradients for optimization

                # Setup selection state in hook with aggregation enabled
                self.grad_hook.setup_selection(
                    train_batch_size=train_batch_size,
                    selection_method=args.method,
                    selection_frac=args.selection_frac,
                    lr=lr,
                    compute_scores_only=False
                )

                # Zero gradients before merged forward/backward
                model.zero_grad()

                # Single forward/backward pass on merged batch
                with self.compute_loss_context_manager():
                    outputs = model(**merged_batch)
                    loss = outputs.loss

                # Apply multi-GPU averaging
                if args.n_gpu > 1:
                    loss = loss.mean()

                # Apply gradient accumulation scaling
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward pass - hook will compute scores and aggregate gradients
                loss.backward()

                # Clear selection state
                self.grad_hook.clear_selection()
                return loss.detach()

            else:
                # Only data selection
                # Compute scores on-the-fly, then do second forward/backward with full gradients

                # Setup selection state in hook with score computation only
                self.grad_hook.setup_selection(
                    train_batch_size=train_batch_size,
                    selection_method=args.method,
                    selection_frac=args.selection_frac,
                    lr=lr,
                    compute_scores_only=True
                )

                # Zero gradients before merged forward/backward
                model.zero_grad()

                # First forward/backward pass on merged batch to compute scores
                with self.compute_loss_context_manager():
                    outputs = model(**merged_batch)
                    loss_for_scoring = outputs.loss

                # Apply multi-GPU averaging
                if args.n_gpu > 1:
                    loss_for_scoring = loss_for_scoring.mean()

                # Apply gradient accumulation scaling
                if args.gradient_accumulation_steps > 1:
                    loss_for_scoring = loss_for_scoring / args.gradient_accumulation_steps

                # Backward pass - hook will maintain running scores (no aggregation)
                loss_for_scoring.backward()

                # Get selected indices based on final accumulated scores
                selected_indices = self.grad_hook.selection_state.get_selected_indices()

                # Clear selection state
                self.grad_hook.clear_selection()

                # Filter inputs to selected samples
                filtered_inputs = {
                    'input_ids': batch_train['input_ids'][selected_indices],
                    'attention_mask': batch_train['attention_mask'][selected_indices],
                    'labels': batch_train['labels'][selected_indices]
                }

                # Zero gradients before optimization forward/backward
                model.zero_grad()

                # Disable hooks for standard gradient computation
                self.grad_hook.disable_hooks()

                # Second forward/backward pass on selected samples with FULL gradients
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, filtered_inputs)

                if args.n_gpu > 1:
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                # Re-enable hooks
                self.grad_hook.enable_hooks()

                return loss.detach()

        elif args.method == "MaxLoss":
            # MaxLoss doesn't use gradient-based selection
            # Fall back to standard selection (select highest loss samples)

            # Prepare inputs
            batch_train = self._prepare_inputs(inputs)

            # Compute per-sample losses
            with torch.no_grad():
                losses = []
                for i in range(len(batch_train['input_ids'])):
                    single_input = {k: v[[i]] for k, v in batch_train.items()}
                    outputs = model(**single_input)
                    losses.append(outputs.loss.item())

            # Select highest loss samples
            from .utils import greedy_selection
            selected_ind = greedy_selection(
                torch.tensor(losses),
                torch.zeros((len(losses), len(losses))),
                int(len(losses) * args.selection_frac)
            )

            # Filter to selected samples
            filtered_inputs = {
                'input_ids': batch_train['input_ids'][selected_ind],
                'attention_mask': batch_train['attention_mask'][selected_ind],
                'labels': batch_train['labels'][selected_ind]
            }

            # Standard forward/backward on selected samples
            model.zero_grad()

            # Disable hooks for standard optimizer
            if not using_compressed_optimizer and self.grad_hook.hooks_registered:
                self.grad_hook.disable_hooks()

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, filtered_inputs)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Re-enable hooks
            if not using_compressed_optimizer and self.grad_hook.hooks_registered:
                self.grad_hook.enable_hooks()

            return loss.detach()

        else:
            # No selection, two cases:
            # 1. MeSO without selection: hooks enabled, use compressed gradients
            # 2. Standard AdamW: hooks disabled, use full gradients

            model.zero_grad()

            # Only disable hooks if NOT using compressed optimizer
            # If using MeSO without selection, we still want compressed gradients
            hooks_were_enabled = self.grad_hook.hooks_enabled if self.grad_hook.hooks_registered else False

            if not using_compressed_optimizer and self.grad_hook.hooks_registered:
                self.grad_hook.disable_hooks()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Restore hooks state
            if not using_compressed_optimizer and hooks_were_enabled:
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

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Accumulate loss
                batch_size = batch["input_ids"].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # Compute average loss and perplexity
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss != float("inf") else float("inf")

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
