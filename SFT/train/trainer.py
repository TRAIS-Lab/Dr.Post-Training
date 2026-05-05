"""
Layer-Wise Subset SFT trainer with unified data curation and model update.
"""

import json
import math
import os
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict
from torch import Tensor

from transformers import Trainer

from drpt.optimizer import MeSOAdamW
from drpt.hook import GradientHook
from drpt.selection import create_separate_batch_strategy, create_merged_batch_strategy

logger = logging.getLogger(__name__)


class LayerWiseSubsetTrainer(Trainer):
    """
    SFT Trainer supporting gradient-based data curation with optional compression.
    """

    def __init__(self, grad_hook: GradientHook, val_dataset, *args, **kwargs):
        """
        Initialize the trainer.

        Args:
            grad_hook: GradientHook instance for gradient capture and compression.
            val_dataset: Small validation set used for data curation during training.
                Batches from this set are merged with training batches to compute
                curation scores that guide which training samples to use.
            *args, **kwargs: Same as transformers.Trainer.
                Must include eval_dataset: Large held-out set for generalization testing.
                This is separate from val_dataset and used only during evaluation.
        """
        # Extract eval_dataset from kwargs before passing to parent
        eval_dataset = kwargs.get('eval_dataset', None)
        if eval_dataset is None:
            raise ValueError("LayerWiseSubsetTrainer requires an eval_dataset to be passed in kwargs")

        # Pass eval_dataset to parent Trainer
        super().__init__(*args, **kwargs)

        # Store our custom datasets
        self.grad_hook = grad_hook
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset

        # Set selection_frac to 1.0 when no curation method is specified (consistency)
        # This means we "select" all samples when not doing data curation
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

        # Track cumulative evaluation time so it can be subtracted from wall time
        self.cumulative_eval_time = 0.0

        # Determine validation batch size for data curation
        # Use val_batch_size_for_selection if provided, otherwise default to per_device_train_batch_size
        self.val_batch_size_for_selection = (
            self.args.val_batch_size_for_selection
            if hasattr(self.args, 'val_batch_size_for_selection') and self.args.val_batch_size_for_selection is not None
            else self.args.per_device_train_batch_size
        )

        # Create validation dataloader iterator for efficient batch sampling during training
        # Only needed if we're doing layer_wise_subset data curation
        if val_dataset is not None:
            self.val_dataloader_iter = iter(
                self.get_val_dataloader(self.val_dataset, batch_size=self.val_batch_size_for_selection, shuffle=True)
            )
        else:
            self.val_dataloader_iter = None

        # Determine if we have update compression (which implies MeSO)
        self.has_compression = (
            self.grad_hook is not None and
            self.grad_hook.compression_mode.uses_compressed_updates
        )

        # Curation recording for case study analysis
        self._record_selections = getattr(self.args, 'record_selections', False)
        self._record_selections_freq = max(1, getattr(self.args, 'record_selections_freq', 1))
        self._selection_records = []

        # Create curation strategy for clean separation of curation methods
        # SFT uses topk mode: select top frac samples by alignment score
        # Strategy is determined by val_strategy argument:
        # - separate_batch_factorized: Separate val pass, store factorized components (default)
        # - separate_batch: Separate val pass, store mean gradient
        # - merged_batch: Merge train+val into single batch
        val_strategy = getattr(self.args, 'val_strategy', 'separate_batch_factorized')
        scoring_method = getattr(self.args, 'scoring_method', 'reduced_ghost')
        subset_mode = getattr(self.args, 'subset_mode', 'one_pass')
        if val_strategy == 'merged_batch':
            self.selection_strategy = create_merged_batch_strategy(
                method=self.args.method,
                grad_hook=self.grad_hook,
                frac=getattr(self.args, 'selection_frac', 0.5),
                use_second_order=getattr(self.args, 'use_second_order', False),
                selection_mode=getattr(self.args, 'selection_mode', 'topk'),
                record_selections=self._record_selections,
                scoring_method=scoring_method,
                subset_mode=subset_mode,
            )
        else:
            # separate_batch_factorized or separate_batch
            self.selection_strategy = create_separate_batch_strategy(
                method=self.args.method,
                grad_hook=self.grad_hook,
                frac=getattr(self.args, 'selection_frac', 0.5),
                use_second_order=getattr(self.args, 'use_second_order', False),
                selection_mode=getattr(self.args, 'selection_mode', 'topk'),
                record_selections=self._record_selections,
                scoring_method=scoring_method,
                subset_mode=subset_mode,
            )
        self.val_strategy = val_strategy

        logger.info("="*60)
        logger.info("Initialized LayerWiseSubsetTrainer")
        selection_frac = getattr(self.args, 'selection_frac', None)
        logger.info(f"  Method: {self.args.method} (curation fraction: {selection_frac})")
        logger.info(f"  Validation strategy: {self.val_strategy}")
        logger.info(f"  Compression: {self.has_compression}")
        logger.info(f"  Validation set size: {len(val_dataset) if val_dataset is not None else 0}")
        logger.info(f"  Evaluation set size: {len(eval_dataset) if eval_dataset is not None else 0}")
        logger.info(f"  Training batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"  Validation batch size (for curation): {self.val_batch_size_for_selection}")
        if self._record_selections:
            logger.info(f"  Curation recording: enabled (every {self._record_selections_freq} steps)")

        # Log the training mode based on configuration
        # Naming convention: {curation}-{compression}-{training_type}
        if self.args.method in ('LayerWiseSubset', 'GlobalSubset'):
            if self.has_compression:
                logger.info(f"  Mode: {self.args.method} with compression (MeSO optimizer)")
            else:
                logger.info(f"  Mode: {self.args.method} without compression (standard optimizer)")
        elif self.has_compression:
            logger.info(f"  Mode: MeSO only (compressed gradients, no curation)")
        else:
            logger.info(f"  Mode: Baseline (full gradients, no curation)")

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
        """Create validation dataloader for data curation."""
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

        NOTE: This method is kept for backward compatibility but is no longer used
        by default. The trainer now uses separate val/train passes via StoredValStrategy
        to avoid padding overhead when batches have different sequence lengths.

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
        Training step using curation strategy pattern.

        The curation strategy handles the difference between:
        - LayerWiseSubset: Single-pass, per-layer curation
        - GlobalSubset: Two-pass, global curation
        - NA: Baseline (no curation)

        With or without compression (MeSO).
        """
        model.train()
        args = self.args

        # Refresh compressors if needed (only for MeSO)
        if self.has_compression:
            unwrapped_optimizer = self._get_unwrapped_optimizer()
            if isinstance(unwrapped_optimizer, MeSOAdamW):
                unwrapped_optimizer.refresh_compressors_if_needed()

        # === DATA CURATION MODE (LayerWiseSubset or GlobalSubset) ===
        if args.method in ('LayerWiseSubset', 'GlobalSubset'):
            # Get validation batch for curation
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

            # Get current learning rate
            lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, 'optimizer') and self.optimizer else args.learning_rate
            if lr is None or lr == 0:
                lr = args.learning_rate

            if self.val_strategy == 'merged_batch':
                # === MERGED BATCH MODE: Merge train+val, single forward/backward ===
                merged_batch = self._merge_batches(batch_train, batch_val)

                def compute_loss(model, batch):
                    return self._compute_loss_for_selection(model, batch)

                loss = self.selection_strategy.execute_training_step(
                    model=model,
                    merged_batch=merged_batch,
                    train_batch_size=train_batch_size,
                    compute_loss_fn=compute_loss,
                    lr=lr,
                    batch_train=batch_train,  # For GlobalSubset pass 2
                )
            else:
                # === SEPARATE BATCH MODE: Separate val pass, then train with stored grads ===
                # Storage mode (factorized/full/compressed) is derived from scoring_method:
                #   full_ghost   → factorized [V,S,O] + [V,S,I] (for pairwise scoring)
                #   reduced_ghost/direct → full [O,I] (summed gradient, cheaper)
                #   compress     → compressed [k]
                # PASS 1: Capture validation gradients
                self.grad_hook.start_val_capture(
                    scoring_method=getattr(self.args, 'scoring_method', 'reduced_ghost'),
                )
                model.zero_grad()
                val_loss = self._compute_loss_for_selection(model, batch_val)
                val_loss.backward()
                self.grad_hook.end_val_capture()

                # PASS 2: Train with curation using stored val gradients
                def compute_train_loss():
                    loss = self._compute_loss_for_selection(model, batch_train)
                    return loss, {}  # SeparateBatchStrategy expects (loss, stats) tuple

                loss, _ = self.selection_strategy.execute_training_step(
                    model=model,
                    batch_size=train_batch_size,
                    compute_loss_fn=compute_train_loss,
                    lr=lr,
                    # Pass labels for token-based gradient scaling
                    labels=batch_train.get('labels'),
                    # For GlobalSubset pass 2, provide filter function
                    filter_batch_fn=lambda indices: (
                        lambda: (self._compute_loss_for_selection(model, {
                            'input_ids': batch_train['input_ids'][indices],
                            'attention_mask': batch_train['attention_mask'][indices],
                            'labels': batch_train['labels'][indices],
                        }), {})
                    )
                )

                # Cleanup val buffer
                self.grad_hook.clear_val_buffer()

            # Record curation data for case study
            if self._record_selections and self.state.global_step % self._record_selections_freq == 0:
                self._capture_selection_record(batch_train, batch_val)

            return loss

        # === BASELINE MODE (no data curation) ===
        else:
            return self._training_step_baseline(model, inputs)

    def _compute_loss_for_selection(self, model, batch):
        """Compute loss for curation (handles multi-GPU and grad accumulation)."""
        with self.compute_loss_context_manager():
            outputs = model(**batch)
            loss = outputs.loss

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def _training_step_baseline(self, model, inputs):
        """Baseline training step without curation."""
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

    def _capture_selection_record(
        self,
        batch_train: Dict[str, Tensor],
        batch_val: Dict[str, Tensor],
    ):
        """Capture curation record from the last training step.

        Records decoded text for both training and validation samples,
        along with per-layer (LayerWiseSubset) or global (GlobalSubset) curation data.
        """
        record = getattr(self.selection_strategy, 'last_selection_record', None)
        if record is None:
            return

        # Decode training and validation samples to text
        tokenizer = self.processing_class
        train_texts = tokenizer.batch_decode(batch_train['input_ids'], skip_special_tokens=False)
        val_texts = tokenizer.batch_decode(batch_val['input_ids'], skip_special_tokens=False)

        step_record = {
            'step': self.state.global_step,
            'train_samples': train_texts,
            'val_samples': val_texts,
        }

        if self.args.method == 'LayerWiseSubset':
            step_record['layers'] = record
        else:
            # GlobalSubset: single global curation
            step_record['selection'] = record[0] if record else {}

        self._selection_records.append(step_record)

    def _save_selection_records(self):
        """Save accumulated curation records to JSON file."""
        if not self._selection_records:
            return

        output_file = os.path.join(self.args.output_dir, "selection_records.json")
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'method': self.args.method,
                'selection_frac': self.args.selection_frac,
                'train_batch_size': self.args.per_device_train_batch_size,
                'record_freq': self._record_selections_freq,
                'num_layers': len(self.grad_hook.layer_names) if self.grad_hook else 0,
                'layer_names': self.grad_hook.layer_names if self.grad_hook else [],
            },
            'steps': self._selection_records,
        }

        with open(output_file, 'w') as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self._selection_records)} curation records to {output_file}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute both validation and evaluation perplexity.

        Computes:
        - Validation perplexity: on the small validation set (used for data curation)
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

        # Record time before evaluation
        if self.training_start_event is not None:
            eval_start_event = torch.cuda.Event(enable_timing=True)
            eval_start_event.record()
        else:
            eval_start_cpu = time.time()

        # Evaluate on validation dataset (small set for data curation)
        val_loss, val_perplexity = self._evaluate_on_dataset(
            self.val_dataset,
            description="Validation"
        )

        # Evaluate on evaluation dataset (large held-out set for generalization)
        eval_loss, eval_perplexity = self._evaluate_on_dataset(
            self.eval_dataset_custom,
            description="Evaluation"
        )

        # Calculate elapsed wall time and subtract cumulative evaluation time
        if self.training_start_event is not None:
            eval_end_event = torch.cuda.Event(enable_timing=True)
            eval_end_event.record()
            torch.cuda.synchronize()
            wall_time = self.training_start_event.elapsed_time(eval_end_event) / 1000.0  # Convert ms to seconds
            eval_duration = eval_start_event.elapsed_time(eval_end_event) / 1000.0
        else:
            eval_end_cpu = time.time()
            wall_time = eval_end_cpu - self.training_start_time
            eval_duration = eval_end_cpu - eval_start_cpu

        self.cumulative_eval_time += eval_duration
        train_wall_time = wall_time - self.cumulative_eval_time

        # Create metrics dictionary
        eval_metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_perplexity": eval_perplexity,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "wall_time": wall_time,
            "train_wall_time": train_wall_time,
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
            "train_wall_time": train_wall_time,
        }
        self.evaluation_results.append(result_entry)
        self._save_evaluation_results()
        self._save_selection_records()

        logger.info(
            f"Step {self.state.global_step}: "
            f"val_perplexity={val_perplexity:.4f}, "
            f"eval_perplexity={eval_perplexity:.4f}, "
            f"train_wall_time={train_wall_time:.2f}s (eval_time={eval_duration:.2f}s)"
        )

        # Re-enable hooks after evaluation if curation method is active OR compression is used
        # Hooks are needed for both: (1) layer_wise_subset data curation, (2) MeSO compressed gradients
        if self.grad_hook is not None and (self.args.method != "NA" or self.has_compression):
            self.grad_hook.enable_hooks()

        # Reset control.should_evaluate flag (HF Trainer's CallbackHandler.on_evaluate normally
        # does this; without it, target-only runs would double-eval when a step boundary
        # coincides with an epoch boundary — _maybe_log_save_evaluate fires once at on_step_end
        # and again at on_epoch_end with the flag still True).
        if hasattr(self, 'callback_handler') and hasattr(self, 'state') and hasattr(self, 'control'):
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_metrics)

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

                # Skip NaN batches (e.g. all labels are -100 after truncation)
                if torch.isnan(loss):
                    logger.debug(f"{description}: Skipping NaN loss batch {num_batches} "
                                 f"(likely all labels masked after truncation)")
                    continue

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
        self._save_selection_records()
        logger.info(f"Training completed. Final results saved to {self.args.output_dir}/evaluation_results.json")
