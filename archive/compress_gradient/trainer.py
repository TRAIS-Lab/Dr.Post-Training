"""
Hook-based trainer.
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
from .utils import greedy_selection

logger = logging.getLogger(__name__)


class CompGradTrainer(Trainer):
    """
    Trainer with gradient hook manager for per-sample gradient computations.
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
        # The parent Trainer expects eval_dataset for its evaluation strategy
        eval_dataset = kwargs.get('eval_dataset', None)
        if eval_dataset is None:
            raise ValueError("HookTrainer requires an eval_dataset to be passed in kwargs")

        # Pass eval_dataset to parent Trainer (it will use it for triggering evaluation)
        super().__init__(*args, **kwargs)

        # Store our custom datasets
        self.grad_hook = grad_hook
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset

        # Note: self.eval_dataset (from parent) will also point to eval_dataset
        # but we override the evaluate() method anyway

        # Initialize results tracking
        self.evaluation_results = []

        # Create validation dataloader iterator for efficient batch sampling during training
        self.val_dataloader_iter = iter(self.get_val_dataloader(self.val_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True))

        logger.info("="*60)
        logger.info("Initialized HookTrainer")
        logger.info(f"  Selection method: {self.args.method} (fraction: {self.args.selection_frac})")
        logger.info(f"  Validation set size: {len(val_dataset)}")
        logger.info(f"  Evaluation set size: {len(eval_dataset)}")
        logger.info(f"  Compressed optimizer: {self.args.use_compressed_optimizer}")
        if self.args.use_compressed_optimizer and self.args.method in ['GREATS', 'GradNorm']:
            logger.info(f"  Optimization: Reusing compressed gradients (no second forward/backward)")
        elif self.args.method in ['GREATS', 'GradNorm'] and not self.args.use_compressed_optimizer:
            logger.info(f"  Note: Hooks will be toggled per step (enabled for selection, disabled for optimization)")
        logger.info("="*60)

    def _get_unwrapped_optimizer(self):
        """
        Get the underlying optimizer, unwrapping AcceleratedOptimizer if present.

        HuggingFace's Accelerate library wraps optimizers with AcceleratedOptimizer,
        which breaks isinstance() checks. This helper unwraps it.

        Returns:
            The underlying optimizer (e.g., MeSOAdamW or AdamW)
        """
        optimizer = self.optimizer
        if hasattr(optimizer, 'optimizer'):
            # Unwrap AcceleratedOptimizer
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

    def _select_compressed_grads(self, compressed_grads, selected_indices):
        """
        Select compressed gradients for chosen samples.

        This keeps all selection logic within the trainer - the optimizer doesn't
        need to know about data selection at all. The optimizer will then aggregate
        these selected gradients.

        Args:
            compressed_grads: List of per-sample compressed gradients [batch_size, k_l] for each layer
            selected_indices: Indices of selected samples (list or tensor)

        Returns:
            List of selected compressed gradients [num_selected, k_l] for each layer
        """
        selected_grads = []
        for grad in compressed_grads:
            if grad is None:
                selected_grads.append(None)
            else:
                # Select samples only - optimizer will average them
                selected_grad = grad[selected_indices]  # [num_selected, k_l]
                selected_grads.append(selected_grad)
        return selected_grads

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

    def compute_grad_dotprod(
        self,
        model: nn.Module,
        batch_train: Dict[str, Tensor],
        batch_val: Dict[str, Tensor],
        return_similarity: bool = True,
        return_compressed_grads: bool = False
    ):
        """
        Compute gradient dot products using layer-by-layer accumulation.

        This implementation accumulates dot products layer-by-layer instead of
        concatenating all gradients first, which is more memory efficient and
        provides better cache locality.

        Uses the same loss computation approach as training (with context manager,
        loss computation, multi-GPU handling, and gradient accumulation scaling)
        to ensure consistency between gradients used for selection and optimization.

        Args:
            model: The model
            batch_train: Training batch of size [batch_size]
            batch_val: Validation batch of size [batch_size]
            return_similarity: Whether to return similarity matrix
            return_compressed_grads: Whether to return per-sample compressed gradients for reuse

        Returns:
            - grad_dot_scores: [batch_size] gradient similarity with validation
            - similarity_matrix: [batch_size, batch_size] if return_similarity else None
            - train_compressed_grads: List of per-sample compressed gradients if return_compressed_grads else None
            - train_loss: Training loss (already scaled for multi-GPU and gradient accumulation)
        """
        # Prepare inputs using trainer's method (handles device placement and preprocessing)
        batch_val = self._prepare_inputs(batch_val)
        batch_train = self._prepare_inputs(batch_train)

        # Step 1: Compute validation gradients

        # IMPORTANT: Disable gradient aggregation for GREATS (needs per-sample gradients)
        self.grad_hook.aggregate_grads = False

        model.zero_grad()

        with self.compute_loss_context_manager():
            val_loss = self.compute_loss(model, batch_val)

        # Apply multi-GPU averaging
        if self.args.n_gpu > 1:
            val_loss = val_loss.mean()

        # Apply gradient accumulation scaling for consistency with training
        if self.args.gradient_accumulation_steps > 1:
            val_loss = val_loss / self.args.gradient_accumulation_steps

        val_loss.backward()

        # Get validation gradients from hooks and immediately average them
        # We only need the mean validation gradient for dot product computation
        val_grads_per_sample = self.grad_hook.get_compressed_grads()
        val_grads = [g.mean(dim=0) if g is not None else None for g in val_grads_per_sample]

        # Step 2: Compute training gradients (per-sample)
        model.zero_grad()

        with self.compute_loss_context_manager():
            train_loss = self.compute_loss(model, batch_train)

        # Apply multi-GPU averaging
        if self.args.n_gpu > 1:
            train_loss = train_loss.mean()

        # Apply gradient accumulation scaling for consistency with training
        if self.args.gradient_accumulation_steps > 1:
            train_loss = train_loss / self.args.gradient_accumulation_steps

        train_loss.backward()

        # Get training gradients from hooks
        train_grads = self.grad_hook.get_compressed_grads()

        # Step 3: Compute dot products layer-by-layer
        # Determine batch size from first non-None gradient
        batch_size = None
        for g in train_grads:
            if g is not None:
                batch_size = g.shape[0]
                break

        if batch_size is None:
            raise ValueError("No valid gradients found in train_grads")

        # Get device from the prepared batch
        device = batch_train['input_ids'].device

        # Initialize accumulators on device
        grad_dot_scores = torch.zeros(batch_size, device=device)
        similarity_matrix = None
        if return_similarity:
            similarity_matrix = torch.zeros(batch_size, batch_size, device=device)

        # Accumulate dot products layer by layer
        for train_g, val_g in zip(train_grads, val_grads):
            # val_g is already averaged: [layer_dim]
            # train_g is per-sample: [batch_size, layer_dim]
            if train_g is None or val_g is None:
                continue

            # Gradient dot product for this layer using fused matrix-vector multiply
            # torch.addmv(input, mat, vec) computes: input + mat @ vec
            # [batch_size, layer_dim] @ [layer_dim] -> [batch_size]
            torch.addmv(grad_dot_scores, train_g, val_g, out=grad_dot_scores)

            # Similarity matrix for this layer if requested using fused matrix-matrix multiply
            # torch.addmm(input, mat1, mat2) computes: input + mat1 @ mat2
            # [batch_size, layer_dim] @ [layer_dim, batch_size] -> [batch_size, batch_size]
            if return_similarity:
                torch.addmm(similarity_matrix, train_g, train_g.t(), out=similarity_matrix)

        # Detach from computation graph (keep on GPU for efficiency)
        grad_dot_scores = grad_dot_scores.detach()
        if similarity_matrix is not None:
            similarity_matrix = similarity_matrix.detach()

        # Step 4: Return per-sample compressed gradients if requested
        # These can be reused for optimization to avoid recomputing gradients
        train_compressed_grads = None
        if return_compressed_grads:
            # Deep copy to avoid being overwritten by subsequent operations
            train_compressed_grads = [g.clone() if g is not None else None for g in train_grads]

        return grad_dot_scores, similarity_matrix, train_compressed_grads, train_loss.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Training step with GREATS selection using simple hook approach.
        """
        model.train()
        args = self.args

        # Refresh compressors if needed
        unwrapped_optimizer = self._get_unwrapped_optimizer()
        using_compressed_optimizer = isinstance(unwrapped_optimizer, MeSOAdamW)

        if using_compressed_optimizer:
            # Note: We call this on the unwrapped optimizer because refresh_compressors_if_needed()
            # is a custom MeSOAdamW method that AcceleratedOptimizer doesn't know about.
            unwrapped_optimizer.refresh_compressors_if_needed()

        try:
            val_batch = next(self.val_dataloader_iter)
        except StopIteration:
            # Restart iterator when exhausted
            self.val_dataloader_iter = iter(self.get_val_dataloader(self.val_dataset, batch_size=4, shuffle=True))
            val_batch = next(self.val_dataloader_iter)


        # Perform selection based on method
        if args.method == 'GREATS':
            # Compute gradients and scores using simple backward
            # If using MeSO, also return compressed gradients to reuse them
            scores, similarity_matrix, saved_compressed_grads, train_loss = self.compute_grad_dotprod(
                model=model,
                batch_train=inputs,
                batch_val=val_batch,
                return_similarity=True,
                return_compressed_grads=using_compressed_optimizer
            )

            # Select samples
            lr = self.optimizer.param_groups[0]["lr"]
            selected_ind = greedy_selection(
                scores * lr,
                similarity_matrix * (lr ** 2),
                int(len(scores) * args.selection_frac)
            )

        elif args.method == "GradNorm":
            # Use diagonal of similarity matrix as gradient norm
            _, similarity_matrix, saved_compressed_grads, train_loss = self.compute_grad_dotprod(
                model=model,
                batch_train=inputs,
                batch_val=val_batch,
                return_similarity=True,
                return_compressed_grads=using_compressed_optimizer
            )

            scores = torch.diag(similarity_matrix)
            selected_ind = greedy_selection(
                scores,
                similarity_matrix * 0,
                int(len(scores) * args.selection_frac)
            )

        elif args.method == "MaxLoss":
            # Select highest loss samples
            # No compressed gradients to save for MaxLoss
            saved_compressed_grads = None
            train_loss = None  # Will compute in standard path
            with torch.no_grad():
                losses = []
                for i in range(len(inputs['input_ids'])):
                    single_input = {k: v[[i]] for k, v in inputs.items()}
                    outputs = model(**single_input)
                    losses.append(outputs.loss.item())

            selected_ind = greedy_selection(
                torch.tensor(losses),
                torch.zeros((len(losses), len(losses))),
                int(len(losses) * args.selection_frac)
            )
        else:
            selected_ind = None
            saved_compressed_grads = None
            train_loss = None  # Will compute in standard path

        # Reuse compressed gradients for MeSO optimizer
        # If we have saved compressed gradients and using MeSO, skip second forward/backward
        can_reuse_grads = (
            using_compressed_optimizer and
            saved_compressed_grads is not None and
            selected_ind is not None
        )

        if can_reuse_grads:
            # Select compressed gradients for chosen samples
            self.grad_hook.compressed_grads = self._select_compressed_grads(
                saved_compressed_grads,
                selected_ind
            )
            return train_loss

        else:
            if selected_ind is not None:
                inputs = {
                    'input_ids': inputs['input_ids'][selected_ind],
                    'attention_mask': inputs['attention_mask'][selected_ind],
                    'labels': inputs['labels'][selected_ind]
                }

            # IMPORTANT: Zero gradients before optimization backward pass
            # This ensures we compute fresh gradients for the selected samples
            # without any residual state from the selection phase
            model.zero_grad()

            # Disable hooks for standard optimizer (allows full gradient computation)
            # MeSOAdamW uses compressed gradients, standard optimizers need full gradients
            if not using_compressed_optimizer and self.grad_hook.hooks_registered:
                self.grad_hook.disable_hooks()

            # Regular training step on selected batch
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            # Re-enable hooks for next selection phase
            if not using_compressed_optimizer and self.grad_hook.hooks_registered:
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

        # Create metrics dictionary
        eval_metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_perplexity": eval_perplexity,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
        }

        # Save results
        result_entry = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
        }
        self.evaluation_results.append(result_entry)
        self._save_evaluation_results()

        logger.info(
            f"Step {self.state.global_step}: "
            f"val_perplexity={val_perplexity:.4f}, "
            f"eval_perplexity={eval_perplexity:.4f}"
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
