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


def compute_grad_dotprod(
    model: nn.Module,
    grad_hook: GradientHook,
    batch_train: Dict[str, Tensor],
    batch_val: Dict[str, Tensor],
    return_similarity: bool = True
):
    """
    Compute gradient dot products using simple backward pass + hooks.

    Args:
        model: The model
        grad_hook: GradientHook with hooks attached
        batch_train: Training batch of size [batch_size]
        batch_val: Validation batch of size [batch_size]
        return_similarity: Whether to return similarity matrix

    Returns:
        - grad_dot_scores: [batch_size] gradient similarity with validation
        - similarity_matrix: [batch_size, batch_size] if return_similarity else None
    """
    # Move validation batch to same device as training
    device = batch_train['input_ids'].device
    batch_val = {k: v.to(device) for k, v in batch_val.items()}

    # Step 1: Compute validation gradients
    model.zero_grad()
    val_outputs = model(**batch_val)
    val_loss = val_outputs.loss
    val_loss.backward()

    # Get validation gradients from hooks
    val_grads = grad_hook.get_compressed_grads()
    # Concatenate all layers: each is [batch_size, compressed_grad_dim]
    val_grads_concat = torch.cat([g for g in val_grads if g is not None], dim=1)  # [batch_size, total_dim]
    # Average over validation batch
    val_grad_avg = val_grads_concat.mean(dim=0, keepdim=True)  # [1, total_dim]

    # Step 2: Compute training gradients (per-sample)
    model.zero_grad()
    train_outputs = model(**batch_train)
    train_loss = train_outputs.loss
    train_loss.backward()

    # Get training gradients from hooks
    train_grads = grad_hook.get_compressed_grads()
    # Concatenate all layers: each is [batch_size, compressed_grad_dim]
    train_grads_concat = torch.cat([g for g in train_grads if g is not None], dim=1)  # [batch_size, total_dim]

    # Step 3: Compute GradDot scores
    # [batch_size, total_dim] x [total_dim, 1] -> [batch_size]
    grad_dot_scores = torch.matmul(train_grads_concat, val_grad_avg.t()).squeeze(-1)
    grad_dot_scores = grad_dot_scores.cpu().detach()

    # Step 4: Compute similarity matrix if requested
    similarity_matrix = None
    if return_similarity:
        # [batch_size, total_dim] x [total_dim, batch_size] -> [batch_size, batch_size]
        similarity_matrix = torch.matmul(train_grads_concat, train_grads_concat.t())
        similarity_matrix = similarity_matrix.cpu().detach()

    return grad_dot_scores, similarity_matrix


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

        # Track selected indices for compressed optimizer
        self.current_selected_indices = None

        logger.info("="*60)
        logger.info("Initialized HookTrainer")
        logger.info(f"  Selection method: {self.args.method} (fraction: {self.args.selection_frac})")
        logger.info(f"  Validation set size: {len(val_dataset)}")
        logger.info(f"  Evaluation set size: {len(eval_dataset)}")
        logger.info(f"  Compressed optimizer: {self.args.use_compressed_optimizer}")
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

    def optimizer_step(self, *args, **kwargs):
        """
        Override optimizer step to pass selected indices to compressed optimizer.
        """
        # Unwrap only to check the type
        unwrapped_optimizer = self._get_unwrapped_optimizer()

        # Check if using compressed optimizer and have selected indices
        if isinstance(unwrapped_optimizer, MeSOAdamW) and self.current_selected_indices is not None:
            # Call through the (potentially wrapped) optimizer to preserve Accelerate functionality
            # AcceleratedOptimizer will forward the selected_indices kwarg to MeSOAdamW
            self.optimizer.step(selected_indices=list(self.current_selected_indices))
            # Reset selected indices after use
            self.current_selected_indices = None
        else:
            # Standard optimizer step
            super().optimizer_step(*args, **kwargs)

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Training step with GREATS selection using simple hook approach.
        """
        model.train()
        args = self.args

        # Refresh compressors if needed
        # This must happen BEFORE forward/backward passes to ensure gradients
        # are computed with the refreshed projectors
        # Note: We call this on the unwrapped optimizer because refresh_compressors_if_needed()
        # is a custom MeSOAdamW method that AcceleratedOptimizer doesn't know about.
        # This is safe because it only reads state and refreshes compressors, not performing optimizer steps.
        unwrapped_optimizer = self._get_unwrapped_optimizer()

        if isinstance(unwrapped_optimizer, MeSOAdamW):
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
            scores, similarity_matrix = compute_grad_dotprod(
                model=model,
                grad_hook=self.grad_hook,
                batch_train=inputs,
                batch_val=val_batch,
                return_similarity=True
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
            _, similarity_matrix = compute_grad_dotprod(
                model=model,
                grad_hook=self.grad_hook,
                batch_train=inputs,
                batch_val=val_batch,
                optimizer=self.optimizer,
                return_similarity=True
            )

            scores = torch.diag(similarity_matrix)
            selected_ind = greedy_selection(
                scores,
                similarity_matrix * 0,
                int(len(scores) * args.selection_frac)
            )

        elif args.method == "MaxLoss":
            # Select highest loss samples
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

        # Store selected indices for compressed optimizer
        self.current_selected_indices = selected_ind

        if selected_ind is not None:
            inputs = {
                'input_ids': inputs['input_ids'][selected_ind],
                'attention_mask': inputs['attention_mask'][selected_ind],
                'labels': inputs['labels'][selected_ind]
            }

        # Regular training step on selected batch
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

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
