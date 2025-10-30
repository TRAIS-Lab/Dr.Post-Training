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
import torch.nn as nn
from typing import Dict, Optional
from torch import Tensor
from torch.utils.data import DataLoader

from transformers import Trainer

from ..GradComp.core.hook import HookManager

logger = logging.getLogger(__name__)

def compute_grad_dotprod(
    model: nn.Module,
    hook_manager: HookManager,
    batch_train: Dict[str, Tensor],
    batch_val: Dict[str, Tensor],
    return_similarity: bool = True
):
    """
    Compute gradient dot products using simple backward pass + hooks.

    Args:
        model: The model
        hook_manager: HookManager with hooks attached
        batch_train: Training batch
        batch_val: Validation batch (single batch)
        return_similarity: Whether to return similarity matrix

    Returns:
        - grad_dot_scores: [train_bs] gradient similarity with validation
        - similarity_matrix: [train_bs, train_bs] if return_similarity else None
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
    val_grads = hook_manager.get_compressed_grads()
    # Concatenate all layers: each is [batch_size, grad_dim]
    val_grads_concat = torch.cat([g for g in val_grads if g is not None], dim=1)  # [val_bs, total_dim]
    # Average over validation batch
    val_grad_avg = val_grads_concat.mean(dim=0, keepdim=True)  # [1, total_dim]

    # Step 2: Compute training gradients (per-sample)
    model.zero_grad()
    train_outputs = model(**batch_train)
    train_loss = train_outputs.loss
    train_loss.backward()

    # Get training gradients from hooks
    train_grads = hook_manager.get_compressed_grads()
    # Concatenate all layers
    train_grads_concat = torch.cat([g for g in train_grads if g is not None], dim=1)  # [train_bs, total_dim]

    # Step 3: Compute GradDot scores
    # [train_bs, total_dim] x [total_dim, 1] -> [train_bs]
    grad_dot_scores = torch.matmul(train_grads_concat, val_grad_avg.t()).squeeze(-1)
    grad_dot_scores = grad_dot_scores.cpu().detach()

    # Step 4: Compute similarity matrix if requested
    similarity_matrix = None
    if return_similarity:
        # [train_bs, total_dim] x [total_dim, train_bs] -> [train_bs, train_bs]
        similarity_matrix = torch.matmul(train_grads_concat, train_grads_concat.t())
        similarity_matrix = similarity_matrix.cpu().detach()

    return grad_dot_scores, similarity_matrix

def greedy_selection(scores, interaction_matrix, k):
    """
    Select k data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - k: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    selected_indices = []

    for _ in range(k):
        idx_max = torch.argmax(scores).item()
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -float('inf')

    return selected_indices

class HookTrainer(Trainer):
    """
    Simplified trainer using GradComp.core.hook.HookManager.
    """

    def __init__(self, hook_manager, val_dataset, *args, **kwargs):
        """
        Initialize the hook-based trainer.

        Args:
            hook_manager: GradComp.core.hook.HookManager instance
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
        self.hook_manager = hook_manager
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset  # The large evaluation set

        # Note: self.eval_dataset (from parent) will also point to eval_dataset
        # but we override the evaluate() method anyway

        # Initialize results tracking
        self.evaluation_results = []

        # Create validation dataloader iterator for efficient batch sampling during training
        self.val_dataloader_iter = iter(self.get_val_dataloader(self.val_dataset, batch_size=4, shuffle=True))

        logger.info("="*60)
        logger.info("Initialized HookTrainer")
        logger.info(f"  Selection method: {self.args.method}")
        logger.info(f"  Fracinv: {self.args.fracinv}")
        logger.info(f"  Validation set size: {len(val_dataset)}")
        logger.info(f"  Evaluation set size: {len(eval_dataset)}")
        logger.info("="*60)

    def get_val_dataloader(self, val_dataset, batch_size=4, shuffle=True):
        """Create validation dataloader for data selection."""
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

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

        # Get next validation batch from cached iterator
        try:
            val_batch = next(self.val_dataloader_iter)
        except StopIteration:
            # Restart iterator when exhausted
            self.val_dataloader_iter = iter(self.get_val_dataloader(self.val_dataset, batch_size=4, shuffle=True))
            val_batch = next(self.val_dataloader_iter)

        # Perform selection based on method
        if args.method == 'GREATS':
            torch.cuda.synchronize()
            start_time = time.time()

            # Compute gradients and scores using simple backward
            tracin_scores, similarity_matrix = compute_grad_dotprod(
                model=model,
                hook_manager=self.hook_manager,
                batch_train=inputs,
                batch_val=val_batch,
                return_similarity=True
            )

            torch.cuda.synchronize()
            print(f'\nExtra Time for Attribution: {time.time() - start_time:.2f} (s)')

            torch.cuda.synchronize()
            start_time = time.time()

            # Select samples
            lr = self.optimizer.param_groups[0]["lr"]
            selected_ind = greedy_selection(
                tracin_scores * lr,
                similarity_matrix * (lr ** 2),
                int(len(tracin_scores) / args.fracinv)
            )

            # Filter to selected samples
            inputs = {
                'input_ids': inputs['input_ids'][selected_ind],
                'attention_mask': inputs['attention_mask'][selected_ind],
                'labels': inputs['labels'][selected_ind]
            }

            torch.cuda.synchronize()
            print(f'Extra Time for Selection: {time.time() - start_time:.2f} seconds')

        elif args.method == "GradNorm":
            # Use diagonal of similarity matrix as gradient norm
            _, similarity_matrix = compute_grad_dotprod(
                model=model,
                hook_manager=self.hook_manager,
                batch_train=inputs,
                batch_val=val_batch,
                optimizer=self.optimizer,
                return_similarity=True
            )

            tracin_scores = torch.diag(similarity_matrix)
            selected_ind = greedy_selection(
                tracin_scores,
                similarity_matrix * 0,
                int(len(tracin_scores) / 2)
            )

            inputs = {
                'input_ids': inputs['input_ids'][selected_ind],
                'attention_mask': inputs['attention_mask'][selected_ind],
                'labels': inputs['labels'][selected_ind]
            }

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
                int(len(losses) / 2)
            )

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

        logger.debug(f"Saved evaluation results to {output_file}")

    def on_train_end(self):
        """Called at the end of training to save final results."""
        self._save_evaluation_results()
        logger.info(f"Training completed. Final results saved to {self.args.output_dir}/evaluation_results.json")
