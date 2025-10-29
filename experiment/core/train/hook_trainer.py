"""
Hook-based trainer.
"""

import time
import logging

import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

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

def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    selected_indices = []

    for _ in range(K):
        # Select the index with the highest score
        idx_max = torch.argmax(scores).item()
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to -inf
        # to ensure it's not selected again
        scores[idx_max] = -float('inf')

    return selected_indices

class HookTrainer(Trainer):
    """
    Simplified trainer using GradComp.core.hook.HookManager.
    """

    def __init__(self, hook_manager, test_dataset, *args, **kwargs):
        """
        Initialize the hook-based trainer.

        Args:
            hook_manager: GradComp.core.hook.HookManager instance
            test_dataset: Test dataset
            *args, **kwargs: Same as transformers.Trainer
        """
        super().__init__(*args, **kwargs)
        self.hook_manager = hook_manager
        self.test_dataset = test_dataset

        logger.info("="*60)
        logger.info("Initialized HookTrainer")
        logger.info(f"  Selection method: {self.args.method}")
        logger.info(f"  Fracinv: {self.args.fracinv}")
        logger.info("="*60)

    def get_gc_eval_dataloader(self, eval_dataset, val_batchsize=2, shuffle=True):
        """Create validation dataloader."""
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        if shuffle:
            sampler = RandomSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=val_batchsize,
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

        # Get validation dataloader
        eval_dataloader = self.get_gc_eval_dataloader(self.eval_dataset, val_batchsize=2, shuffle=True)
        val_batch = next(iter(eval_dataloader))

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
