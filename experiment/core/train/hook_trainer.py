"""
Simplified hook-based trainer following GPT2_wikitext pattern.
"""

import sys
sys.path.append('/u/phu1/Project/Efficient-Fine-Tuning')

import time
import torch
import logging

from transformers import Trainer
from .hook_gradcomp import compute_grad_dotprod
from .utils import greedy_selection

logger = logging.getLogger(__name__)


class HookTrainer(Trainer):
    """
    Simplified trainer using _GradComp.core.hook.HookManager.
    """

    def __init__(self, hook_manager, test_dataset, *args, **kwargs):
        """
        Initialize the hook-based trainer.

        Args:
            hook_manager: _GradComp.core.hook.HookManager instance
            test_dataset: Test dataset
            *args, **kwargs: Same as transformers.Trainer
        """
        super().__init__(*args, **kwargs)
        self.hook_manager = hook_manager
        self.test_dataset = test_dataset

        logger.info("="*60)
        logger.info("Initialized HookGCTrainer (simplified, GPT2_wikitext-style)")
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
            start_time = time.time()

            # Compute gradients and scores using simple backward
            tracin_scores, similarity_matrix = compute_grad_dotprod(
                model=model,
                hook_manager=self.hook_manager,
                batch_train=inputs,
                batch_val=val_batch,
                optimizer=self.optimizer,
                return_similarity=True
            )

            print(f'Total Extra Time for GREATS: {time.time() - start_time:.2f} seconds')

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

            print(f'Total Extra Time for GradSelection: {time.time() - start_time:.2f} seconds')

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
