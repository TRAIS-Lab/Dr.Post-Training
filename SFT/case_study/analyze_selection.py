#!/usr/bin/env python
# coding=utf-8
"""
Case study: Selection analysis during standard training.

Runs standard full-parameter training (no selection, full batch Adam) while
simultaneously computing what LayerWiseSubset and GlobalSubset selection would pick at
each step. This provides an apples-to-apples comparison of both selection
methods on the same model state and same data batches.

Output: selection_records.json with per-step records of:
  - Decoded training and validation sample texts
  - LayerWiseSubset: per-layer scores and selected indices
  - GlobalSubset: global accumulated scores and selected indices
"""

import json
import logging
import math
import os
import sys
import time
import warnings

import datasets
import torch
import torch.distributed as dist
import transformers

warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, HfArgumentParser, Trainer, set_seed,
)

from SFT.data.get_train_dataset import get_training_dataset
from SFT.data.get_val_dataset import get_dataset, DEFAULT_SEQ_LENGTH_MULTIPLIER
from SFT.train.data_arguments import DataArguments, get_data_statistics
from SFT.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from SFT.train.training_arguments import TrainingArguments

from drpt import GradientHook
from drpt.selection.state import LayerWiseSubsetState, GlobalSubsetState

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_trainable_layers(model):
    """Find all Linear layers for full fine-tuning."""
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_names.append(name)
    return layer_names


class CaseStudyTrainer(Trainer):
    """
    Full-Training trainer that also computes LayerWiseSubset and GlobalSubset selection scores
    at each training step for analysis purposes. The actual training uses
    standard full-batch Adam (no selection).
    """

    def __init__(self, grad_hook, val_dataset, selection_frac=0.5,
                 record_freq=1, *args, **kwargs):
        eval_dataset = kwargs.get('eval_dataset', None)
        super().__init__(*args, **kwargs)

        self.grad_hook = grad_hook
        self.val_dataset = val_dataset
        self.eval_dataset_custom = eval_dataset
        self.selection_frac = selection_frac
        self._record_freq = record_freq

        # Selection records storage
        self._selection_records = []

        # Evaluation results tracking
        self.evaluation_results = []
        if torch.cuda.is_available():
            self.training_start_event = torch.cuda.Event(enable_timing=True)
            self.training_start_event.record()
        else:
            self.training_start_event = None
            self.training_start_time = time.time()

        # Val dataloader for scoring
        val_batch_size = getattr(self.args, 'val_batch_size_for_selection', 1) or 1
        self.val_dataloader_iter = iter(
            self._make_val_dataloader(val_dataset, batch_size=val_batch_size, shuffle=True)
        )
        self.val_batch_size = val_batch_size

        logger.info("=" * 60)
        logger.info("Case Study Trainer initialized")
        logger.info(f"  Training: Full-Training Full (no selection)")
        logger.info(f"  Scoring: LayerWiseSubset + GlobalSubset at each step")
        logger.info(f"  Selection fraction: {selection_frac}")
        logger.info(f"  Record frequency: every {record_freq} steps")
        logger.info(f"  Val set size: {len(val_dataset)}")
        logger.info(f"  Eval set size: {len(eval_dataset) if eval_dataset else 0}")
        logger.info("=" * 60)

    def _make_val_dataloader(self, dataset, batch_size=1, shuffle=True):
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_val_batch(self):
        try:
            return next(self.val_dataloader_iter)
        except StopIteration:
            self.val_dataloader_iter = iter(
                self._make_val_dataloader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True)
            )
            return next(self.val_dataloader_iter)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Full-Training training step + scoring-only passes for both methods.

        Flow:
        1. Disable hooks → standard forward+backward → real gradients
        2. Save gradients
        3. Enable hooks → capture val grads → LayerWiseSubset scoring → extract records
        4. Enable hooks → GlobalSubset scoring (reuse val cache) → extract records
        5. Restore real gradients for optimizer step
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ============================================================
        # Step 1: Full-Training forward + backward (no hooks, real training)
        # ============================================================
        self.grad_hook.disable_hooks()

        model.zero_grad()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        self.grad_hook.enable_hooks()

        # ============================================================
        # Step 2: Run scoring passes (if this is a recording step)
        # ============================================================
        current_step = self.state.global_step + 1  # +1 because step increments after
        if current_step % self._record_freq == 0:
            # Save real gradients
            saved_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    saved_grads[name] = param.grad.clone()

            # Run scoring
            self._run_dual_scoring(model, inputs, current_step)

            # Restore real gradients
            model.zero_grad()
            for name, param in model.named_parameters():
                if name in saved_grads:
                    param.grad = saved_grads[name]

        return loss.detach()

    def _compute_loss_for_scoring(self, model, batch):
        """Compute loss for scoring passes."""
        with self.compute_loss_context_manager():
            outputs = model(**batch)
            loss = outputs.loss
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        return loss

    def _run_dual_scoring(self, model, train_batch, step):
        """
        Run both LayerWiseSubset and GlobalSubset scoring on the same batch.

        Uses the existing drpt infrastructure:
        1. Capture val gradients (shared for both methods)
        2. Run LayerWiseSubset scoring backward → extract per-layer records
        3. Run GlobalSubset scoring backward → extract global records
        """
        val_batch = self._get_val_batch()
        val_batch = self._prepare_inputs(val_batch)
        train_batch_size = train_batch['input_ids'].shape[0]

        # Use lr=1.0 for scoring since we only care about relative scores.
        # The actual lr doesn't matter for ranking (topk), and lr=0 during
        # warmup would zero out all scores making selection meaningless.
        lr = 1.0

        # ---- Capture validation gradients (shared for both methods) ----
        self.grad_hook.start_val_capture(scoring_method="reduced_ghost")
        model.zero_grad()
        val_loss = self._compute_loss_for_scoring(model, val_batch)
        val_loss.backward()
        self.grad_hook.end_val_capture()

        # ---- LayerWiseSubset scoring ----
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=train_batch_size,
            selection_method="LayerWiseSubset",
            frac=self.selection_frac,
            lr=lr,
            record_selections=True,
        )
        self.grad_hook.set_token_counts(train_batch['labels'], train_batch_size)

        model.zero_grad()
        train_loss_lw = self._compute_loss_for_scoring(model, train_batch)
        train_loss_lw.backward()

        # Extract layer_wise_subset records
        lw_state = self.grad_hook.selection_state
        layer_wise_subset_records = list(lw_state._selection_records) if lw_state._selection_records else []

        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()

        # ---- GlobalSubset scoring (reuse val cache) ----
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=train_batch_size,
            selection_method="GlobalSubset",
            frac=self.selection_frac,
            lr=lr,
            compute_scores_only=True,
            record_selections=True,
        )
        self.grad_hook.set_token_counts(train_batch['labels'], train_batch_size)

        model.zero_grad()
        train_loss_ss = self._compute_loss_for_scoring(model, train_batch)
        train_loss_ss.backward()

        # Extract subset records
        ss_state = self.grad_hook.selection_state
        subset_selected = ss_state.get_final_selection()
        subset_records = list(ss_state._selection_records) if ss_state._selection_records else []

        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()

        # Clear val cache
        self.grad_hook.clear_val_buffer()

        # ---- Build combined record ----
        tokenizer = self.processing_class
        train_texts = tokenizer.batch_decode(train_batch['input_ids'], skip_special_tokens=False)
        val_texts = tokenizer.batch_decode(val_batch['input_ids'], skip_special_tokens=False)

        step_record = {
            'step': step,
            'train_loss': train_loss_lw.item(),
            'train_samples': train_texts,
            'val_samples': val_texts,
            'layer_wise_subset': {
                'layers': layer_wise_subset_records,
            },
            'subset': {
                'selection': subset_records[0] if subset_records else {},
                'selected_indices': subset_selected.cpu().tolist(),
            },
        }

        self._selection_records.append(step_record)

        # Log progress
        if layer_wise_subset_records:
            lw_first_layer = layer_wise_subset_records[0]
            lw_selected = lw_first_layer.get('selected_indices', [])
        else:
            lw_selected = []
        ss_selected = subset_selected.cpu().tolist()

        logger.info(
            f"Step {step}: Scored {train_batch_size} samples | "
            f"LayerWiseSubset L0 selected: {lw_selected} | "
            f"GlobalSubset selected: {ss_selected}"
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Evaluate on both val and eval datasets, save results."""
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        val_loss, val_ppl = self._evaluate_on_dataset(self.val_dataset, "Validation")
        eval_loss, eval_ppl = self._evaluate_on_dataset(self.eval_dataset_custom, "Evaluation")

        # Wall time
        if self.training_start_event is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            torch.cuda.synchronize()
            wall_time = self.training_start_event.elapsed_time(ev) / 1000.0
        else:
            wall_time = time.time() - self.training_start_time

        eval_metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_perplexity": eval_ppl,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "wall_time": wall_time,
        }

        self.evaluation_results.append({
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_ppl,
            "wall_time": wall_time,
        })
        self._save_results()

        logger.info(
            f"Step {self.state.global_step}: "
            f"val_ppl={val_ppl:.4f}, eval_ppl={eval_ppl:.4f}, wall={wall_time:.1f}s"
        )

        if self.grad_hook is not None:
            self.grad_hook.enable_hooks()

        return eval_metrics

    def _evaluate_on_dataset(self, dataset, description="Evaluation"):
        model = self.model
        model.eval()

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

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                outputs = model(**batch)
                batch_size = batch["input_ids"].shape[0]
                total_loss += outputs.loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss != float("inf") else float("inf")

        model.train()
        return avg_loss, ppl

    def _save_results(self):
        """Save evaluation results and selection records."""
        output_dir = self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save evaluation results
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

        # Save selection records
        if self._selection_records:
            data = {
                'metadata': {
                    'experiment': 'case_study_dual_scoring',
                    'description': (
                        'Full-Training (full batch Adam) with simultaneous '
                        'LayerWiseSubset and GlobalSubset scoring at each step. '
                        'Same model, same batches, different selection methods.'
                    ),
                    'selection_frac': self.selection_frac,
                    'train_batch_size': self.args.per_device_train_batch_size,
                    'record_freq': self._record_freq,
                    'num_layers': len(self.grad_hook.layer_names),
                    'layer_names': self.grad_hook.layer_names,
                },
                'steps': self._selection_records,
            }
            with open(os.path.join(output_dir, "selection_records.json"), "w") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(self._selection_records)} selection records")

    def on_train_end(self):
        self._save_results()
        logger.info(f"Training completed. Results saved to {self.args.output_dir}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training parameters {training_args}")
    set_seed(training_args.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if model_args.torch_dtype is None and (training_args.bf16 or training_args.fp16):
        model_args.torch_dtype = "bfloat16" if training_args.bf16 else "float16"

    model_kwargs = {"torch_dtype": model_args.torch_dtype}
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs)
    add_padding_to_tokenizer(tokenizer)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    train_dataset = get_training_dataset(
        data_dir=data_args.data_dir,
        task=training_args.analysis_dataset,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        sample_percentage=data_args.percentage,
        seed=training_args.data_seed if training_args.data_seed is not None else training_args.seed,
        train_files=data_args.train_files if data_args.train_files else None,
        train_dataset_names=training_args.train_dataset_names,
    )

    avg_train_seq_length = get_data_statistics(train_dataset, return_avg_length=True)
    val_seq_length_threshold = int(avg_train_seq_length * DEFAULT_SEQ_LENGTH_MULTIPLIER)

    val_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        split="validation",
        k=training_args.n_val,
        subject=training_args.subject,
        max_seq_length_threshold=val_seq_length_threshold,
    )

    eval_dataset = get_dataset(
        task=training_args.analysis_dataset,
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        split="test",
        k=training_args.n_eval,
        subject=training_args.subject,
    )

    # Set up gradient hook for scoring (no compressors = exact scoring)
    layer_names = find_trainable_layers(model)
    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(training_args.device),
    )

    logger.info(f"Registered hooks on {len(layer_names)} layers")
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Eval dataset: {len(eval_dataset)} samples")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Force method to NA for standard training
    training_args.method = "NA"

    # Create trainer
    selection_frac = getattr(training_args, 'selection_frac', 0.5)
    record_freq = max(1, getattr(training_args, 'record_selections_freq', 1))

    trainer = CaseStudyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        grad_hook=grad_hook,
        selection_frac=selection_frac,
        record_freq=record_freq,
    )

    # Initial evaluation
    logger.info("*** Initial evaluation ***")
    trainer.evaluate()

    # Train
    logger.info("*** Starting training with dual scoring ***")
    trainer.train()

    # Final evaluation
    logger.info("*** Final evaluation ***")
    trainer.evaluate()

    # Save everything
    trainer.on_train_end()

    # Cleanup
    if grad_hook.hooks_registered:
        grad_hook.remove_hooks()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
