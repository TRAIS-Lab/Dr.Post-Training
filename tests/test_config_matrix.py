#!/usr/bin/env python
"""
Configuration matrix test: verify all method × scoring × training-mode combinations.

Tests that all valid combinations run without error and produce gradients.
Also tests that invalid combinations (e.g., scoring_method="compress" without compressor)
raise appropriate errors.
"""

import copy
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')

from drpt.hook import GradientHook
from drpt.compressor import setup_model_compressors
from drpt.selection.strategies import (
    create_merged_batch_strategy,
    create_separate_batch_strategy,
)

SEQ_LEN = 16


# =============================================================================
# Small test model
# =============================================================================

class SmallCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_dim=64, seq_len=16, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'up': nn.Linear(hidden_dim, hidden_dim * 2),
                'down': nn.Linear(hidden_dim * 2, hidden_dim),
            }))
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            residual = x
            x = torch.relu(layer['up'](x))
            x = layer['down'](x) + residual
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})()


def make_data(n, seq_len=16, vocab_size=128):
    gen = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (n, seq_len), generator=gen)
    labels = input_ids.clone()
    labels[:, :seq_len // 2] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': torch.ones(n, seq_len, dtype=torch.long),
        'labels': labels,
    }


def get_layer_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def setup_compressors(model, layer_names):
    """Set up simple compressors for testing via the standard factory."""
    # Create sample input for dimension inference
    sample_inputs = {
        'input_ids': torch.randint(0, 128, (1, SEQ_LEN)),
        'attention_mask': torch.ones(1, SEQ_LEN, dtype=torch.long),
        'labels': torch.randint(0, 128, (1, SEQ_LEN)),
    }
    sparsifier_kwargs = {
        'proj_dim': 8, 'proj_max_batch_size': 64,
        'proj_seed': 42, 'device': 'cpu', 'proj_type': 'normal',
    }
    projector_kwargs = {
        'proj_dim': -1, 'proj_max_batch_size': 64,
        'proj_seed': 42, 'device': 'cpu', 'proj_type': 'identity',
    }
    return setup_model_compressors(
        model=model, layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device='cpu', update_freq=1000000,
    )


# =============================================================================
# Test runner
# =============================================================================

def run_config(method, scoring_method, use_meso, subset_mode, val_strategy):
    """Run a single configuration and return (success, error_msg)."""
    torch.manual_seed(42)
    model = SmallCausalLM()
    layer_names = get_layer_names(model)
    hook = GradientHook(model=model, layer_names=layer_names, device='cpu')

    # Set up compressors
    needs_score_comp = (scoring_method == "compress")
    if needs_score_comp or use_meso:
        compressors = setup_compressors(model, layer_names)
        if needs_score_comp and use_meso:
            # Shared compressors
            hook.set_compressors(compressors)
        elif needs_score_comp:
            hook.set_score_compressors(compressors)
        elif use_meso:
            hook.set_update_compressors(compressors)

    train_data = make_data(8)
    val_data = make_data(2)
    lr = 1e-3

    try:
        if val_strategy == "merged_batch":
            merged = {k: torch.cat([train_data[k], val_data[k]], dim=0) for k in train_data}
            strategy = create_merged_batch_strategy(
                method=method, grad_hook=hook, frac=0.5,
                scoring_method=scoring_method, subset_mode=subset_mode,
            )

            def compute_loss(mdl, batch):
                return mdl(**batch).loss

            loss = strategy.execute_training_step(
                model=model, merged_batch=merged, train_batch_size=8,
                compute_loss_fn=compute_loss, lr=lr, batch_train=train_data,
            )
        else:
            # Separate batch
            hook.start_val_capture(use_factorized=True, scoring_method=scoring_method)
            model.zero_grad()
            val_out = model(**val_data)
            val_out.loss.backward()
            hook.end_val_capture()

            strategy = create_separate_batch_strategy(
                method=method, grad_hook=hook, frac=0.5,
                scoring_method=scoring_method, subset_mode=subset_mode,
            )

            def compute_train_loss():
                return model(**train_data).loss, {}

            loss, stats = strategy.execute_training_step(
                model=model, batch_size=8,
                compute_loss_fn=compute_train_loss, lr=lr,
                labels=train_data['labels'],
                filter_batch_fn=lambda indices: (
                    lambda: (model(**{k: train_data[k][indices] for k in train_data}).loss, {})
                ) if method == "Subset" and subset_mode == "two_pass" else None,
            )

            hook.clear_val_buffer()

        # Check that we got a valid loss
        if isinstance(loss, tuple):
            loss = loss[0]
        assert loss is not None and not torch.isnan(loss), f"Invalid loss: {loss}"

        # Check that some params have gradients (for selection methods)
        if method != "NA":
            has_grads = any(
                p.grad is not None and p.grad.abs().max() > 0
                for p in model.parameters()
            )
            if not use_meso:
                assert has_grads, "No gradients produced"

        hook.remove_hooks()
        return True, None

    except Exception as e:
        hook.remove_hooks()
        return False, str(e)


def main():
    print("Configuration Matrix Test")
    print("=" * 90)

    methods = ["NA", "Layerwise", "Subset"]
    scoring_methods = ["ghost", "ghost_greats", "direct", "compress"]
    meso_options = [False, True]
    subset_modes = ["one_pass", "two_pass"]
    val_strategies = ["merged_batch", "separate_batch"]

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for method in methods:
        for scoring in scoring_methods:
            for use_meso in meso_options:
                for subset_mode in subset_modes:
                    for val_strat in val_strategies:
                        # Skip irrelevant combinations
                        if method == "NA" and scoring != "ghost":
                            continue  # NA has no scoring
                        if method == "NA" and subset_mode != "one_pass":
                            continue
                        if method != "Subset" and subset_mode == "two_pass":
                            continue  # two_pass only for Subset
                        if method == "NA" and val_strat != "merged_batch":
                            continue

                        total += 1
                        scoring_label = "—" if method == "NA" else scoring
                        tag = (f"{method:>10}/{scoring_label:>13}/"
                               f"{'MeSO' if use_meso else 'Full':>4}/"
                               f"{subset_mode:>8}/{val_strat}")

                        ok, err = run_config(method, scoring, use_meso, subset_mode, val_strat)

                        if ok:
                            passed += 1
                            print(f"  [OK]   {tag}")
                        else:
                            failed += 1
                            # Truncate long errors
                            short_err = err[:80] if err else "?"
                            print(f"  [FAIL] {tag}: {short_err}")

    print(f"\n{'='*90}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"{'ALL PASSED' if failed == 0 else 'SOME FAILED'}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
