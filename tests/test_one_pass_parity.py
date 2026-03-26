#!/usr/bin/env python
"""
Verification tests for one-pass vs two-pass subset descent parity,
and scoring method consistency.

Tests:
1. One-pass vs two-pass subset produce identical selected indices
2. One-pass vs two-pass produce identical weight gradients (after scale correction)
3. Different scoring methods (ghost, ghost_greats, direct) produce identical scores
4. Model weights are identical after one optimizer step with one-pass vs two-pass
"""

import copy
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')


# =============================================================================
# Small test model (no pretrained download needed)
# =============================================================================

class SmallCausalLM(nn.Module):
    """Tiny causal LM for testing: embedding -> 2 transformer-like linear blocks -> lm_head."""

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
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()


class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # Fixed seed for reproducibility
        gen = torch.Generator().manual_seed(42)
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), generator=gen)
        self.labels = self.input_ids.clone()
        # Mask first half of each sequence (prompt, not scored)
        self.labels[:, :seq_len // 2] = -100

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
            'labels': self.labels[idx],
        }


# =============================================================================
# Helpers
# =============================================================================

def get_hookable_layer_names(model):
    """Get names of all Linear layers to hook."""
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def collect_param_grads(model, layer_names):
    """Collect weight gradients for hooked layers as a dict."""
    grads = {}
    for name, module in model.named_modules():
        if name in layer_names:
            if module.weight.grad is not None:
                grads[name + '.weight'] = module.weight.grad.clone()
            if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
                grads[name + '.bias'] = module.bias.grad.clone()
    return grads


def collect_all_params(model):
    """Snapshot all model parameters."""
    return {n: p.clone() for n, p in model.named_parameters()}


def compare_grads(grads_a, grads_b, label_a="A", label_b="B", atol=1e-5, rtol=1e-4):
    """Compare two gradient dicts, return True if all match."""
    all_match = True
    keys = set(grads_a.keys()) | set(grads_b.keys())
    for key in sorted(keys):
        if key not in grads_a:
            print(f"  MISSING in {label_a}: {key}")
            all_match = False
            continue
        if key not in grads_b:
            print(f"  MISSING in {label_b}: {key}")
            all_match = False
            continue
        ga, gb = grads_a[key], grads_b[key]
        if ga.shape != gb.shape:
            print(f"  SHAPE MISMATCH {key}: {ga.shape} vs {gb.shape}")
            all_match = False
            continue
        max_diff = (ga - gb).abs().max().item()
        rel_diff = ((ga - gb).abs() / (gb.abs().max() + 1e-12)).max().item()
        match = torch.allclose(ga, gb, atol=atol, rtol=rtol)
        status = "OK" if match else "FAIL"
        if not match:
            all_match = False
        print(f"  [{status}] {key}: max_abs_diff={max_diff:.2e}, max_rel_diff={rel_diff:.2e}")
    return all_match


def compare_params(params_a, params_b, label_a="A", label_b="B", atol=1e-5, rtol=1e-4):
    """Compare two parameter dicts."""
    all_match = True
    for key in sorted(params_a.keys()):
        pa, pb = params_a[key], params_b[key]
        max_diff = (pa - pb).abs().max().item()
        match = torch.allclose(pa, pb, atol=atol, rtol=rtol)
        if not match:
            rel_diff = ((pa - pb).abs() / (pb.abs().max() + 1e-12)).max().item()
            print(f"  [FAIL] {key}: max_abs_diff={max_diff:.2e}, max_rel_diff={rel_diff:.2e}")
            all_match = False
    if all_match:
        print(f"  All parameters match between {label_a} and {label_b}")
    return all_match


# =============================================================================
# Test 1: Scoring method consistency (ghost vs ghost_greats vs direct)
# =============================================================================

def test_scoring_methods():
    """Verify ghost, ghost_greats, and direct produce identical scores."""
    print("\n" + "=" * 70)
    print("TEST 1: Scoring method consistency (ghost vs ghost_greats vs direct)")
    print("=" * 70)

    from drpt.selection.utils import (
        compute_scores_and_similarity,
        compute_scores_ghost_greats,
        compute_scores_direct_materialization,
    )

    torch.manual_seed(42)
    B, S, O, I = 8, 16, 32, 64

    # 3D case (sequential, the common LLM case)
    print("\n--- 3D case (sequential) ---")
    train_go = torch.randn(B, S, O)
    train_inp = torch.randn(B, S, I)
    val_go = torch.randn(2, S, O)
    val_inp = torch.randn(2, S, I)

    scores_ghost, sim_ghost = compute_scores_and_similarity(
        train_go, train_inp, val_go, val_inp, None, True)
    scores_greats, sim_greats = compute_scores_ghost_greats(
        train_go, train_inp, val_go, val_inp, None, True)
    scores_direct, sim_direct = compute_scores_direct_materialization(
        train_go, train_inp, val_go, val_inp, None, True)

    print(f"  ghost scores:   {scores_ghost}")
    print(f"  greats scores:  {scores_greats}")
    print(f"  direct scores:  {scores_direct}")

    score_match_3d_greats = torch.allclose(scores_ghost, scores_greats, atol=1e-4, rtol=1e-3)
    score_match_3d_direct = torch.allclose(scores_ghost, scores_direct, atol=1e-4, rtol=1e-3)
    sim_match_3d_greats = torch.allclose(sim_ghost, sim_greats, atol=1e-4, rtol=1e-3)
    sim_match_3d_direct = torch.allclose(sim_ghost, sim_direct, atol=1e-4, rtol=1e-3)

    print(f"  Scores ghost==greats: {score_match_3d_greats} (max diff: {(scores_ghost - scores_greats).abs().max():.2e})")
    print(f"  Scores ghost==direct: {score_match_3d_direct} (max diff: {(scores_ghost - scores_direct).abs().max():.2e})")
    print(f"  Similarity ghost==greats: {sim_match_3d_greats} (max diff: {(sim_ghost - sim_greats).abs().max():.2e})")
    print(f"  Similarity ghost==direct: {sim_match_3d_direct} (max diff: {(sim_ghost - sim_direct).abs().max():.2e})")

    # 2D case (non-sequential)
    print("\n--- 2D case (non-sequential) ---")
    train_go_2d = torch.randn(B, O)
    train_inp_2d = torch.randn(B, I)
    val_go_2d = torch.randn(2, O)
    val_inp_2d = torch.randn(2, I)

    scores_ghost_2d, sim_ghost_2d = compute_scores_and_similarity(
        train_go_2d, train_inp_2d, val_go_2d, val_inp_2d, None, True)
    scores_greats_2d, sim_greats_2d = compute_scores_ghost_greats(
        train_go_2d, train_inp_2d, val_go_2d, val_inp_2d, None, True)
    scores_direct_2d, sim_direct_2d = compute_scores_direct_materialization(
        train_go_2d, train_inp_2d, val_go_2d, val_inp_2d, None, True)

    score_match_2d_greats = torch.allclose(scores_ghost_2d, scores_greats_2d, atol=1e-4, rtol=1e-3)
    score_match_2d_direct = torch.allclose(scores_ghost_2d, scores_direct_2d, atol=1e-4, rtol=1e-3)

    print(f"  Scores ghost==greats: {score_match_2d_greats} (max diff: {(scores_ghost_2d - scores_greats_2d).abs().max():.2e})")
    print(f"  Scores ghost==direct: {score_match_2d_direct} (max diff: {(scores_ghost_2d - scores_direct_2d).abs().max():.2e})")

    # With precomputed val_grad_total
    print("\n--- With val_grad_total (stored val mode) ---")
    val_grad_total = torch.einsum('vso,vsi->oi', val_go, val_inp)
    scores_ghost_t, _ = compute_scores_and_similarity(
        train_go, train_inp, None, None, val_grad_total, False)
    scores_greats_t, _ = compute_scores_ghost_greats(
        train_go, train_inp, None, None, val_grad_total, False)
    scores_direct_t, _ = compute_scores_direct_materialization(
        train_go, train_inp, None, None, val_grad_total, False)

    match_total_greats = torch.allclose(scores_ghost_t, scores_greats_t, atol=1e-4, rtol=1e-3)
    match_total_direct = torch.allclose(scores_ghost_t, scores_direct_t, atol=1e-4, rtol=1e-3)
    print(f"  Scores ghost==greats (val_total): {match_total_greats} (max diff: {(scores_ghost_t - scores_greats_t).abs().max():.2e})")
    print(f"  Scores ghost==direct (val_total): {match_total_direct} (max diff: {(scores_ghost_t - scores_direct_t).abs().max():.2e})")

    all_pass = all([score_match_3d_greats, score_match_3d_direct,
                    sim_match_3d_greats, sim_match_3d_direct,
                    score_match_2d_greats, score_match_2d_direct,
                    match_total_greats, match_total_direct])
    print(f"\n  TEST 1 {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# Test 2: One-pass vs two-pass subset - SeparateBatch mode
# =============================================================================

def test_one_pass_vs_two_pass_separate_batch():
    """
    Verify one-pass and two-pass subset produce identical:
    - Selected indices
    - Weight gradients
    - Model weights after optimizer step
    Uses SeparateBatch mode where scale factors are provably identical.
    """
    print("\n" + "=" * 70)
    print("TEST 2: One-pass vs two-pass subset (SeparateBatch)")
    print("=" * 70)

    from drpt.hook import GradientHook
    from drpt.selection.strategies import (
        SeparateBatchSubsetStrategy,
        SeparateBatchSubsetOnePassStrategy,
    )

    device = 'cpu'
    torch.manual_seed(123)
    vocab_size, hidden_dim, seq_len = 128, 64, 16
    train_bs, val_bs = 8, 2
    frac = 0.5

    # Create model and data
    model_orig = SmallCausalLM(vocab_size, hidden_dim, seq_len, num_layers=2).to(device)
    dataset = DummyDataset(train_bs + val_bs, seq_len, vocab_size)
    train_batch = {k: v[:train_bs].to(device) for k, v in
                   {k: torch.stack([dataset[i][k] for i in range(train_bs)]) for k in dataset[0]}.items()}
    val_batch = {k: v.to(device) for k, v in
                 {k: torch.stack([dataset[i][k] for i in range(train_bs, train_bs + val_bs)]) for k in dataset[0]}.items()}

    layer_names = get_hookable_layer_names(model_orig)
    lr = 1e-3

    results = {}
    for mode_name, StrategyClass in [
        ("two_pass", SeparateBatchSubsetStrategy),
        ("one_pass", SeparateBatchSubsetOnePassStrategy),
    ]:
        print(f"\n--- Running {mode_name} ---")
        # Fresh model copy
        model = copy.deepcopy(model_orig)
        hook = GradientHook(model=model, layer_names=layer_names, device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        strategy = StrategyClass(
            grad_hook=hook, frac=frac, use_second_order=False,
            selection_mode="topk", scoring_method="ghost",
        )

        # Capture val gradients
        hook.start_val_capture(use_factorized=True)
        model.zero_grad()
        val_out = model(**val_batch)
        val_out.loss.backward()
        hook.end_val_capture()

        # Training step
        def compute_train_loss():
            out = model(**train_batch)
            return out.loss, {}

        loss, stats = strategy.execute_training_step(
            model=model,
            batch_size=train_bs,
            compute_loss_fn=compute_train_loss,
            lr=lr,
            labels=train_batch['labels'],
            filter_batch_fn=lambda indices: (
                lambda: (model(**{
                    'input_ids': train_batch['input_ids'][indices],
                    'attention_mask': train_batch['attention_mask'][indices],
                    'labels': train_batch['labels'][indices],
                }).loss, {})
            ) if mode_name == "two_pass" else None,
        )

        # Collect results before optimizer step
        grads = collect_param_grads(model, layer_names)
        n_selected = stats.get("selection/n_selected", "N/A")
        print(f"  Loss: {loss.item():.6f}, n_selected: {n_selected}")
        print(f"  Num grads collected: {len(grads)}")

        # Optimizer step
        optimizer.step()
        params_after = collect_all_params(model)

        results[mode_name] = {
            'grads': grads,
            'params': params_after,
            'loss': loss.item(),
            'n_selected': n_selected,
        }

        hook.clear_val_buffer()
        hook.remove_hooks()

    # Compare
    print("\n--- Comparing weight gradients ---")
    grad_match = compare_grads(
        results['one_pass']['grads'], results['two_pass']['grads'],
        "one_pass", "two_pass", atol=1e-5, rtol=1e-4
    )

    print("\n--- Comparing model parameters after optimizer step ---")
    param_match = compare_params(
        results['one_pass']['params'], results['two_pass']['params'],
        "one_pass", "two_pass", atol=1e-5, rtol=1e-4
    )

    all_pass = grad_match and param_match
    print(f"\n  TEST 2 {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# Test 3: One-pass vs two-pass subset - MergedBatch mode
# =============================================================================

def test_one_pass_vs_two_pass_merged_batch():
    """
    Verify one-pass and two-pass subset produce identical results
    with MergedBatch mode (tests the batch_total_tokens scale correction).
    """
    print("\n" + "=" * 70)
    print("TEST 3: One-pass vs two-pass subset (MergedBatch)")
    print("=" * 70)

    from drpt.hook import GradientHook
    from drpt.selection.strategies import (
        MergedBatchSubsetStrategy,
        MergedBatchSubsetOnePassStrategy,
    )

    device = 'cpu'
    torch.manual_seed(456)
    vocab_size, hidden_dim, seq_len = 128, 64, 16
    train_bs, val_bs = 8, 2
    frac = 0.5

    model_orig = SmallCausalLM(vocab_size, hidden_dim, seq_len, num_layers=2).to(device)
    dataset = DummyDataset(train_bs + val_bs, seq_len, vocab_size)
    all_data = {k: torch.stack([dataset[i][k] for i in range(train_bs + val_bs)]).to(device)
                for k in dataset[0]}
    train_batch = {k: v[:train_bs] for k, v in all_data.items()}
    merged_batch = all_data  # train + val merged

    layer_names = get_hookable_layer_names(model_orig)
    lr = 1e-3

    results = {}
    for mode_name, StrategyClass in [
        ("two_pass", MergedBatchSubsetStrategy),
        ("one_pass", MergedBatchSubsetOnePassStrategy),
    ]:
        print(f"\n--- Running {mode_name} ---")
        model = copy.deepcopy(model_orig)
        hook = GradientHook(model=model, layer_names=layer_names, device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        strategy = StrategyClass(
            grad_hook=hook, frac=frac, use_second_order=False,
            selection_mode="topk", scoring_method="ghost",
        )

        def compute_loss(mdl, batch):
            out = mdl(**batch)
            return out.loss

        loss = strategy.execute_training_step(
            model=model,
            merged_batch=merged_batch,
            train_batch_size=train_bs,
            compute_loss_fn=compute_loss,
            lr=lr,
            batch_train=train_batch,
        )

        grads = collect_param_grads(model, layer_names)
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Num grads collected: {len(grads)}")

        optimizer.step()
        params_after = collect_all_params(model)

        results[mode_name] = {
            'grads': grads,
            'params': params_after,
            'loss': loss.item(),
        }

        hook.remove_hooks()

    print("\n--- Comparing weight gradients ---")
    grad_match = compare_grads(
        results['one_pass']['grads'], results['two_pass']['grads'],
        "one_pass", "two_pass", atol=1e-5, rtol=1e-4
    )

    print("\n--- Comparing model parameters after optimizer step ---")
    param_match = compare_params(
        results['one_pass']['params'], results['two_pass']['params'],
        "one_pass", "two_pass", atol=1e-5, rtol=1e-4
    )

    all_pass = grad_match and param_match
    print(f"\n  TEST 3 {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# Test 4: Scoring methods produce same selection in full pipeline
# =============================================================================

def test_scoring_methods_in_pipeline():
    """
    Verify different scoring methods produce identical selected indices
    and identical weight gradients in the full SeparateBatch Subset pipeline.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Scoring methods produce same results in pipeline")
    print("=" * 70)

    from drpt.hook import GradientHook
    from drpt.selection.strategies import SeparateBatchSubsetOnePassStrategy

    device = 'cpu'
    vocab_size, hidden_dim, seq_len = 128, 64, 16
    train_bs, val_bs = 8, 2
    frac = 0.5

    model_orig = SmallCausalLM(vocab_size, hidden_dim, seq_len, num_layers=2).to(device)
    torch.manual_seed(789)
    dataset = DummyDataset(train_bs + val_bs, seq_len, vocab_size)
    train_batch = {k: torch.stack([dataset[i][k] for i in range(train_bs)]).to(device) for k in dataset[0]}
    val_batch = {k: torch.stack([dataset[i][k] for i in range(train_bs, train_bs + val_bs)]).to(device) for k in dataset[0]}

    layer_names = get_hookable_layer_names(model_orig)
    lr = 1e-3

    results = {}
    for scoring_method in ["ghost", "ghost_greats", "direct"]:
        print(f"\n--- scoring_method={scoring_method} ---")
        torch.manual_seed(789)  # Reset seed for reproducibility
        model = copy.deepcopy(model_orig)
        hook = GradientHook(model=model, layer_names=layer_names, device=device)

        strategy = SeparateBatchSubsetOnePassStrategy(
            grad_hook=hook, frac=frac, use_second_order=False,
            selection_mode="topk", scoring_method=scoring_method,
        )

        # Val capture
        hook.start_val_capture(use_factorized=True)
        model.zero_grad()
        val_out = model(**val_batch)
        val_out.loss.backward()
        hook.end_val_capture()

        # Train step
        def compute_train_loss():
            out = model(**train_batch)
            return out.loss, {}

        loss, stats = strategy.execute_training_step(
            model=model, batch_size=train_bs,
            compute_loss_fn=compute_train_loss, lr=lr,
            labels=train_batch['labels'],
        )

        grads = collect_param_grads(model, layer_names)
        print(f"  Loss: {loss.item():.6f}, n_selected: {stats.get('selection/n_selected', 'N/A')}")

        results[scoring_method] = {'grads': grads, 'loss': loss.item()}

        hook.clear_val_buffer()
        hook.remove_hooks()

    # Compare ghost vs ghost_greats
    print("\n--- Comparing ghost vs ghost_greats ---")
    match_greats = compare_grads(
        results['ghost']['grads'], results['ghost_greats']['grads'],
        "ghost", "ghost_greats", atol=1e-4, rtol=1e-3
    )

    # Compare ghost vs direct
    print("\n--- Comparing ghost vs direct ---")
    match_direct = compare_grads(
        results['ghost']['grads'], results['direct']['grads'],
        "ghost", "direct", atol=1e-4, rtol=1e-3
    )

    all_pass = match_greats and match_direct
    print(f"\n  TEST 4 {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Dr. Post-Training: One-Pass vs Two-Pass Parity Tests")
    print("=" * 70)

    results = []

    results.append(("Scoring method consistency", test_scoring_methods()))
    results.append(("One-pass vs two-pass (SeparateBatch)", test_one_pass_vs_two_pass_separate_batch()))
    results.append(("One-pass vs two-pass (MergedBatch)", test_one_pass_vs_two_pass_merged_batch()))
    results.append(("Scoring methods in pipeline", test_scoring_methods_in_pipeline()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
