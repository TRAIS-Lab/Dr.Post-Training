"""
Fused kernels (drpt/kernels: CuTe DSL and Triton backends) vs the PyTorch reference.

Checks, on CUDA, for every importable backend, that gip/pip scores and the
selected-sample weight gradient match an fp64 reference at least as well as the
bf16 PyTorch path does, on odd shapes (boundary predication), with bias, with V > 1,
with a single or an empty selection, and that the dispatch in drpt.selection.utils
takes the fused path only when it should (3-D CUDA half precision, supported shapes).

    python tests/test_fused_kernels.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drpt import kernels  # noqa: E402
from drpt.selection import utils as su  # noqa: E402

DEV = "cuda"


def _rel(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _data(B, S, O, I, V, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    go = torch.randn(B, S, O, device=DEV, dtype=torch.float32, generator=g).to(dtype) * 1e-2
    inp = torch.randn(B, S, I, device=DEV, dtype=torch.float32, generator=g).to(dtype)
    vgo = torch.randn(V, S, O, device=DEV, dtype=torch.float32, generator=g).to(dtype) * 1e-2
    vinp = torch.randn(V, S, I, device=DEV, dtype=torch.float32, generator=g).to(dtype)
    return go, inp, vgo, vinp


def _ref_scores(go, inp, vgo, vinp):
    G = torch.einsum("vto,vti->oi", vgo.double(), vinp.double())
    return torch.einsum("bso,bsi,oi->b", go.double(), inp.double(), G)


SHAPES = [  # (B, S, O, I, V)
    (3, 200, 1000, 1504, 1),     # S, O, I not multiples of the 128/128/32 tiles
    (8, 512, 1024, 2048, 1),     # k_proj-like
    (4, 128, 384, 256, 3),       # V > 1
    (2, 1024, 2048, 6144, 1),    # gate_proj-like, longer sequence
    (3, 387, 960, 2560, 2),      # SmolLM2 widths, odd sequence length
    (1, 64, 64, 64, 1),          # tiny
]


def _ops():
    return kernels.kernel_ops()


def test_gip_matches_fp64():
    gip_scores = _ops().gip_scores
    for (B, S, O, I, V) in SHAPES:
        if not _ops().supports("gip", (B, S, O), (B, S, I)):
            print(f"  gip {(B, S, O, I, V)}: shape not supported by {kernels.backend()} backend (reference path)")
            continue
        go, inp, vgo, vinp = _data(B, S, O, I, V)
        ref = _ref_scores(go, inp, vgo, vinp)
        fused = gip_scores(go, inp, vgo, vinp)
        first = gip_scores(go, inp, vgo, vinp)  # second call: same result (autotune must not leak into out)
        assert fused.dtype == torch.float32 and fused.shape == (B,)
        e = _rel(fused, ref)
        assert e < 2e-3, f"gip shape {(B, S, O, I, V)}: rel err {e:.2e}"
        assert torch.allclose(fused, first, rtol=1e-4, atol=1e-6), "gip not deterministic across calls"
        print(f"  gip {(B, S, O, I, V)}: rel err {e:.1e}")


def test_pip_matches_fp64():
    pip_scores = _ops().pip_scores
    for (B, S, O, I, V) in SHAPES:
        if not _ops().supports("pip", (B, S, O), (B, S, I)):
            print(f"  pip {(B, S, O, I, V)}: shape not supported by {kernels.backend()} backend (reference path)")
            continue
        go, inp, vgo, vinp = _data(B, S, O, I, V)
        ref = _ref_scores(go, inp, vgo, vinp)
        G = torch.einsum("vto,vti->oi", vgo, vinp)  # bf16 G, as the reference path uses
        fused = pip_scores(go, inp, G)
        pip_ref, _ = su._compute_scores_and_similarity_ref(go, inp, vgo, vinp, None, False)
        e_f, e_r = _rel(fused, ref), _rel(pip_ref, ref)
        assert e_f < 2e-2 and e_f <= 2 * e_r + 1e-3, f"pip shape {(B, S, O, I, V)}: fused {e_f:.2e} vs ref {e_r:.2e}"
        print(f"  pip {(B, S, O, I, V)}: fused rel err {e_f:.1e} (torch ref {e_r:.1e})")


def test_selected_wgrad_matches_fp64():
    selected_wgrad = _ops().selected_wgrad
    for (B, S, O, I, V) in SHAPES:
        if not _ops().supports("wgrad", (B, S, O), (B, S, I)):
            print(f"  wgrad {(B, S, O, I)}: shape not supported by {kernels.backend()} backend (reference path)")
            continue
        go, inp, _, _ = _data(B, S, O, I, V)
        for sel in (torch.arange(0, B, 2, device=DEV), torch.tensor([B - 1], device=DEV), torch.empty(0, dtype=torch.long, device=DEV)):
            scale = torch.tensor(1.75, device=DEV)
            for has_bias in (False, True):
                w, b = selected_wgrad(go, inp, sel, scale, has_bias)
                assert w.shape == (O, I) and w.dtype == go.dtype
                if sel.numel() == 0:
                    assert not w.any() and (b is None or not b.any())
                    continue
                w_ref = torch.einsum("kso,ksi->oi", go[sel].double(), inp[sel].double()) * 1.75
                e = _rel(w, w_ref)
                assert e < 5e-3, f"wgrad {(B, S, O, I)} K={sel.numel()}: rel err {e:.2e}"
                if has_bias:
                    b_ref = go[sel].double().sum(dim=(0, 1)) * 1.75
                    eb = _rel(b, b_ref)
                    assert eb < 5e-3, f"bias grad rel err {eb:.2e}"
                else:
                    assert b is None
        print(f"  wgrad {(B, S, O, I)}: ok (K in {{{B // 2 + B % 2}, 1, 0}}, bias on/off)")


def test_total_wgrad_matches_fp64():
    total_wgrad = _ops().total_wgrad
    for (B, S, O, I, V) in SHAPES:
        if not _ops().supports("wgrad", (B, S, O), (B, S, I)):
            print(f"  total_wgrad {(B, S, O, I)}: shape not supported by {kernels.backend()} backend (reference path)")
            continue
        go, inp, vgo, vinp = _data(B, S, O, I, V)
        for a, c in ((go, inp), (vgo, vinp)):
            w = total_wgrad(a, c)
            assert w.shape == (O, I) and w.dtype == a.dtype
            e = _rel(w, torch.einsum("bso,bsi->oi", a.double(), c.double()))
            assert e < 5e-3, f"total_wgrad {tuple(a.shape)}: rel err {e:.2e}"
            # the shared helper must dispatch here for half-precision CUDA data
            e2 = _rel(su.compute_total_gradient(a, c), w)
            assert e2 < 1e-6, f"compute_total_gradient did not use the fused kernel: {e2:.2e}"
        print(f"  total_wgrad {(B, S, O, I)}: ok (B={B} and V={V} rows)")


def test_compressed_grad_matches_reference():
    """CuTe-only: the fused compress path equals Compressor.forward's reference math and dispatches inside Compressor."""
    if kernels.backend() != "cute":
        print("  compressed projection kernel is CuTe-only; skipped")
        return
    from drpt.compressor import Compressor, Sparsifier, Projector
    from drpt.projection import random_project
    from drpt.kernels.cute_ops import compressed_grad

    def ref(go, inp, P_O, P_I, k):
        B, S, O = go.shape
        c1 = (go.reshape(-1, O) @ P_O) / (k ** 0.5)
        c2 = (inp.reshape(-1, inp.shape[-1]) @ P_I) / (k ** 0.5)
        return torch.einsum("bsi,bsj->bij", c1.reshape(B, S, -1), c2.reshape(B, S, -1)).reshape(B, -1)

    for (B, S, O, I, k) in [(9, 512, 1024, 2048, 64), (3, 387, 960, 2560, 64), (4, 300, 1000, 1504, 32)]:
        go, inp, _, _ = _data(B, S, O, I, 1)
        g = torch.Generator(device=DEV).manual_seed(1)
        P_O = torch.randn(O, k, device=DEV, generator=g).to(torch.bfloat16)
        P_I = torch.randn(I, k, device=DEV, generator=g).to(torch.bfloat16)
        r64 = ref(go.double(), inp.double(), P_O.double(), P_I.double(), k)
        fused = compressed_grad(go, inp, P_O, P_I, scale=1.0 / k)
        e_f, e_r = _rel(fused, r64), _rel(ref(go, inp, P_O, P_I, k), r64)
        assert fused.shape == (B, k * k) and fused.dtype == go.dtype
        assert e_f < 1e-2 and e_f <= 1.5 * e_r + 1e-3, f"compressed_grad {(B, S, O, I, k)}: fused {e_f:.2e} vs ref {e_r:.2e}"
        print(f"  compressed_grad {(B, S, O, I, k)}: fused rel err {e_f:.1e} (torch ref {e_r:.1e})")

    # end-to-end through a Compressor with dense normal sparsifiers and an identity projector
    B, S, O, I, k = 5, 256, 512, 768, 64
    go, inp, _, _ = _data(B, S, O, I, 1)
    comp = Compressor("layer", 0)
    sp = Sparsifier("layer", 0)
    s1 = random_project(torch.zeros(1, O, device=DEV, dtype=torch.bfloat16), 1, proj_dim=k, proj_max_batch_size=64,
                        proj_seed=3, proj_type="normal", device=torch.device(DEV))
    s2 = random_project(torch.zeros(1, I, device=DEV, dtype=torch.bfloat16), 1, proj_dim=k, proj_max_batch_size=64,
                        proj_seed=4, proj_type="normal", device=torch.device(DEV))
    s1.project(torch.zeros(1, O, device=DEV, dtype=torch.bfloat16), ensemble_id=0)
    s2.project(torch.zeros(1, I, device=DEV, dtype=torch.bfloat16), ensemble_id=0)
    sp.sparsifier_comp = (s1, s2)
    sp.intermediate_dims = (k, k)
    pr = Projector("layer", 0)
    pr.projector = random_project(torch.zeros(1, k * k, device=DEV, dtype=torch.bfloat16), 1, proj_dim=k * k,
                                  proj_max_batch_size=64, proj_seed=5, proj_type="identity", device=torch.device(DEV))
    comp.sparsifier, comp.projector = sp, pr
    fused = comp._fused_components((go, inp))
    assert fused is not None, "Compressor did not take the fused path"
    reference = comp._forward_ref((go, inp))
    e = _rel(fused, reference.double())
    assert fused.shape == reference.shape and fused.dtype == reference.dtype and e < 1e-2, f"Compressor fused vs ref {e:.2e}"
    assert torch.equal(comp.forward((go, inp)), fused)
    print(f"  Compressor.forward fused vs reference rel diff {e:.1e}; 2-D / bias-augmented inputs fall back: "
          f"{comp._fused_components((go[:, 0], inp[:, 0])) is None}")


def test_score_select_matches_torch():
    """CuTe-only: scores and sorted top-k from compressed-gradient partials equal the torch chain."""
    if kernels.backend() != "cute":
        print("  score_select kernel is CuTe-only; skipped")
        return
    from drpt.kernels.cute_ops import score_select, compressed_partials, compressed_grad
    g = torch.Generator(device=DEV).manual_seed(7)
    for (B_total, n_train, tiles, k) in [(9, 8, 8, 4), (9, 8, 1, 4), (3, 2, 4, 1), (17, 16, 8, 8), (12, 8, 3, 0), (66, 64, 2, 32)]:
        P = torch.randn(B_total, tiles, 64, 64, device=DEV, generator=g) * 1e-2
        corr = torch.tensor(1.7, device=DEV)
        scores, sel = score_select(P, n_train, k, corr)
        c = P.double().sum(1).reshape(B_total, -1)
        ref = (c[:n_train] @ c[n_train:].sum(0)) * 1.7
        ref_sel = torch.topk(ref, k).indices.sort().values if k > 0 else sel.new_empty(0)
        e = _rel(scores, ref)
        assert e < 1e-5 and torch.equal(sel, ref_sel), f"score_select {(B_total, n_train, tiles, k)}: err {e:.1e}, sel {sel.tolist()} vs {ref_sel.tolist()}"
        print(f"  score_select B={B_total} n={n_train} tiles={tiles} k={k}: rel err {e:.1e}, selection identical")
    # partials are consistent with compressed_grad (k1 = k2 = 32 exercises the zero padding)
    go, inp, _, _ = _data(5, 200, 512, 768, 1)
    P_O = torch.randn(512, 32, device=DEV, generator=g).to(torch.bfloat16)
    P_I = torch.randn(768, 32, device=DEV, generator=g).to(torch.bfloat16)
    parts = compressed_partials(go, inp, P_O, P_I, scale=1 / 32)
    full = compressed_grad(go, inp, P_O, P_I, scale=1 / 32)
    summed = parts.sum(1)
    assert not summed[:, 32:, :].any() and not summed[:, :, 32:].any(), "padding columns must be zero"
    e = _rel(summed[:, :32, :32].reshape(5, -1).to(torch.bfloat16), full)
    assert e < 1e-2, f"partials vs compressed_grad {e:.1e}"
    print(f"  compressed_partials consistent with compressed_grad (rel {e:.1e}), padding zero")


def test_reduce_select_and_partials():
    """CuTe-only: reduce_select equals the torch chain and pip/gip partials sum to the scores."""
    if kernels.backend() != "cute":
        print("  reduce_select is CuTe-only; skipped")
        return
    from drpt.kernels.cute_ops import reduce_select, pip_partials, pip_scores, gip_partials, gip_scores
    g = torch.Generator(device=DEV).manual_seed(11)
    for (n, R, k) in [(8, 32, 4), (8, 1, 4), (2, 16, 1), (16, 64, 8), (8, 33, 0), (64, 7, 32)]:
        P = torch.randn(n, R, device=DEV, generator=g)
        corr = torch.tensor(0.7, device=DEV, dtype=torch.bfloat16)  # bf16 correction like state.score_correction
        scores, sel = reduce_select(P, corr, k)
        ref = P.double().sum(1) * corr.double().item()  # the kernel applies the (bf16-rounded) tensor value
        ref_sel = torch.topk(ref, k).indices.sort().values if k > 0 else sel.new_empty(0)
        e = _rel(scores, ref)
        assert e < 1e-5 and torch.equal(sel, ref_sel), f"reduce_select {(n, R, k)}: err {e:.1e}, {sel.tolist()} vs {ref_sel.tolist()}"
    print("  reduce_select: scores and selections identical to torch on 6 shapes")
    go, inp, vgo, vinp = _data(5, 200, 320, 448, 2)
    G = torch.einsum("vto,vti->oi", vgo, vinp)
    assert torch.allclose(pip_partials(go, inp, G).sum(1), pip_scores(go, inp, G), rtol=1e-5, atol=1e-7)
    assert torch.allclose(gip_partials(go, inp, vgo, vinp).sum(1), gip_scores(go, inp, vgo, vinp), rtol=1e-5, atol=1e-7)
    print("  pip/gip partials sum to the fused scores")


def test_dispatch_matches_reference():
    """drpt.selection.utils entry points: fused on vs off agree, and only eligible inputs take the fused path."""
    from unittest import mock
    go, inp, vgo, vinp = _data(4, 96, 320, 192, 2)
    sel = torch.tensor([0, 3], device=DEV)
    scale = torch.tensor(2.0, device=DEV)
    out = {}
    active_backend = kernels.backend()
    for on in (True, False):
        kernels.set_fused_kernels(on)
        assert kernels.fused_kernels_enabled() == on
        s_gip, _ = su.compute_scores_gip(go, inp, vgo, vinp, None, False)
        s_pip, _ = su.compute_scores_and_similarity(go, inp, vgo, vinp, None, False)
        w, b = su.compute_selected_gradients(go, inp, sel, True, scale)
        out[on] = (s_gip.double(), s_pip.double(), w.double(), b.double())
    kernels.set_backend(active_backend)
    for name, a, r in zip(("gip", "pip", "wgrad", "bgrad"), out[True], out[False]):
        e = _rel(a, r)
        assert e < 2e-2, f"dispatch {name}: fused vs reference rel err {e:.2e}"
        print(f"  dispatch {name}: fused vs reference rel err {e:.1e}")
    # 2-D inputs, fp32 inputs and CPU tensors must not be routed to the kernels
    mod = f"drpt.kernels.{kernels.backend()}_ops"
    boom = dict(side_effect=AssertionError("fused path taken"))
    with mock.patch(f"{mod}.gip_scores", **boom), mock.patch(f"{mod}.pip_scores", **boom), \
         mock.patch(f"{mod}.selected_wgrad", **boom):
        su.compute_scores_gip(go[:, 0], inp[:, 0], vgo[:, 0], vinp[:, 0], None, False)
        su.compute_scores_and_similarity(go.float(), inp.float(), vgo.float(), vinp.float(), None, False)
        su.compute_selected_gradients(go.cpu(), inp.cpu(), sel.cpu(), False, scale.cpu())
    assert not kernels.fused_ok(go, inp.float()) and not kernels.fused_ok(go[:, 0], inp[:, 0]) and kernels.fused_ok(go, inp)
    print("  dispatch: ineligible inputs take the reference path")
    # and switching the kernels off routes eligible inputs to the reference path too
    active = kernels.backend()
    kernels.set_fused_kernels(False)
    try:
        with mock.patch(f"{mod}.gip_scores", **boom), mock.patch(f"{mod}.pip_scores", **boom), \
             mock.patch(f"{mod}.selected_wgrad", **boom):
            su.compute_scores_gip(go, inp, vgo, vinp, None, False)
            su.compute_scores_and_similarity(go, inp, vgo, vinp, None, False)
            su.compute_selected_gradients(go, inp, sel, False, scale)
    finally:
        kernels.set_backend(active)
    print("  dispatch: set_fused_kernels(False) routes to the reference path")


ALL_TESTS = [
    test_gip_matches_fp64,
    test_pip_matches_fp64,
    test_selected_wgrad_matches_fp64,
    test_total_wgrad_matches_fp64,
    test_compressed_grad_matches_reference,
    test_score_select_matches_torch,
    test_reduce_select_and_partials,
    test_dispatch_matches_reference,
]

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required; skipping")
        sys.exit(0)
    backends = [b for b, ok in (("cute", kernels.HAS_CUTE), ("triton", kernels.HAS_TRITON)) if ok]
    if not backends:
        print("no fused-kernel backend importable; skipping")
        sys.exit(0)
    failed = total = 0
    for b in backends:
        kernels.set_backend(b)
        print(f"===== backend: {b}")
        for t in ALL_TESTS:
            total += 1
            print(f"[{t.__name__}]")
            try:
                t()
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"  FAILED: {e}")
    print(f"\n{total - failed}/{total} tests passed")
    sys.exit(1 if failed else 0)
