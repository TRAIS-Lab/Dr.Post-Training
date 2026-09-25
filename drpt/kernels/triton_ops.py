"""
Triton kernels behind :mod:`drpt.kernels`.

All three kernels are tensor-core GEMMs whose *epilogue* replaces the follow-up
PyTorch ops of the reference implementation, so no intermediate ever reaches
HBM.  Accumulation is fp32 throughout; only the final result is rounded.

Notation: ``B`` training samples, ``V`` validation samples, ``S`` sequence
length, ``O``/``I`` output/input features of the Linear layer,
``go = dL/d(pre-activation)`` ``[*, S, O]``, ``inp`` = layer input ``[*, S, I]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _swizzle(pid, num_m, num_n, GROUP: tl.constexpr):
    """Grouped tile ordering so tiles sharing an operand row-block run close together (L2 reuse)."""
    num_in_group = GROUP * num_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP
    gsize = tl.minimum(num_m - first_m, GROUP)
    pid_m = first_m + (pid % num_in_group) % gsize
    pid_n = (pid % num_in_group) // gsize
    return pid_m, pid_n


# =============================================================================
# Ghost inner product (GIP)
#   s_b = sum_v sum_{s,t} (go_b[s] . go_v[t]) * (inp_b[s] . inp_v[t])
# One program per (b, v, s-tile, t-tile): two GEMM tiles with K = O and K = I,
# Hadamard product and full reduction in registers, one atomic add per program.
# Never materialises the [B, V, S, S] pair of dot-product tensors.
# =============================================================================

def _gip_configs():
    # Autotuned on A40 (sm_86) over 512 <= S <= 4096, 1024 <= O, I <= 12288: 128x128x32 with 8 warps
    # wins everywhere; the 64-row tiles only matter for very short sequences.
    return [
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 64, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_gip_configs(), key=["S", "O", "I", "V"], reset_to_zero=["out"])
@triton.jit
def _gip_kernel(go_t, inp_t, go_v, inp_v, out, S, O, I, V,
                s_gtb, s_gts, s_gto, s_itb, s_its, s_iti,
                s_gvb, s_gvs, s_gvo, s_ivb, s_ivs, s_ivi,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid_bv = tl.program_id(1)
    b = pid_bv // V
    v = pid_bv % V
    pid_s, pid_t = _swizzle(tl.program_id(0), tl.cdiv(S, BM), tl.cdiv(S, BN), GROUP)
    rs = pid_s * BM + tl.arange(0, BM)
    rt = pid_t * BN + tl.arange(0, BN)
    ms = rs < S
    mt = rt < S
    acc_a = tl.zeros((BM, BN), dtype=tl.float32)
    acc_c = tl.zeros((BM, BN), dtype=tl.float32)
    gt_base = go_t + b.to(tl.int64) * s_gtb
    gv_base = go_v + v.to(tl.int64) * s_gvb
    for k0 in range(0, O, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < O
        a = tl.load(gt_base + rs[:, None] * s_gts + rk[None, :] * s_gto, mask=ms[:, None] & mk[None, :], other=0.0)
        bt = tl.load(gv_base + rk[:, None] * s_gvo + rt[None, :] * s_gvs, mask=mk[:, None] & mt[None, :], other=0.0)
        acc_a = tl.dot(a, bt, acc_a)
    it_base = inp_t + b.to(tl.int64) * s_itb
    iv_base = inp_v + v.to(tl.int64) * s_ivb
    for k0 in range(0, I, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < I
        a = tl.load(it_base + rs[:, None] * s_its + rk[None, :] * s_iti, mask=ms[:, None] & mk[None, :], other=0.0)
        bt = tl.load(iv_base + rk[:, None] * s_ivi + rt[None, :] * s_ivs, mask=mk[:, None] & mt[None, :], other=0.0)
        acc_c = tl.dot(a, bt, acc_c)
    partial = tl.sum(tl.sum(acc_a * acc_c, axis=1), axis=0)
    tl.atomic_add(out + b, partial)


def gip_scores(go_t: torch.Tensor, inp_t: torch.Tensor, go_v: torch.Tensor, inp_v: torch.Tensor) -> torch.Tensor:
    """Ghost inner product scores.

    Args:
        go_t: training grad_output ``[B, S, O]``
        inp_t: training input ``[B, S, I]``
        go_v: validation grad_output ``[V, S, O]``
        inp_v: validation input ``[V, S, I]``
    Returns:
        fp32 scores ``[B]`` = ``<g_b, sum_v g_v>`` with ``g = go^T inp``.
    """
    B, S, O = go_t.shape
    I = inp_t.shape[2]
    V = go_v.shape[0]
    if not (inp_t.shape[:2] == (B, S) and go_v.shape[1:] == (S, O) and inp_v.shape == (V, S, I)):
        raise ValueError("gip_scores: inconsistent shapes")
    go_t, inp_t, go_v, inp_v = (_inner_contiguous(t) for t in (go_t, inp_t, go_v, inp_v))
    out = torch.zeros(B, dtype=torch.float32, device=go_t.device)
    grid = lambda meta: (triton.cdiv(S, meta["BM"]) * triton.cdiv(S, meta["BN"]), B * V)
    _gip_kernel[grid](go_t, inp_t, go_v, inp_v, out, S, O, I, V,
                      *go_t.stride(), *inp_t.stride(), *go_v.stride(), *inp_v.stride())
    return out


# =============================================================================
# Per-token inner product (PIP)
#   s_b = sum_{s,o} go_b[s, o] * (inp_b[s, :] . G[o, :]),  G = sum_v go_v^T inp_v  [O, I]
# GEMM inp_b @ G^T with the [B, S, O] result never stored: each tile is dotted
# with the matching go tile in registers and reduced.
# =============================================================================

def _pip_configs():
    # A40 winners: 128x128x32 / 4 warps for O <= 3072, 256-wide tiles / 8 warps for the MLP widths.
    return [
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_pip_configs(), key=["S", "O", "I"], reset_to_zero=["out"])
@triton.jit
def _pip_kernel(inp, G, go, out, S, O, I,
                s_ib, s_is, s_ii, s_Go, s_Gi, s_gb, s_gs, s_go,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    b = tl.program_id(1)
    pid_s, pid_o = _swizzle(tl.program_id(0), tl.cdiv(S, BM), tl.cdiv(O, BN), GROUP)
    rs = pid_s * BM + tl.arange(0, BM)
    ro = pid_o * BN + tl.arange(0, BN)
    ms = rs < S
    mo = ro < O
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ibase = inp + b.to(tl.int64) * s_ib
    for k0 in range(0, I, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < I
        a = tl.load(ibase + rs[:, None] * s_is + rk[None, :] * s_ii, mask=ms[:, None] & mk[None, :], other=0.0)
        g = tl.load(G + rk[:, None] * s_Gi + ro[None, :] * s_Go, mask=mk[:, None] & mo[None, :], other=0.0)
        acc = tl.dot(a, g, acc)
    gt = tl.load(go + b.to(tl.int64) * s_gb + rs[:, None] * s_gs + ro[None, :] * s_go,
                 mask=ms[:, None] & mo[None, :], other=0.0)
    partial = tl.sum(tl.sum(acc * gt.to(tl.float32), axis=1), axis=0)
    tl.atomic_add(out + b, partial)


def pip_scores(go_t: torch.Tensor, inp_t: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Per-token inner product scores against the total validation gradient.

    Args:
        go_t: training grad_output ``[B, S, O]``
        inp_t: training input ``[B, S, I]``
        G: total validation weight gradient ``[O, I]`` (same dtype as ``go_t``)
    Returns:
        fp32 scores ``[B]``.
    """
    B, S, O = go_t.shape
    I = inp_t.shape[2]
    if G.shape != (O, I) or inp_t.shape[:2] != (B, S):
        raise ValueError("pip_scores: inconsistent shapes")
    if G.dtype != go_t.dtype:
        G = G.to(go_t.dtype)
    go_t, inp_t, G = _inner_contiguous(go_t), _inner_contiguous(inp_t), _inner_contiguous(G)
    out = torch.zeros(B, dtype=torch.float32, device=go_t.device)
    grid = lambda meta: (triton.cdiv(S, meta["BM"]) * triton.cdiv(O, meta["BN"]), B)
    _pip_kernel[grid](inp_t, G, go_t, out, S, O, I,
                      *inp_t.stride(), *G.stride(), *go_t.stride())
    return out


# =============================================================================
# Selected-sample weight gradient
#   grad_W = scale * sum_{k in sel} go_k^T inp_k        [O, I]
#   grad_b = scale * sum_{k in sel} sum_s go_k[s]       [O]     (optional)
# GEMM with M = O, N = I, K = |sel| * S, where the K index walks the selected
# samples through an index list, so nothing is gathered/copied.  The item-count
# scale and the bias column-sum are fused into the same pass.
# =============================================================================

def _wgrad_configs():
    # A40 winners: 128x128x32 / 4 warps for |sel|*S <= 4096, 256-wide tiles / 8 warps for long sequences.
    return [
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "GROUP": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP": 8}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_wgrad_configs(), key=["S", "O", "I", "K"])
@triton.jit
def _sel_wgrad_kernel(go, inp, sel, scale_ptr, out, bias_out, K, S, O, I,
                      s_gb, s_gs, s_go, s_ib, s_is, s_ii, s_oo, s_oi,
                      HAS_BIAS: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid_m, pid_n = _swizzle(tl.program_id(0), tl.cdiv(O, BM), tl.cdiv(I, BN), GROUP)
    ro = pid_m * BM + tl.arange(0, BM)
    ri = pid_n * BN + tl.arange(0, BN)
    mo = ro < O
    mi = ri < I
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    bacc = tl.zeros((BM,), dtype=tl.float32)
    for kk in range(0, K):
        bidx = tl.load(sel + kk).to(tl.int64)
        gbase = go + bidx * s_gb
        ibase = inp + bidx * s_ib
        for s0 in range(0, S, BK):
            rs = s0 + tl.arange(0, BK)
            msk = rs < S
            a = tl.load(gbase + ro[:, None] * s_go + rs[None, :] * s_gs, mask=mo[:, None] & msk[None, :], other=0.0)
            bt = tl.load(ibase + rs[:, None] * s_is + ri[None, :] * s_ii, mask=msk[:, None] & mi[None, :], other=0.0)
            acc = tl.dot(a, bt, acc)
            if HAS_BIAS:
                bacc += tl.sum(a.to(tl.float32), axis=1)
    scale = tl.load(scale_ptr).to(tl.float32)
    tl.store(out + ro[:, None] * s_oo + ri[None, :] * s_oi, (acc * scale).to(out.dtype.element_ty),
             mask=mo[:, None] & mi[None, :])
    if HAS_BIAS:
        if pid_n == 0:
            tl.store(bias_out + ro, (bacc * scale).to(bias_out.dtype.element_ty), mask=mo)


def selected_wgrad(go: torch.Tensor, inp: torch.Tensor, sel: torch.Tensor, scale, has_bias: bool,
                   out_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Weight (and bias) gradient aggregated over the selected samples.

    Args:
        go: grad_output ``[B, S, O]``
        inp: input ``[B, S, I]``
        sel: selected sample indices ``[K]`` (any integer dtype, CUDA)
        scale: scalar (0-dim tensor or float) multiplied into the result
        has_bias: also return the bias gradient ``[O]``
        out_dtype: output dtype (default: ``go.dtype``)
    Returns:
        ``(grad_weight [O, I], grad_bias [O] or None)``
    """
    B, S, O = go.shape
    I = inp.shape[2]
    out_dtype = out_dtype or go.dtype
    out = torch.empty(O, I, dtype=out_dtype, device=go.device)
    bias = torch.empty(O, dtype=out_dtype, device=go.device) if has_bias else None
    K = sel.numel()
    if K == 0:
        out.zero_()
        if has_bias:
            bias.zero_()
        return out, bias
    if sel.dtype != torch.int64:
        sel = sel.to(torch.int64)
    if not sel.is_contiguous():
        sel = sel.contiguous()
    scale_t = torch.as_tensor(scale, device=go.device, dtype=torch.float32).reshape(1)
    go, inp = _inner_contiguous(go), _inner_contiguous(inp)
    grid = lambda meta: (triton.cdiv(O, meta["BM"]) * triton.cdiv(I, meta["BN"]),)
    _sel_wgrad_kernel[grid](go, inp, sel, scale_t, out, bias if has_bias else out, K, S, O, I,
                            *go.stride(), *inp.stride(), *out.stride(), HAS_BIAS=has_bias)
    return out, bias


def supports(op: str, *shapes) -> bool:
    """The Triton kernels mask every dimension, so any shape is supported."""
    return op in ("pip", "gip", "wgrad")


def total_wgrad(go: torch.Tensor, inp: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """sum_{b,s} go[b,s,:] (x) inp[b,s,:] -> [O, I] via :func:`selected_wgrad` over all samples."""
    sel = torch.arange(go.shape[0], device=go.device, dtype=torch.int64)
    return selected_wgrad(go, inp, sel, 1.0, False, out_dtype)[0]


def _inner_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Kernels stream the last dim; make sure it is unit-stride (batch/seq strides are passed explicitly)."""
    return t if t.stride(-1) == 1 else t.contiguous()


__all__ = ["gip_scores", "pip_scores", "selected_wgrad", "supports"]
