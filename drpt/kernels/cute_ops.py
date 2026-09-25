"""
CuTe DSL kernels (NVIDIA CUTLASS Python DSL) for the drpt custom Linear backward.

Target: Ampere-class tensor cores (sm_80/86/89) via ``cp.async`` multistage gmem->smem
pipelining, ``ldmatrix`` smem->register copies and ``mma.sync.m16n8k16`` (bf16/fp16 in,
fp32 accumulate).  The same code runs on Hopper (sm_90) through the compatibility
path; a wgmma/TMA mainloop would be the Hopper-native alternative.

Three fused ops share one tiled mainloop (:meth:`AmpereFusedGemm.mainloop`) and differ
in operand majorness and epilogue:

  pip    partial[b, tile]  = sum_{s,o in tile} (inp_b[s,:] . G[o,:]) * go_b[s,o]
         A = inp_b (S x I, K-major), B = G (O x I, K-major); the C tile is dotted with the
         matching go tile in registers; one fp32 partial per CTA, summed on the host.
  gip    partial[bv, tile] = sum_{s,t in tile} (go_b[s,:] . go_v[t,:]) (inp_b[s,:] . inp_v[t,:])
         two K-major mainloops (K = O, then K = I) into two accumulators, Hadamard product and
         CTA reduction in registers -> no [B, V, S, S] intermediates.
  wgrad  W[o, i] = scale * sum_{k in sel} sum_s go[k, s, o] inp[k, s, i]
         A = go_k^T (O x S, M-major), B = inp_k (I x S, N-major); the K loop walks the selected
         samples through the index list (no gathered copies); scale fused into the bf16 store.

Boundary handling: rows (M) / columns (N) are predicated at copy-atom granularity and the
last K tile is predicated per K slice, so any sequence length works and feature dims only
have to be multiples of 8 (16-byte rows).

  proj   compressed-scoring projection (go P_O)^T (inp P_I) per 64-row tile (see compressed_grad /
         compressed_partials); ScoreSelect is its single-CTA epilogue that turns the tile partials of
         the merged batch into scores against the validation rows and a sorted top-k selection.

Kernels are compiled once per (op, dtype) with symbolic shapes and called through the
TVM-FFI entry point with torch tensors directly (about 25 us of host overhead per call).
Layout/tiling helpers follow NVIDIA's ``examples/python/CuTeDSL/cute/ampere`` GEMM.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Type

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

K_MAJOR, MN_MAJOR = "k", "mn"


class AmpereFusedGemm:
    """Shared tile configuration + mainloop; `op` selects the epilogue / operand majorness."""

    # Tuned on A40 (sm_86): pip/wgrad 128x128x32, 4 warps, 3 stages (48 KB smem -> 2 CTAs/SM);
    # gip keeps two 128x128 fp32 accumulators live, so 8 warps (4x2 atoms) keep them at 64 regs/thread,
    # and a 64-wide K tile halves the barrier count of its two mainloops.
    CONFIG = {
        "pip": dict(bm=128, bn=128, bk=32, stages=3, atoms=(2, 2, 1)),
        "gip": dict(bm=128, bn=128, bk=64, stages=3, atoms=(4, 2, 1)),
        "wgrad": dict(bm=128, bn=128, bk=32, stages=3, atoms=(2, 2, 1)),
        # compressed projection: N = kappa^(1/2) <= 64, small M tile for CTA-level parallelism (B * S/64 CTAs)
        "proj": dict(bm=64, bn=64, bk=64, stages=3, atoms=(2, 2, 1)),
    }

    def __init__(self, op: str, ab_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
                 num_stages: Optional[int] = None, atom_layout_mnk=None, bk: Optional[int] = None):
        assert op in ("pip", "gip", "wgrad", "proj")
        cfg = self.CONFIG[op]
        self.op = op
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.cta_tiler = (cfg["bm"], cfg["bn"], bk or cfg["bk"])
        self.bM, self.bN, self.bK = self.cta_tiler
        self.num_stages = num_stages or cfg["stages"]
        self.atom_layout_mnk = atom_layout_mnk or cfg["atoms"]
        self.num_warps = self.atom_layout_mnk[0] * self.atom_layout_mnk[1] * self.atom_layout_mnk[2]
        self.num_threads = 32 * self.num_warps
        self.mma_inst_shape = (16, 8, 16)
        self.a_major = MN_MAJOR if op == "wgrad" else K_MAJOR
        self.b_major = MN_MAJOR if op in ("wgrad", "proj") else K_MAJOR
        # The last K tile is issued slice-by-slice with per-slice predicates, so K only has to be a
        # multiple of the 8-element copy atom: S (wgrad) can be any padded sequence length and the
        # feature dims (pip/gip/proj) any multiple of 8 — e.g. 32-wide k/v projections with bK = 64.
        self.k_residue = True

    # ------------------------------------------------------------------ host (jit) side
    @cute.jit
    def __call__(self, t0: cute.Tensor, t1: cute.Tensor, t2: cute.Tensor, t3: cute.Tensor, t4: cute.Tensor,
                 stream: cuda.CUstream):
        """Argument conventions (all torch-contiguous); ``stream`` is torch's current stream, supplied by
        the TVM-FFI runtime (compiled with a ``use_tvm_ffi_env_stream`` placeholder, not a call argument):
        pip  : t0=inp (B,S,I)  t1=G (O,I)      t2=go (B,S,O)   t3=G (unused dup)  t4=out (B, tiles) f32
        gip  : t0=go_t (B,S,O) t1=go_v (V,S,O) t2=inp_t (B,S,I) t3=inp_v (V,S,I) t4=out (B*V, tiles) f32
        wgrad: t0=go (B,S,O)   t1=inp (B,S,I)  t2=sel (K,) i64  t3=scale (1,) f32 t4=out (O,I) bf16
        """
        cb = 128  # copy bits
        sA_layout, sA_swz = self._smem_layout(self.ab_dtype, self.a_major, cb, (self.bM, self.bK, self.num_stages))
        sB_layout, sB_swz = self._smem_layout(self.ab_dtype, self.b_major, cb, (self.bN, self.bK, self.num_stages))
        atom_g2s = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                                       self.ab_dtype, num_bits_per_copy=cb)
        tiled_copy_A = self._gmem_tiled_copy(atom_g2s, self.ab_dtype, self.a_major, cb, self.bM)
        tiled_copy_B = self._gmem_tiled_copy(atom_g2s, self.ab_dtype, self.b_major, cb, self.bN)
        mma_op = warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        perm = (self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.atom_layout_mnk[2] * self.mma_inst_shape[2])
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout(self.atom_layout_mnk), permutation_mnk=perm)

        if cutlass.const_expr(self.op == "pip"):
            mA = cute.make_tensor(t0.iterator, cute.select(t0.layout, mode=[1, 2, 0]))   # (S, I, B) = (M, K, L)
            mB = t1                                                                      # (O, I)    = (N, K)
            mE = cute.make_tensor(t2.iterator, cute.select(t2.layout, mode=[1, 2, 0]))   # (S, O, B) = (M, N, L)
            mA2, mB2 = mA, mB
            m_tiles = cute.ceil_div(cute.size(mA.shape[0]), self.bM)
            n_tiles = cute.ceil_div(cute.size(mB.shape[0]), self.bN)
            L = cute.size(mA.shape[2])
        elif cutlass.const_expr(self.op == "gip"):
            mA = cute.make_tensor(t0.iterator, cute.select(t0.layout, mode=[1, 2, 0]))   # go_t  (S, O, B)
            mB = cute.make_tensor(t1.iterator, cute.select(t1.layout, mode=[1, 2, 0]))   # go_v  (S, O, V)
            mA2 = cute.make_tensor(t2.iterator, cute.select(t2.layout, mode=[1, 2, 0]))  # inp_t (S, I, B)
            mB2 = cute.make_tensor(t3.iterator, cute.select(t3.layout, mode=[1, 2, 0]))  # inp_v (S, I, V)
            mE = mA
            m_tiles = cute.ceil_div(cute.size(mA.shape[0]), self.bM)
            n_tiles = cute.ceil_div(cute.size(mB.shape[0]), self.bN)
            L = cute.size(mA.shape[2]) * cute.size(mB.shape[2])
        else:
            mA = cute.make_tensor(t0.iterator, cute.select(t0.layout, mode=[2, 1, 0]))   # go  -> (O, S, B) = (M, K, L) M-major
            mB = cute.make_tensor(t1.iterator, cute.select(t1.layout, mode=[2, 1, 0]))   # inp -> (I, S, B) = (N, K, L) N-major
            mA2, mB2, mE = mA, mB, mA
            m_tiles = cute.ceil_div(cute.size(mA.shape[0]), self.bM)
            n_tiles = cute.ceil_div(cute.size(mB.shape[0]), self.bN)
            L = 1

        raster = 1
        if n_tiles > 5:
            raster = 8
        elif n_tiles > 2:
            raster = 4
        elif n_tiles > 1:
            raster = 2
        grid = (m_tiles * raster, (n_tiles + raster - 1) // raster, L)
        self.kernel(mA, mB, mA2, mB2, mE, t2, t3, t4, sA_layout, sA_swz, sB_layout, sB_swz,
                    tiled_copy_A, tiled_copy_B, tiled_mma, raster).launch(
            grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    # ------------------------------------------------------------------ device side
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mA2: cute.Tensor, mB2: cute.Tensor, mE: cute.Tensor,
               mSel: cute.Tensor, mScale: cute.Tensor, mOut: cute.Tensor,
               sA_layout: cute.Layout, sA_swz: cute.Swizzle, sB_layout: cute.Layout, sB_swz: cute.Swizzle,
               tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy, tiled_mma: cute.TiledMma,
               raster: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, l_idx = cute.arch.block_idx()
        m_tiles = cute.ceil_div(cute.size(mA.shape[0]), self.bM)
        n_tiles = cute.ceil_div(cute.size(mB.shape[0]), self.bN)
        m_blk = bidx // raster
        n_blk = (bidx % raster) + bidy * raster
        if m_blk >= m_tiles or n_blk >= n_tiles:
            pass
        else:
            tiler_coord = (m_blk, n_blk, None)

            @cute.struct
            class SharedStorage:
                a: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(sA_layout)], 128]
                b: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(sB_layout)], 128]
                red: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.num_warps], 16]

            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sA = storage.a.get_tensor(sA_layout, swizzle=sA_swz)
            sB = storage.b.get_tensor(sB_layout, swizzle=sB_swz)
            sRed = storage.red.get_tensor(cute.make_layout(self.num_warps))

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAsA = thr_copy_A.partition_D(sA)
            tBsB = thr_copy_B.partition_D(sB)

            # Boundary predication (M for A, N for B) at copy-atom granularity, as in NVIDIA's tensorop_gemm:
            # identity tensors tiled/partitioned like the operands give each copy atom its (row, k) coordinate.
            mcA = cute.make_identity_tensor(mA.shape)
            mcB = cute.make_identity_tensor(mB.shape)
            if cutlass.const_expr(self.op == "pip"):
                cA = cute.local_tile(mcA[None, None, 0], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                cB = cute.local_tile(mcB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
            else:
                cA = cute.local_tile(mcA[None, None, 0], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                cB = cute.local_tile(mcB[None, None, 0], self.cta_tiler, tiler_coord, proj=(None, 1, 1))
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)
            tApA = cute.make_rmem_tensor(
                cute.make_layout((tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                                 stride=(cute.size(tAsA, mode=[1]), 1, 0)), cutlass.Boolean)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout((tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                                 stride=(cute.size(tBsB, mode=[1]), 1, 0)), cutlass.Boolean)
            for rest_v in cutlass.range_constexpr(tApA.shape[0]):
                for m in cutlass.range_constexpr(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in cutlass.range_constexpr(tBpB.shape[0]):
                for n in cutlass.range_constexpr(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])
            # predicated-off smem elements are never written, so zero them once
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), self.acc_dtype)
            acc.fill(0.0)

            atom_s2r_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(self.a_major == MN_MAJOR, 4), self.ab_dtype)
            atom_s2r_B = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(self.b_major == MN_MAJOR, 4), self.ab_dtype)
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_s2r_B.get_slice(tidx)
            tCsA_view = thr_s2r_A.partition_S(sA)
            tCrA_view = thr_s2r_A.retile(tCrA)
            tCsB_view = thr_s2r_B.partition_S(sB)
            tCrB_view = thr_s2r_B.retile(tCrB)

            if cutlass.const_expr(self.op == "wgrad"):
                # K walks the selected samples: for each sel[k], a full pipelined pass over its S/bK tiles.
                n_sel = cute.size(mSel.shape[0])
                for kk in range(n_sel):
                    sidx = mSel[kk]
                    gA = cute.local_tile(mA[None, None, sidx], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                    gB = cute.local_tile(mB[None, None, sidx], self.cta_tiler, tiler_coord, proj=(None, 1, 1))
                    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
                    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
                    tAgA = thr_copy_A.partition_S(gA)
                    tBgB = thr_copy_B.partition_S(gB)
                    self.mainloop(tiled_copy_A, tiled_copy_B, tAgA, tAsA, tBgB, tBsB, tAcA, tBcB, tApA, tBpB, mA.shape[1],
                                  tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc, tiled_mma)
                scale = mScale[0]
                gC = cute.local_tile(mOut, self.cta_tiler, tiler_coord, proj=(1, 1, None))          # (bM, bN) of (O, I)
                tCgC = thr_mma.partition_C(gC)
                tCrD = cute.make_fragment_like(tCgC, mOut.element_type)
                tCrD.store((acc.load() * scale).to(mOut.element_type))
                if (m_blk + 1) * self.bM <= cute.size(mOut.shape[0]) and (n_blk + 1) * self.bN <= cute.size(mOut.shape[1]):
                    cute.autovec_copy(tCrD, tCgC)
                else:
                    cC = cute.local_tile(cute.make_identity_tensor(mOut.shape), self.cta_tiler, tiler_coord, proj=(1, 1, None))
                    tCcC = thr_mma.partition_C(cC)
                    for i in cutlass.range_constexpr(cute.size(tCrD)):
                        if cute.elem_less(tCcC[i][0], mOut.shape[0]) and cute.elem_less(tCcC[i][1], mOut.shape[1]):
                            tCgC[i] = tCrD[i]
            else:
                if cutlass.const_expr(self.op == "pip"):
                    b = l_idx
                    gA = cute.local_tile(mA[None, None, b], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                    gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
                    gE = cute.local_tile(mE[None, None, b], self.cta_tiler, tiler_coord, proj=(1, 1, None))
                    gE = cute.make_tensor(gE.iterator.align(16), gE.layout)
                else:
                    V = cute.size(mB.shape[2])
                    b = l_idx // V
                    v = l_idx % V
                    gA = cute.local_tile(mA[None, None, b], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                    gB = cute.local_tile(mB[None, None, v], self.cta_tiler, tiler_coord, proj=(None, 1, 1))
                    gA2 = cute.local_tile(mA2[None, None, b], self.cta_tiler, tiler_coord, proj=(1, None, 1))
                    gB2 = cute.local_tile(mB2[None, None, v], self.cta_tiler, tiler_coord, proj=(None, 1, 1))
                    gA2 = cute.make_tensor(gA2.iterator.align(16), gA2.layout)
                    gB2 = cute.make_tensor(gB2.iterator.align(16), gB2.layout)
                gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
                gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
                tAgA = thr_copy_A.partition_S(gA)
                tBgB = thr_copy_B.partition_S(gB)
                self.mainloop(tiled_copy_A, tiled_copy_B, tAgA, tAsA, tBgB, tBsB, tAcA, tBcB, tApA, tBpB, mA.shape[1],
                              tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc, tiled_mma)
                part = cutlass.Float32(0.0)
                if cutlass.const_expr(self.op == "pip"):
                    tCgE = thr_mma.partition_C(gE)
                    tCrE = cute.make_fragment_like(tCgE, self.ab_dtype)
                    if (m_blk + 1) * self.bM <= cute.size(mE.shape[0]) and (n_blk + 1) * self.bN <= cute.size(mE.shape[1]):
                        cute.autovec_copy(tCgE, tCrE)
                    else:
                        tCrE.fill(0)
                        cE = cute.local_tile(cute.make_identity_tensor(mE.shape)[None, None, 0], self.cta_tiler, tiler_coord, proj=(1, 1, None))
                        tCcE = thr_mma.partition_C(cE)
                        for i in cutlass.range_constexpr(cute.size(tCrE)):
                            if cute.elem_less(tCcE[i][0], mE.shape[0]) and cute.elem_less(tCcE[i][1], mE.shape[1]):
                                tCrE[i] = tCgE[i]
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        part = part + acc[i] * tCrE[i].to(cutlass.Float32)
                else:
                    acc2 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), self.acc_dtype)
                    acc2.fill(0.0)
                    tAgA2 = thr_copy_A.partition_S(gA2)
                    tBgB2 = thr_copy_B.partition_S(gB2)
                    self.mainloop(tiled_copy_A, tiled_copy_B, tAgA2, tAsA, tBgB2, tBsB, tAcA, tBcB, tApA, tBpB, mA2.shape[1],
                                  tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc2, tiled_mma)
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        part = part + acc[i] * acc2[i]
                part = cute.arch.warp_reduction_sum(part)
                warp_id = tidx // 32
                lane = tidx % 32
                if lane == 0:
                    sRed[warp_id] = part
                cute.arch.sync_threads()
                if tidx == 0:
                    total = cutlass.Float32(0.0)
                    for w in cutlass.range_constexpr(self.num_warps):
                        total = total + sRed[w]
                    mOut[l_idx, m_blk * n_tiles + n_blk] = total

    # ------------------------------------------------------------------ compressed projection ("proj")
    @cute.jit
    def call_proj(self, t0: cute.Tensor, t1: cute.Tensor, t2: cute.Tensor, t3: cute.Tensor, t4: cute.Tensor,
                  scale: cutlass.Float32, stream: cuda.CUstream):
        """proj: t0=go (B,S,O)  t1=inp (B,S,I)  t2=P_O (O,k1)  t3=P_I (I,k2)  t4=out (B, S_tiles, 64, 64) f32
        out[b, tile] = scale * (go_b[tile] @ P_O)^T (inp_b[tile] @ P_I): the per-sample gradient go^T inp
        in the Kronecker-projected space; the sum over tiles is done on the host."""
        cb = 128
        sA_layout, sA_swz = self._smem_layout(self.ab_dtype, K_MAJOR, cb, (self.bM, self.bK, self.num_stages))
        sB_layout, sB_swz = self._smem_layout(self.ab_dtype, MN_MAJOR, cb, (self.bN, self.bK, self.num_stages))
        # stage-3 staging tiles (bM x 64, plain row-major) reuse the pipeline buffers; the tile is
        # small enough that the bank conflicts of the unswizzled ldmatrix.trans reads do not matter
        s3_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
        atom_g2s = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                                       self.ab_dtype, num_bits_per_copy=cb)
        tiled_copy_A = self._gmem_tiled_copy(atom_g2s, self.ab_dtype, K_MAJOR, cb, self.bM)
        tiled_copy_B = self._gmem_tiled_copy(atom_g2s, self.ab_dtype, MN_MAJOR, cb, self.bN)
        mma_op = warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        perm = (self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.atom_layout_mnk[2] * self.mma_inst_shape[2])
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout(self.atom_layout_mnk), permutation_mnk=perm)
        mA = cute.make_tensor(t0.iterator, cute.select(t0.layout, mode=[1, 2, 0]))   # go  (S, O, B) = (M, K, L)
        mA2 = cute.make_tensor(t1.iterator, cute.select(t1.layout, mode=[1, 2, 0]))  # inp (S, I, B)
        mB = cute.make_tensor(t2.iterator, cute.select(t2.layout, mode=[1, 0]))      # P_O^T (k1, O) = (N, K) N-major
        mB2 = cute.make_tensor(t3.iterator, cute.select(t3.layout, mode=[1, 0]))     # P_I^T (k2, I)
        m_tiles = cute.ceil_div(cute.size(mA.shape[0]), self.bM)
        L = cute.size(mA.shape[2])
        self.kernel_proj(mA, mA2, mB, mB2, t4, scale, sA_layout, sA_swz, sB_layout, sB_swz, s3_layout,
                         tiled_copy_A, tiled_copy_B, tiled_mma).launch(
            grid=(m_tiles, L, 1), block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel_proj(self, mA: cute.Tensor, mA2: cute.Tensor, mB: cute.Tensor, mB2: cute.Tensor, mOut: cute.Tensor,
                    scale: cutlass.Float32,
                    sA_layout: cute.Layout, sA_swz: cute.Swizzle, sB_layout: cute.Layout, sB_swz: cute.Swizzle,
                    s3_layout: cute.Layout,
                    tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy, tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        m_blk, b, _ = cute.arch.block_idx()
        tiler_coord = (m_blk, 0, None)

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(sA_layout)], 128]
            b: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(sB_layout)], 128]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.a.get_tensor(sA_layout, swizzle=sA_swz)
        sB = storage.b.get_tensor(sB_layout, swizzle=sB_swz)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAsA = thr_copy_A.partition_D(sA)
        tBsB = thr_copy_B.partition_D(sB)

        # predicates: rows of go/inp (S) and rows of P^T (k <= 64)
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        mcA2 = cute.make_identity_tensor(mA2.shape)
        mcB2 = cute.make_identity_tensor(mB2.shape)
        cA = cute.local_tile(mcA[None, None, 0], self.cta_tiler, tiler_coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        cA2 = cute.local_tile(mcA2[None, None, 0], self.cta_tiler, tiler_coord, proj=(1, None, 1))
        cB2 = cute.local_tile(mcB2, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        tAcA2 = thr_copy_A.partition_S(cA2)
        tBcB2 = thr_copy_B.partition_S(cB2)
        tApA = cute.make_rmem_tensor(
            cute.make_layout((tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                             stride=(cute.size(tAsA, mode=[1]), 1, 0)), cutlass.Boolean)
        tBpB = cute.make_rmem_tensor(
            cute.make_layout((tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                             stride=(cute.size(tBsB, mode=[1]), 1, 0)), cutlass.Boolean)
        tBpB2 = cute.make_rmem_tensor(
            cute.make_layout((tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                             stride=(cute.size(tBsB, mode=[1]), 1, 0)), cutlass.Boolean)
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for m in cutlass.range_constexpr(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
        for rest_v in cutlass.range_constexpr(tBpB.shape[0]):
            for n in cutlass.range_constexpr(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])
                tBpB2[rest_v, n, 0] = cute.elem_less(tBcB2[(0, rest_v), n, 0, 0][0], mB2.shape[0])
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        acc_a = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), self.acc_dtype)
        acc_c = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), self.acc_dtype)
        acc_a.fill(0.0)
        acc_c.fill(0.0)
        atom_s2r_A = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), self.ab_dtype)
        atom_s2r_B = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), self.ab_dtype)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_s2r_B.get_slice(tidx)
        tCsA_view = thr_s2r_A.partition_S(sA)
        tCrA_view = thr_s2r_A.retile(tCrA)
        tCsB_view = thr_s2r_B.partition_S(sB)
        tCrB_view = thr_s2r_B.retile(tCrB)

        # mainloop 1: acc_a = go_b[tile] @ P_O ; mainloop 2: acc_c = inp_b[tile] @ P_I
        gA = cute.local_tile(mA[None, None, b], self.cta_tiler, tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
        self.mainloop(tiled_copy_A, tiled_copy_B, thr_copy_A.partition_S(gA), tAsA, thr_copy_B.partition_S(gB), tBsB,
                      tAcA, tBcB, tApA, tBpB, mA.shape[1],
                      tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc_a, tiled_mma)
        gA2 = cute.local_tile(mA2[None, None, b], self.cta_tiler, tiler_coord, proj=(1, None, 1))
        gB2 = cute.local_tile(mB2, self.cta_tiler, tiler_coord, proj=(None, 1, 1))
        gA2 = cute.make_tensor(gA2.iterator.align(16), gA2.layout)
        gB2 = cute.make_tensor(gB2.iterator.align(16), gB2.layout)
        self.mainloop(tiled_copy_A, tiled_copy_B, thr_copy_A.partition_S(gA2), tAsA, thr_copy_B.partition_S(gB2), tBsB,
                      tAcA2, tBcB2, tApA, tBpB2, mA2.shape[1],
                      tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc_c, tiled_mma)

        # stage 3: out = acc_a^T @ acc_c over the tile rows, via bf16 smem staging (mirrors the reference rounding)
        sA3 = storage.a.get_tensor(s3_layout)   # (bM, 64) row-major: (s, k1)
        sC3 = storage.b.get_tensor(s3_layout)   # (bM, 64) row-major: (s, k2)
        rA = cute.make_fragment_like(acc_a, self.ab_dtype)
        rA.store(acc_a.load().to(self.ab_dtype))
        rC = cute.make_fragment_like(acc_c, self.ab_dtype)
        rC.store(acc_c.load().to(self.ab_dtype))
        cute.autovec_copy(rA, thr_mma.partition_C(sA3))
        cute.autovec_copy(rC, thr_mma.partition_C(sC3))
        cute.arch.sync_threads()
        sA3t = cute.composition(sA3, cute.make_layout((self.bN, self.bM), stride=(self.bM, 1)))   # (k1, s) = (M, K) M-major
        sC3t = cute.composition(sC3, cute.make_layout((self.bN, self.bM), stride=(self.bM, 1)))   # (k2, s) = (N, K) N-major
        atom_t = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), self.ab_dtype)
        tiled_t_A = cute.make_tiled_copy_A(atom_t, tiled_mma)
        tiled_t_B = cute.make_tiled_copy_B(atom_t, tiled_mma)
        thr_t_A = tiled_t_A.get_slice(tidx)
        thr_t_B = tiled_t_B.get_slice(tidx)
        tCrA3 = tiled_mma.make_fragment_A(thr_mma.partition_A(sA3t))
        tCrB3 = tiled_mma.make_fragment_B(thr_mma.partition_B(sC3t))
        cute.copy(tiled_t_A, thr_t_A.partition_S(sA3t), thr_t_A.retile(tCrA3))
        cute.copy(tiled_t_B, thr_t_B.partition_S(sC3t), thr_t_B.retile(tCrB3))
        acc3 = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bN, self.bN)), self.acc_dtype)
        acc3.fill(0.0)
        for kb in cutlass.range_constexpr(cute.size(tCrA3, mode=[2])):
            cute.gemm(tiled_mma, acc3, tCrA3[None, None, kb], tCrB3[None, None, kb], acc3)
        acc3.store(acc3.load() * scale)
        gOut = mOut[b, m_blk, None, None]   # (64, 64) f32
        cute.autovec_copy(acc3, thr_mma.partition_C(gOut))

    @cute.jit
    def issue_tile(self, tiled_copy, tg, ts, tc, tp, k_tile_index, stage, k_extent):
        """gmem->smem cp.async of k-tile `k_tile_index` into ring `stage`, predicated on rows (tp) and,
        for the last tile of a K dimension that is not a multiple of bK (wgrad: S), on columns too."""
        if cutlass.const_expr(self.k_residue):
            k_tile_count = cute.size(tg, mode=[3])
            if k_tile_index == k_tile_count - 1:
                for ks in cutlass.range_constexpr(cute.size(tg, mode=[2])):
                    if cute.elem_less(tc[0, 0, ks, k_tile_index][1], k_extent):
                        cute.copy(tiled_copy, tg[None, None, ks, k_tile_index], ts[None, None, ks, stage], pred=tp[None, None, ks])
                    else:
                        ts[None, None, ks, stage].fill(0)
            else:
                cute.copy(tiled_copy, tg[None, None, None, k_tile_index], ts[None, None, None, stage], pred=tp)
        else:
            cute.copy(tiled_copy, tg[None, None, None, k_tile_index], ts[None, None, None, stage], pred=tp)

    @cute.jit
    def mainloop(self, tiled_copy_A, tiled_copy_B, tAgA, tAsA, tBgB, tBsB, tAcA, tBcB, tApA, tBpB, k_extent,
                 tiled_s2r_A, tiled_s2r_B, tCsA_view, tCrA_view, tCsB_view, tCrB_view, tCrA, tCrB, acc, tiled_mma):
        """cp.async multistage (gmem->smem) + ldmatrix double buffering (smem->rmem) + mma.sync; acc += A @ B."""
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)
        for k_tile in range(num_smem_stages - 1):
            if k_tile < k_tile_count:
                self.issue_tile(tiled_copy_A, tAgA, tAsA, tAcA, tApA, k_tile_index, k_tile, k_extent)
                self.issue_tile(tiled_copy_B, tBgB, tBsB, tBcB, tBpB, k_tile_index, k_tile, k_extent)
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(num_smem_stages - 1)
        tCsA_p = tCsA_view[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_view[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_p[None, None, 0], tCrA_view[None, None, 0])
            cute.copy(tiled_s2r_B, tCsB_p[None, None, 0], tCrB_view[None, None, 0])
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_view[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_view[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                k_block_next = (k_block + 1) % num_k_block
                cute.copy(tiled_s2r_A, tCsA_p[None, None, k_block_next], tCrA_view[None, None, k_block_next])
                cute.copy(tiled_s2r_B, tCsB_p[None, None, k_block_next], tCrB_view[None, None, k_block_next])
                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        self.issue_tile(tiled_copy_A, tAgA, tAsA, tAcA, tApA, k_tile_index, smem_pipe_write, k_extent)
                        self.issue_tile(tiled_copy_B, tBgB, tBsB, tBcB, tBpB, k_tile_index, smem_pipe_write, k_extent)
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0
                cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    # ------------------------------------------------------------------ layout helpers (from NVIDIA's tensorop_gemm example)
    def _smem_layout(self, dtype, major, copy_bits, tiler):
        major_size = min(tiler[1] if major == K_MAJOR else tiler[0], 128 * 8 // dtype.width)
        swizzle_bits = min(int(math.log2(major_size * dtype.width // copy_bits)), 3)
        base_bits = int(math.log2(copy_bits // 8))
        shift_bits = int(math.log2(copy_bits // dtype.width))
        swz = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
        atom = (cute.make_layout((8, major_size), stride=(major_size, 1)) if major == K_MAJOR
                else cute.make_layout((major_size, 8), stride=(1, major_size)))
        return cute.tile_to_shape(atom, tiler, (0, 1, 2)), swz

    def _smem_layout_c(self, dtype, copy_bits, tiler):
        """Row-major (rows, cols) swizzled smem layout for storing accumulator fragments (NVIDIA example)."""
        major_size = tiler[1]
        swizzle_bits = min(int(math.log2(major_size * dtype.width // copy_bits)), 3)
        atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 4), 0,
                                         cute.make_layout((8, major_size), stride=(major_size, 1)))
        return cute.tile_to_shape(atom, tiler, (0, 1))

    def _gmem_tiled_copy(self, atom, dtype, major, copy_bits, rows):
        copy_elems = copy_bits // dtype.width
        if major == K_MAJOR:
            d1 = self.bK // copy_elems
            thr = cute.make_layout((self.num_threads // d1, d1), stride=(d1, 1))
            val = cute.make_layout((1, copy_elems))
        else:
            d0 = rows // copy_elems
            thr = cute.make_layout((d0, self.num_threads // d0), stride=(1, d0))
            val = cute.make_layout((copy_elems, 1))
        return cute.make_tiled_copy_tv(atom, thr, val)


# ---------------------------------------------------------------------- compile cache + torch entry points
_COMPILED = {}
_CUTE_DT = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_PROJ_BM = AmpereFusedGemm.CONFIG["proj"]["bm"]


def _fake3(dt, div_last=8):
    return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=div_last)),
                                    stride_order=(2, 1, 0), assumed_align=16)


def _fake2(dt, div=(8, 8)):
    return make_fake_compact_tensor(dt, (cute.sym_int(divisibility=div[0]), cute.sym_int(divisibility=div[1])),
                                    stride_order=(1, 0), assumed_align=16)


def _get(op, torch_dtype):
    """Compiled TVM-FFI entry for (op, dtype); shapes are symbolic so one binary serves all layers."""
    key = (op, torch_dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        dt = _CUTE_DT[torch_dtype]
        # The stream placeholder is bound by TVM-FFI to the caller's (torch) current stream at
        # every call and is not part of the compiled function's Python signature.
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        if op == "pip":
            args = (_fake3(dt), _fake2(dt), _fake3(dt), _fake2(dt), _fake2(cutlass.Float32, (1, 1)), stream)
        elif op == "gip":
            args = (_fake3(dt), _fake3(dt), _fake3(dt), _fake3(dt), _fake2(cutlass.Float32, (1, 1)), stream)
        elif op == "wgrad":
            sel = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8)
            scl = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
            args = (_fake3(dt), _fake3(dt), sel, scl, _fake2(dt), stream)
        else:  # proj
            out4 = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), cute.sym_int(), 64, 64),
                                            stride_order=(3, 2, 1, 0), assumed_align=16)
            args = (_fake3(dt), _fake3(dt), _fake2(dt), _fake2(dt), out4, cutlass.Float32(1.0), stream)
        entry = AmpereFusedGemm(op, dt)
        fn = cute.compile(entry.call_proj if op == "proj" else entry, *args, options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def _ok(*dims):
    return all(d % m == 0 for d, m in dims)


def _cdiv(a, b):
    return (a + b - 1) // b


def supports(op: str, *shapes) -> bool:
    """Whether the CuTe kernels handle these operands: feature dims must be multiples of 8
    (16-byte rows); any sequence length works."""
    if op == "pip":
        (B, S, O), (_, _, I) = shapes[0], shapes[1]
        return _ok((O, 8), (I, 8))
    if op == "gip":
        (B, S, O), (_, _, I) = shapes[0], shapes[1]
        return _ok((O, 8), (I, 8))
    if op == "wgrad":
        (B, S, O), (_, _, I) = shapes[0], shapes[1]
        return _ok((O, 8), (I, 8))
    if op == "proj":
        (B, S, O), (_, _, I) = shapes[0], shapes[1]
        return _ok((O, 8), (I, 8))
    return False


def pip_partials(go: torch.Tensor, inp: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Per-CTA partial sums of the per-token inner product scores: fp32 ``[B, tiles]``; ``.sum(1)`` = scores."""
    B, S, O = go.shape
    if not supports("pip", go.shape, inp.shape):
        raise ValueError("cute pip_scores: needs O % 8 == 0 and I % 8 == 0")
    if G.dtype != go.dtype:
        G = G.to(go.dtype)
    go, inp, G = go.contiguous(), inp.contiguous(), G.contiguous()
    out = torch.empty(B, _cdiv(S, 128) * _cdiv(O, 128), dtype=torch.float32, device=go.device)
    _get("pip", go.dtype)(inp, G, go, G, out)
    return out


def pip_scores(go: torch.Tensor, inp: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Per-token inner product scores: go [B,S,O], inp [B,S,I], G [O,I] (same dtype) -> fp32 [B]."""
    return pip_partials(go, inp, G).sum(dim=1)


def gip_partials(go_t: torch.Tensor, inp_t: torch.Tensor, go_v: torch.Tensor, inp_v: torch.Tensor) -> torch.Tensor:
    """Per-CTA partial sums of the ghost inner product scores: fp32 ``[B, V*tiles]``; ``.sum(1)`` = scores."""
    B, S, O = go_t.shape
    V = go_v.shape[0]
    if not supports("gip", go_t.shape, inp_t.shape):
        raise ValueError("cute gip_scores: needs O % 8 == 0 and I % 8 == 0")
    go_t, inp_t, go_v, inp_v = (t.contiguous() for t in (go_t, inp_t, go_v, inp_v))
    out = torch.empty(B * V, _cdiv(S, 128) ** 2, dtype=torch.float32, device=go_t.device)
    _get("gip", go_t.dtype)(go_t, go_v, inp_t, inp_v, out)
    return out.view(B, -1)


def gip_scores(go_t: torch.Tensor, inp_t: torch.Tensor, go_v: torch.Tensor, inp_v: torch.Tensor) -> torch.Tensor:
    """Ghost inner product scores: go_t [B,S,O], inp_t [B,S,I], go_v [V,S,O], inp_v [V,S,I] -> fp32 [B]."""
    return gip_partials(go_t, inp_t, go_v, inp_v).sum(dim=1)


def selected_wgrad(go: torch.Tensor, inp: torch.Tensor, sel: torch.Tensor, scale, has_bias: bool,
                   out_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Selected-sample weight gradient: go [B,S,O], inp [B,S,I], sel int [K], scale scalar
    -> (grad_weight [O,I] in out_dtype (default go.dtype), grad_bias [O] or None)."""
    B, S, O = go.shape
    I = inp.shape[2]
    if not supports("wgrad", go.shape, inp.shape):
        raise ValueError("cute selected_wgrad: needs O % 8 == 0 and I % 8 == 0")
    if not go.is_contiguous():
        go = go.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()
    out_dtype = out_dtype or go.dtype
    out = torch.empty(O, I, dtype=out_dtype, device=go.device)
    if sel.numel() == 0:
        out.zero_()
        return out, (torch.zeros(O, dtype=out_dtype, device=go.device) if has_bias else None)
    sel64 = sel if (sel.dtype == torch.int64 and sel.is_contiguous()) else sel.to(torch.int64).contiguous()
    if isinstance(scale, torch.Tensor) and scale.dtype == torch.float32 and scale.numel() == 1 and scale.is_cuda:
        scale_t = scale.reshape(1)
    else:
        scale_t = torch.as_tensor(scale, device=go.device, dtype=torch.float32).reshape(1)
    _get("wgrad", go.dtype)(go, inp, sel64, scale_t, out)
    bias = (go[sel64].sum(dim=(0, 1)).float() * scale_t).to(out_dtype) if has_bias else None
    return out, bias


_ARANGE: Dict[Tuple[int, torch.device], torch.Tensor] = {}
_ONE: Dict[torch.device, torch.Tensor] = {}


def total_wgrad(go: torch.Tensor, inp: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """sum_{b,s} go[b,s,:] (x) inp[b,s,:] -> [O, I]: the batch (e.g. target) weight gradient.

    The same mainloop as :func:`selected_wgrad` with every sample selected and scale 1;
    on A40 it runs ~20 % faster than the cuBLAS einsum once K = B*S exceeds ~2k and
    sidesteps its slow kernel choice on some [O, I] shapes."""
    key = (go.shape[0], go.device)
    sel = _ARANGE.get(key)
    if sel is None:
        sel = _ARANGE[key] = torch.arange(go.shape[0], device=go.device, dtype=torch.int64)
    one = _ONE.get(go.device)
    if one is None:
        one = _ONE[go.device] = torch.ones(1, device=go.device, dtype=torch.float32)
    return selected_wgrad(go, inp, sel, one, False, out_dtype)[0]


_SS_THREADS, _SS_COLS, _SS_MAX_ROWS = 256, 64 * 64, 256
NTHREADS, NCOLS, MAX_ROWS = _SS_THREADS, _SS_COLS, _SS_MAX_ROWS  # names used inside the kernel body


class ScoreSelect:
    """Single-CTA epilogue of compressed scoring (see :func:`score_select`)."""
    @cute.jit
    def __call__(self, mP: cute.Tensor, mCorr: cute.Tensor, mScores: cute.Tensor, mSel: cute.Tensor,
                 n_train: cutlass.Int32, k: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(mP, mCorr, mScores, mSel, n_train, k).launch(grid=(1, 1, 1), block=(NTHREADS, 1, 1), stream=stream)

    @cute.jit
    def argmax_step(self, best_v, best_i, off: cutlass.Constexpr):
        """One butterfly step of a warp argmax over (value, index); ties -> lowest index."""
        ov = cute.arch.shuffle_sync_bfly(best_v, offset=off)
        oi = cute.arch.shuffle_sync_bfly(best_i, offset=off)
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
        return best_v, best_i

    @cute.kernel
    def kernel(self, mP: cute.Tensor, mCorr: cute.Tensor, mScores: cute.Tensor, mSel: cute.Tensor,
               n_train: cutlass.Int32, k: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        lane = tidx % 32
        B_total = cute.size(mP.shape[0])
        tiles = cute.size(mP.shape[1])
        smem = cutlass.memory.SmemAllocator()
        s_val = smem.allocate_tensor(cutlass.Float32, cute.make_layout(NCOLS), 16)
        s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(8), 16)
        s_sc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(MAX_ROWS), 16)
        s_taken = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAX_ROWS), 16)
        s_sel = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAX_ROWS), 16)

        # phase 1: validation vector (sum of the validation rows, all tiles)
        for j0 in range(tidx * 4, NCOLS, NTHREADS * 4):
            v0 = cutlass.Float32(0.0)
            v1 = cutlass.Float32(0.0)
            v2 = cutlass.Float32(0.0)
            v3 = cutlass.Float32(0.0)
            for b in range(n_train, B_total):
                for t in range(tiles):
                    v0 = v0 + mP[b, t, j0]
                    v1 = v1 + mP[b, t, j0 + 1]
                    v2 = v2 + mP[b, t, j0 + 2]
                    v3 = v3 + mP[b, t, j0 + 3]
            s_val[j0] = v0
            s_val[j0 + 1] = v1
            s_val[j0 + 2] = v2
            s_val[j0 + 3] = v3
        cute.arch.sync_threads()

        # phase 2: one warp per train row (rows warp, warp+8, ...), lanes stride over the columns
        corr = mCorr[0]
        for b in range(warp, n_train, 8):
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            acc2 = cutlass.Float32(0.0)
            acc3 = cutlass.Float32(0.0)
            # four consecutive columns per lane per step: independent loads for memory-level parallelism
            for j0 in range(lane * 4, NCOLS, 128):
                c0 = cutlass.Float32(0.0)
                c1 = cutlass.Float32(0.0)
                c2 = cutlass.Float32(0.0)
                c3 = cutlass.Float32(0.0)
                for t in range(tiles):
                    c0 = c0 + mP[b, t, j0]
                    c1 = c1 + mP[b, t, j0 + 1]
                    c2 = c2 + mP[b, t, j0 + 2]
                    c3 = c3 + mP[b, t, j0 + 3]
                acc0 = acc0 + c0 * s_val[j0]
                acc1 = acc1 + c1 * s_val[j0 + 1]
                acc2 = acc2 + c2 * s_val[j0 + 2]
                acc3 = acc3 + c3 * s_val[j0 + 3]
            acc = cute.arch.warp_reduction_sum((acc0 + acc1) + (acc2 + acc3))
            if lane == 0:
                s_sc[b] = acc * corr
                mScores[b] = acc * corr
        cute.arch.sync_threads()

        # phase 3 (warp 0): k rounds of warp-parallel argmax (lowest index wins ties), then ascending sort
        if k > 0:
            if warp == 0:
                for b in range(lane, n_train, 32):
                    s_taken[b] = 0
                cute.arch.sync_warp()
                for i in range(k):
                    best_v = cutlass.Float32(-3.0e38)
                    best_i = cutlass.Int32(2147483647)
                    for b in range(lane, n_train, 32):
                        if s_taken[b] == 0:
                            v = s_sc[b]
                            if v > best_v:
                                best_v = v
                                best_i = b
                    best_v, best_i = self.argmax_step(best_v, best_i, 16)
                    best_v, best_i = self.argmax_step(best_v, best_i, 8)
                    best_v, best_i = self.argmax_step(best_v, best_i, 4)
                    best_v, best_i = self.argmax_step(best_v, best_i, 2)
                    best_v, best_i = self.argmax_step(best_v, best_i, 1)
                    if lane == 0:
                        s_taken[best_i] = 1
                        s_sel[i] = best_i
                    cute.arch.sync_warp()
                if lane == 0:
                    for i in range(1, k):
                        key = s_sel[i]
                        j = i - 1
                        while j >= 0 and s_sel[j] > key:
                            s_sel[j + 1] = s_sel[j]
                            j = j - 1
                        s_sel[j + 1] = key
                    for i in range(k):
                        mSel[i] = cutlass.Int64(s_sel[i])



class ReduceSelect:
    """Single-CTA epilogue for exact scoring: rows of an ``[n, R]`` fp32 partial matrix are summed
    (one warp per row), scaled by ``corr``, and the k largest are selected (ascending index order)."""

    @cute.jit
    def argmax_step(self, best_v, best_i, off: cutlass.Constexpr):
        ov = cute.arch.shuffle_sync_bfly(best_v, offset=off)
        oi = cute.arch.shuffle_sync_bfly(best_i, offset=off)
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
        return best_v, best_i

    @cute.jit
    def __call__(self, mP: cute.Tensor, mCorr: cute.Tensor, mScores: cute.Tensor, mSel: cute.Tensor,
                 k: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(mP, mCorr, mScores, mSel, k).launch(grid=(1, 1, 1), block=(NTHREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mP: cute.Tensor, mCorr: cute.Tensor, mScores: cute.Tensor, mSel: cute.Tensor, k: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        lane = tidx % 32
        n = cute.size(mP.shape[0])
        R = cute.size(mP.shape[1])
        smem = cutlass.memory.SmemAllocator()
        s_sc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(MAX_ROWS), 16)
        s_taken = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAX_ROWS), 16)
        s_sel = smem.allocate_tensor(cutlass.Int32, cute.make_layout(MAX_ROWS), 16)
        corr = mCorr[0]
        for b in range(warp, n, 8):
            acc = cutlass.Float32(0.0)
            for r in range(lane, R, 32):
                acc = acc + mP[b, r]
            acc = cute.arch.warp_reduction_sum(acc)
            if lane == 0:
                s_sc[b] = acc * corr
                mScores[b] = acc * corr
        cute.arch.sync_threads()
        if k > 0:
            if warp == 0:
                for b in range(lane, n, 32):
                    s_taken[b] = 0
                cute.arch.sync_warp()
                for i in range(k):
                    best_v = cutlass.Float32(-3.0e38)
                    best_i = cutlass.Int32(2147483647)
                    for b in range(lane, n, 32):
                        if s_taken[b] == 0:
                            v = s_sc[b]
                            if v > best_v:
                                best_v = v
                                best_i = b
                    best_v, best_i = self.argmax_step(best_v, best_i, 16)
                    best_v, best_i = self.argmax_step(best_v, best_i, 8)
                    best_v, best_i = self.argmax_step(best_v, best_i, 4)
                    best_v, best_i = self.argmax_step(best_v, best_i, 2)
                    best_v, best_i = self.argmax_step(best_v, best_i, 1)
                    if lane == 0:
                        s_taken[best_i] = 1
                        s_sel[i] = best_i
                    cute.arch.sync_warp()
                if lane == 0:
                    for i in range(1, k):
                        key = s_sel[i]
                        j = i - 1
                        while j >= 0 and s_sel[j] > key:
                            s_sel[j + 1] = s_sel[j]
                            j = j - 1
                        s_sel[j + 1] = key
                    for i in range(k):
                        mSel[i] = cutlass.Int64(s_sel[i])


def _get_reduce_select():
    fn = _COMPILED.get("reduce_select")
    if fn is None:
        fn = cute.compile(
            ReduceSelect(),
            make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=4),
            make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4),
            make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4),
            make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8),
            cutlass.Int32(0), make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _COMPILED["reduce_select"] = fn
    return fn


def reduce_select(partials: torch.Tensor, corr, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``scores = corr * partials.sum(1)`` and the k largest indices (ascending), one launch.

    ``partials`` fp32 ``[n, R]`` (e.g. from :func:`pip_partials` / :func:`gip_partials`, optionally with
    extra columns such as a bias-gradient term); ``k = 0`` -> scores only. Ties resolve to the lowest index.
    """
    n, R = partials.shape
    if not (0 < n <= _SS_MAX_ROWS and 0 <= k <= n):
        raise ValueError("reduce_select: need 0 < n <= 256 and 0 <= k <= n")
    P = partials if partials.is_contiguous() else partials.contiguous()
    scores = torch.empty(n, dtype=torch.float32, device=P.device)
    sel = torch.empty(max(k, 1), dtype=torch.int64, device=P.device)
    if isinstance(corr, torch.Tensor):
        corr_t = corr.reshape(1) if (corr.dtype == torch.float32 and corr.is_cuda) else corr.to(device=P.device, dtype=torch.float32).reshape(1)
    else:
        corr_t = torch.tensor([float(corr)], dtype=torch.float32, device=P.device)
    _get_reduce_select()(P, corr_t, scores, sel, int(k))
    return scores, sel[:k]


def _get_score_select():
    fn = _COMPILED.get("score_select")
    if fn is None:
        fn = cute.compile(
            ScoreSelect(),
            make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), cute.sym_int(), _SS_COLS), stride_order=(2, 1, 0), assumed_align=16),
            make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4),
            make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4),
            make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8),
            cutlass.Int32(1), cutlass.Int32(0), make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi")
        _COMPILED["score_select"] = fn
    return fn


def compressed_partials(go: torch.Tensor, inp: torch.Tensor, P_O: torch.Tensor, P_I: torch.Tensor,
                        scale: float = 1.0) -> torch.Tensor:
    """Per-tile partials of :func:`compressed_grad`: fp32 ``[B, S_tiles, 64, 64]`` (already scaled);
    ``partials.sum(1)`` flattened is the compressed gradient of each sample (zero-padded beyond k1, k2)."""
    B, S, O = go.shape
    I = inp.shape[2]
    if not (P_O.shape[0] == O and P_I.shape[0] == I and P_O.shape[1] <= 64 and P_I.shape[1] <= 64 and _ok((O, 8), (I, 8))):
        raise ValueError("cute compressed_partials: needs P_O [O,k1], P_I [I,k2] with k <= 64 and O % 8 == I % 8 == 0")
    if not go.is_contiguous():
        go = go.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()
    if P_O.dtype != go.dtype or not P_O.is_contiguous():
        P_O = P_O.to(go.dtype).contiguous()
    if P_I.dtype != go.dtype or not P_I.is_contiguous():
        P_I = P_I.to(go.dtype).contiguous()
    out = torch.empty(B, -(-S // _PROJ_BM), 64, 64, dtype=torch.float32, device=go.device)
    _get("proj", go.dtype)(go, inp, P_O, P_I, out, float(scale))
    return out


def score_select(partials: torch.Tensor, n_train: int, k: int, corr) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scores and top-k selection straight from the compressed-gradient partials, in one launch.

    ``partials`` ``[B_total, tiles, 64, 64]`` fp32 from :func:`compressed_partials` for the merged batch
    (training rows first, then validation rows).  Returns ``scores`` ``[n_train]`` fp32 =
    ``corr * <c_b, sum_{val} c_v>`` and ``sel`` ``[k]`` int64 = indices of the k largest scores in
    ascending order (``k = 0`` -> empty).  Ties resolve to the lowest index.
    """
    B_total, tiles = partials.shape[0], partials.shape[1]
    if not (B_total <= _SS_MAX_ROWS and 0 < n_train <= B_total and 0 <= k <= n_train):
        raise ValueError("score_select: need 0 < n_train <= B_total <= 256 and 0 <= k <= n_train")
    P = partials.view(B_total, tiles, _SS_COLS)
    scores = torch.empty(n_train, dtype=torch.float32, device=partials.device)
    sel = torch.empty(max(k, 1), dtype=torch.int64, device=partials.device)
    if isinstance(corr, torch.Tensor):
        corr_t = corr.reshape(1) if (corr.dtype == torch.float32 and corr.is_cuda) else corr.to(device=partials.device, dtype=torch.float32).reshape(1)
    else:
        corr_t = torch.tensor([float(corr)], dtype=torch.float32, device=partials.device)
    _get_score_select()(P, corr_t, scores, sel, int(n_train), int(k))
    return scores, sel[:k]


def compressed_grad(go: torch.Tensor, inp: torch.Tensor, P_O: torch.Tensor, P_I: torch.Tensor,
                    scale: float = 1.0) -> torch.Tensor:
    """Kronecker-projected per-sample gradients (the ``compress`` scoring path).

    go [B,S,O], inp [B,S,I], P_O [O,k1], P_I [I,k2] (k1, k2 <= 64, same dtype as go) ->
    ``scale * (go_b @ P_O)^T (inp_b @ P_I)`` flattened to ``[B, k1*k2]`` in go.dtype, i.e.
    ``einsum('bsi,bsj->bij', go @ P_O, inp @ P_I)`` with the projected components rounded to
    go.dtype exactly as the reference does.  Requires O % 8 == I % 8 == 0.
    """
    B, S, O = go.shape
    I = inp.shape[2]
    k1, k2 = P_O.shape[1], P_I.shape[1]
    if not (P_O.shape[0] == O and P_I.shape[0] == I and k1 <= 64 and k2 <= 64 and _ok((O, 8), (I, 8))):
        raise ValueError("cute compressed_grad: needs P_O [O,k1], P_I [I,k2] with k <= 64 and O % 8 == I % 8 == 0")
    if not go.is_contiguous():
        go = go.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()
    if P_O.dtype != go.dtype or not P_O.is_contiguous():
        P_O = P_O.to(go.dtype).contiguous()
    if P_I.dtype != go.dtype or not P_I.is_contiguous():
        P_I = P_I.to(go.dtype).contiguous()
    tiles = -(-S // _PROJ_BM)
    out = torch.empty(B, tiles, 64, 64, dtype=torch.float32, device=go.device)
    _get("proj", go.dtype)(go, inp, P_O, P_I, out, float(scale))
    if k1 != 64 or k2 != 64:
        out = out[:, :, :k1, :k2]
    # sum the fp32 tile partials first, then round once (torch.sum(dtype=bf16) would round each partial)
    res = out.sum(dim=1) if tiles > 1 else out[:, 0]
    return res.to(go.dtype).reshape(B, k1 * k2)


__all__ = ["AmpereFusedGemm", "ScoreSelect", "ReduceSelect", "supports", "pip_scores", "pip_partials",
           "gip_scores", "gip_partials", "selected_wgrad", "compressed_grad", "compressed_partials",
           "score_select", "reduce_select"]
