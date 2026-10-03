"""
Hopper (sm_90a) CuTe DSL kernels: TMA loads, wgmma, warp-specialised persistent mainloops.

``selected_wgrad`` -- the selected-sample weight gradient with a TMA-store epilogue:

    W[o, i] = scale * sum_{k in sel} sum_s go[sel[k], s, o] * inp[sel[k], s, i]

A is ``go`` viewed as ``(O, S, B)`` (M-major), B is ``inp`` viewed as ``(I, S, B)`` (N-major).  The K loop of a CTA
walks (selected sample, 64-token tile) pairs: the producer warp issues the TMA loads of both operands straight from
the 3-D activation tensors (the sample index is the third TMA coordinate), so the selected rows are never gathered.
fp32 accumulation; the scale is applied to the fp32 accumulators before the single rounding to the output dtype.
Tile/cluster/rasterisation configuration follows NVIDIA's ``examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm``
persistent kernel; on Qwen3-14B layer shapes the kernel runs within a few percent of cuBLAS on a contiguous copy of the
selected rows, i.e. the row gather of the reference path is the only thing it removes.

``gip_partials`` -- the ghost inner product as a dual GEMM with a Hadamard epilogue:

    partial[b*V + v, m*n_tiles + n] = sum_{s in tile m, t in tile n} (go_t[b,s,:] . go_v[v,t,:]) (inp_t[b,s,:] . inp_v[v,t,:])

One CTA tile streams go_t[b] x go_v[v]^T (K = O) and then inp_t[b] x inp_v[v]^T (K = I) through the same smem ring into
two fp32 accumulators, multiplies them elementwise and reduces to one fp32 per CTA: the ``[B, V, S, S]`` Gram matrices
never exist and nothing is rounded to bf16 before the reduction.

``compressed_partials`` / ``compressed_scores`` -- compressed scoring in two kernels: ``HopperDualProj`` forms both
projections ``go P_O`` and ``inp P_I`` over the flattened tokens in one persistent TMA + wgmma launch (64 x 64 x 64 tiles, no
K split, output rounded once to the activation dtype as the reference's bf16 matmul does) and ``HopperOuterScore`` forms
the per-sample 64 x 64 outer products on tensor cores (K = the tokens of a sample) -- either as fp32 compressed gradients
(``compressed_partials``, the ``[B, 1, 64, 64]`` layout ``score_select`` consumes) or, one CTA per training sample, straight
into ``corr * <c_b, sum_v c_v>`` with the k largest indices selected by the last CTA to finish (``compressed_scores``).

``pip_partials`` -- the per-token inner product ``sum_{s,o} go[b,s,o] (inp[b,s,:] . G[o,:])`` as a TMA + wgmma GEMM
(A = inp as ``(S, I, B)``, B = the target gradient ``G`` ``(O, I)``, both K-major) whose fp32 accumulator tile is dotted
with the matching ``go`` tile in registers; the ``[B, S, O]`` product of the reference is never written.

The kernels use the persistent tile scheduler shipped in the ``cutlass.utils`` wheel (``PersistentTileSchedulerParams`` /
``StaticPersistentTileScheduler``), which NVIDIA has announced will move to the examples tree in a later release.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.memory
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.tensor_utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


class HopperSelectedWgrad:
    def __init__(self, ab_dtype, tile_mn=(128, 256), cluster_mn=(1, 1), swizzle_size=8, raster_along_m=True,
                 max_active_clusters=132):
        self.ab_dtype = ab_dtype
        self.a_dtype = ab_dtype
        self.b_dtype = ab_dtype
        self.c_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.a_layout = LayoutEnum.COL_MAJOR   # (O, S, B) with O contiguous -> M-major
        self.b_layout = LayoutEnum.COL_MAJOR   # (I, S, B) with I contiguous -> N-major
        self.c_layout = LayoutEnum.ROW_MAJOR   # (O, I) with I contiguous -> N-major output
        self.cluster_shape_mn = cluster_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.max_active_clusters = max_active_clusters
        self.tile_shape_mnk = (*tile_mn, 64)
        self.atom_layout_mnk = (2, 1, 1) if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128 else (1, 1, 1)
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

    # ------------------------------------------------------------------ static configuration (inside the jit context)
    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_layout.sm90_mma_major_mode(), self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk, tiler_mn=(64, self.tile_shape_mnk[1]))
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative)
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = cute.size(a_shape) * self.a_dtype.width // 8 + cute.size(b_shape) * self.b_dtype.width // 8
        self.epi_stage = 4
        epi_bytes = cute.size(self.epi_tile) * self.c_dtype.width // 8 * self.epi_stage
        self.ab_stage = (self.smem_capacity - (1024 + epi_bytes)) // ab_bytes
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.a_layout, self.tile_shape_mnk, self.a_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.b_layout, self.tile_shape_mnk, self.b_dtype, self.ab_stage)
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.epi_stage)

    # ------------------------------------------------------------------ host entry
    @cute.jit
    def __call__(self, t_go: cute.Tensor, t_inp: cute.Tensor, t_sel: cute.Tensor, t_scale: cute.Tensor,
                 t_out: cute.Tensor, stream: cuda.CUstream):
        """t_go (B,S,O), t_inp (B,S,I) same half dtype, t_sel (K,) int64, t_scale (1,) f32, t_out (O,I) half."""
        mA = cute.make_tensor(t_go.iterator, cute.select(t_go.layout, mode=[2, 1, 0]))    # (O, S, B) M-major
        mB = cute.make_tensor(t_inp.iterator, cute.select(t_inp.layout, mode=[2, 1, 0]))  # (I, S, B) N-major
        mC = t_out                                                                        # (O, I)
        self._setup_attributes()
        bM, bN, bK = self.tile_shape_mnk

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        op_a = cpasync.CopyBulkTensorTileG2SOp() if self.num_mcast_ctas_a == 1 else cpasync.CopyBulkTensorTileG2SMulticastOp()
        op_b = cpasync.CopyBulkTensorTileG2SOp() if self.num_mcast_ctas_b == 1 else cpasync.CopyBulkTensorTileG2SMulticastOp()
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(op_a, mA, a_smem_layout, (bM, bK), num_multicast=self.num_mcast_ctas_a)
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(op_b, mB, b_smem_layout, (bN, bK), num_multicast=self.num_mcast_ctas_b)
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mC, epi_smem_layout, self.epi_tile)

        m_tiles = cute.ceil_div(cute.size(mC.shape[0]), bM)
        n_tiles = cute.ceil_div(cute.size(mC.shape[1]), bN)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (m_tiles, n_tiles, 1), (*self.cluster_shape_mn, 1), self.swizzle_size, self.raster_along_m)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, self.max_active_clusters)
        s_tiles = cute.ceil_div(cute.size(mA.shape[1]), bK)
        n_sel = cute.size(t_sel.shape[0])

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c, t_sel, t_scale,
                    s_tiles, n_sel, self.tiled_mma, self.cta_layout_mnk, self.a_smem_layout_staged,
                    self.b_smem_layout_staged, self.epi_smem_layout_staged, tile_sched_params).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1, stream=stream)

    # ------------------------------------------------------------------ device kernel
    @cute.kernel
    def kernel(self, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
               tma_atom_c: cute.CopyAtom, mC_mn: cute.Tensor, mSel: cute.Tensor, mScale: cute.Tensor,
               s_tiles: cutlass.Int32, n_sel: cutlass.Int32, tiled_mma: cute.TiledMma, cta_layout_mnk: cute.Layout,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               epi_smem_layout_staged: cute.ComposedLayout, tile_sched_params: utils.PersistentTileSchedulerParams):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr, num_stages=self.ab_stage, producer_group=producer_group,
            consumer_group=consumer_group, tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # (bM, bK, RestM, RestK, RestL) / (bN, bK, RestN, RestK, RestL) / (bM, bN, RestM, RestN)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        gC_mn = cute.local_tile(mC_mn, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None))

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, cluster_coord_mnk[1], a_cta_layout,
                                           cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2))
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, cluster_coord_mnk[0], b_cta_layout,
                                           cute.group_modes(sB, 0, 2), cute.group_modes(gB_nkl, 0, 2))

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_wg_thread_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(mma_wg_thread_layout(warp_group_idx - self.num_dma_warp_groups))
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mn)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        k_tile_cnt = n_sel * s_tiles
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tile_coord = work_tile.tile_idx
                producer_state.reset_count()
                for kk in range(n_sel):
                    b_idx = cutlass.Int32(mSel[kk])
                    tAgA_b = tAgA[(None, tile_coord[0], None, b_idx)]
                    tBgB_b = tBgB[(None, tile_coord[1], None, b_idx)]
                    for kt in range(s_tiles):
                        mainloop_pipeline.producer_acquire(producer_state)
                        cute.copy(tma_atom_a, tAgA_b[(None, kt)], tAsA[(None, producer_state.index)],
                                  tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=a_mcast_mask)
                        cute.copy(tma_atom_b, tBgB_b[(None, kt)], tBsB[(None, producer_state.index)],
                                  tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=b_mcast_mask)
                        mainloop_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(producer_state)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])
            scale = mScale[0]

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype)
            copy_atom_C = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4), self.c_dtype)
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)
            tma_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.epi_stage, producer_group=tma_store_producer_group)

            while work_tile.is_valid_tile:
                tile_coord = work_tile.tile_idx
                gC_slice = gC_mn[(None, None, tile_coord[0], tile_coord[1])]
                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                    warpgroup.commit_group()
                    read_state.advance()
                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                    warpgroup.commit_group()
                    warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                    read_state.advance()
                warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()

                # epilogue: acc * scale -> bf16 -> smem (stmatrix) -> TMA store
                tCgC_for_tma = cute.zipped_divide(gC_slice, self.epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), tCgC_for_tma)
                epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
                epi_tile_shape = tCgC_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store((acc_vec * scale).to(self.c_dtype))
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(tma_atom_c, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tma_store_pipeline.producer_tail()


class HopperGip:
    def __init__(self, ab_dtype, tile_mn=(128, 128), max_active_clusters=132):
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.ab_layout = LayoutEnum.ROW_MAJOR          # (S, F, B) with F contiguous -> K-major
        self.tile_shape_mnk = (*tile_mn, 64)
        self.atom_layout_mnk = (2, 1, 1)               # two consumer warpgroups, 64 rows each
        self.cluster_shape_mn = (1, 1)
        self.max_active_clusters = max_active_clusters
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = 2
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 3 * 128
        self.load_warp_id = 0
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_threads = 256
        self.num_mma_warps = 8
        self.red_barrier_a = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)
        self.red_barrier_b = pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_mma_threads)

    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, self.ab_layout.sm90_mma_major_mode(), self.ab_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk, tiler_mn=(64, self.tile_shape_mnk[1]))
        self.cta_layout_mnk = cute.make_layout((1, 1, 1))
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = (cute.size(a_shape) + cute.size(b_shape)) * self.ab_dtype.width // 8
        self.ab_stage = (self.smem_capacity - 2048) // ab_bytes
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.ab_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.ab_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)

    @cute.jit
    def __call__(self, t_got: cute.Tensor, t_gov: cute.Tensor, t_inpt: cute.Tensor, t_inpv: cute.Tensor,
                 t_out: cute.Tensor, stream: cuda.CUstream):
        """t_got (B,S,O), t_gov (V,Sv,O), t_inpt (B,S,I), t_inpv (V,Sv,I) same half dtype; t_out (B*V, m_tiles*n_tiles) f32."""
        mA1 = cute.make_tensor(t_got.iterator, cute.select(t_got.layout, mode=[1, 2, 0]))   # (S, O, B)
        mB1 = cute.make_tensor(t_gov.iterator, cute.select(t_gov.layout, mode=[1, 2, 0]))   # (Sv, O, V)
        mA2 = cute.make_tensor(t_inpt.iterator, cute.select(t_inpt.layout, mode=[1, 2, 0])) # (S, I, B)
        mB2 = cute.make_tensor(t_inpv.iterator, cute.select(t_inpv.layout, mode=[1, 2, 0])) # (Sv, I, V)
        self._setup_attributes()
        bM, bN, bK = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        atom_a1, tma_a1 = cpasync.make_tiled_tma_atom(op, mA1, a_smem_layout, (bM, bK))
        atom_b1, tma_b1 = cpasync.make_tiled_tma_atom(op, mB1, b_smem_layout, (bN, bK))
        atom_a2, tma_a2 = cpasync.make_tiled_tma_atom(op, mA2, a_smem_layout, (bM, bK))
        atom_b2, tma_b2 = cpasync.make_tiled_tma_atom(op, mB2, b_smem_layout, (bN, bK))
        m_tiles = cute.ceil_div(cute.size(mA1.shape[0]), bM)
        n_tiles = cute.ceil_div(cute.size(mB1.shape[0]), bN)
        V = cute.size(mB1.shape[2])
        L = cute.size(mA1.shape[2]) * V
        k1 = cute.ceil_div(cute.size(mA1.shape[1]), bK)
        k2 = cute.ceil_div(cute.size(mA2.shape[1]), bK)
        tile_sched_params = utils.PersistentTileSchedulerParams((m_tiles, n_tiles, L), (1, 1, 1), 1, True)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, self.max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            red: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.num_mma_warps], 16]
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(atom_a1, tma_a1, atom_b1, tma_b1, atom_a2, tma_a2, atom_b2, tma_b2, t_out, k1, k2, V, n_tiles,
                    self.tiled_mma, self.a_smem_layout_staged, self.b_smem_layout_staged, tile_sched_params).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.jit
    def consume(self, mainloop_pipeline, acc, k_cnt, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma):
        """acc = sum over k_cnt pipeline stages of A @ B (one wgmma group in flight); returns the advanced states."""
        acc.fill(0.0)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.fence()
        prologue = min(1, k_cnt)
        for k_tile in range(prologue):
            mainloop_pipeline.consumer_wait(read_state)
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
            warpgroup.commit_group()
            read_state.advance()
        for k_tile in range(prologue, k_cnt):
            mainloop_pipeline.consumer_wait(read_state)
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            mainloop_pipeline.consumer_release(release_state)
            release_state.advance()
            read_state.advance()
        warpgroup.wait_group(0)
        for k_tile in range(prologue):
            mainloop_pipeline.consumer_release(release_state)
            release_state.advance()
        return read_state, release_state

    @cute.kernel
    def kernel(self, atom_a1: cute.CopyAtom, mA1: cute.Tensor, atom_b1: cute.CopyAtom, mB1: cute.Tensor,
               atom_a2: cute.CopyAtom, mA2: cute.Tensor, atom_b2: cute.CopyAtom, mB2: cute.Tensor, mOut: cute.Tensor,
               k1: cutlass.Int32, k2: cutlass.Int32, V: cutlass.Int32, n_tiles: cutlass.Int32, tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: utils.PersistentTileSchedulerParams):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(atom_a1)
            cpasync.prefetch_descriptor(atom_b1)
            cpasync.prefetch_descriptor(atom_a2)
            cpasync.prefetch_descriptor(atom_b2)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.ab_dtype, a_smem_layout) + cute.size_in_bytes(self.ab_dtype, b_smem_layout)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(), num_stages=self.ab_stage,
            producer_group=producer_group, consumer_group=consumer_group, tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sRed = storage.red.get_tensor(cute.make_layout(self.num_mma_warps))

        tile_a = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        tile_b = cute.slice_(self.tile_shape_mnk, (0, None, None))
        gA1 = cute.local_tile(mA1, tile_a, (None, None, None))    # (bM, bK, RestM, RestK, B)
        gB1 = cute.local_tile(mB1, tile_b, (None, None, None))    # (bN, bK, RestN, RestK, V)
        gA2 = cute.local_tile(mA2, tile_a, (None, None, None))
        gB2 = cute.local_tile(mB2, tile_b, (None, None, None))
        one = cute.make_layout(1)
        tAsA, tAgA1 = cpasync.tma_partition(atom_a1, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA1, 0, 2))
        _, tAgA2 = cpasync.tma_partition(atom_a2, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA2, 0, 2))
        tBsB, tBgB1 = cpasync.tma_partition(atom_b1, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB1, 0, 2))
        _, tBgB2 = cpasync.tma_partition(atom_b2, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB2, 0, 2))

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_wg_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(mma_wg_layout(warp_group_idx - self.num_dma_warp_groups))
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_shape = thr_mma.partition_shape_C((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
        acc1 = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        acc2 = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                b_idx = tc[2] // V
                v_idx = tc[2] % V
                pstate.reset_count()
                tA1 = tAgA1[(None, tc[0], None, b_idx)]
                tB1 = tBgB1[(None, tc[1], None, v_idx)]
                for kt in range(k1):
                    mainloop_pipeline.producer_acquire(pstate)
                    cute.copy(atom_a1, tA1[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    cute.copy(atom_b1, tB1[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    mainloop_pipeline.producer_commit(pstate)
                    pstate.advance()
                tA2 = tAgA2[(None, tc[0], None, b_idx)]
                tB2 = tBgB2[(None, tc[1], None, v_idx)]
                for kt in range(k2):
                    mainloop_pipeline.producer_acquire(pstate)
                    cute.copy(atom_a2, tA2[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    cute.copy(atom_b2, tB2[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    mainloop_pipeline.producer_commit(pstate)
                    pstate.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(pstate)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])
            mma_tidx = tidx - self.num_threads_per_warp_group
            mma_warp = mma_tidx // 32
            lane = mma_tidx % 32
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                read_state.reset_count()
                release_state.reset_count()
                read_state, release_state = self.consume(mainloop_pipeline, acc1, k1, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma)
                read_state, release_state = self.consume(mainloop_pipeline, acc2, k2, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma)
                part = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(cute.size(acc1)):
                    part = part + acc1[i] * acc2[i]
                part = cute.arch.warp_reduction_sum(part)
                if lane == 0:
                    sRed[mma_warp] = part
                self.red_barrier_a.arrive_and_wait()
                if mma_tidx == 0:
                    total = cutlass.Float32(0.0)
                    for w in cutlass.range_constexpr(self.num_mma_warps):
                        total = total + sRed[w]
                    mOut[tc[2], tc[0] * n_tiles + tc[1]] = total
                self.red_barrier_b.arrive_and_wait()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


PROJ_N = 64      # width of the compressed-scoring projections (k1, k2 <= 64, zero-padded)
MAX_ROWS = 256   # merged-batch rows the in-kernel top-k handles


class _WgmmaMainloop:
    @cute.jit
    def consume(self, mainloop_pipeline, acc, k_cnt, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma):
        """acc += A @ B over k_cnt pipeline stages (one wgmma group in flight); returns the advanced states.
        The caller zeroes ``acc``, sets ACCUMULATE and fences: the tiled MMA is passed by value, so a flag set in here
        would not be seen by the caller."""
        prologue = min(1, k_cnt)
        for k_tile in range(prologue):
            mainloop_pipeline.consumer_wait(read_state)
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
            warpgroup.commit_group()
            read_state.advance()
        for k_tile in range(prologue, k_cnt):
            mainloop_pipeline.consumer_wait(read_state)
            for kb in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, kb, read_state.index)
                cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            mainloop_pipeline.consumer_release(release_state)
            release_state.advance()
            read_state.advance()
        warpgroup.wait_group(0)
        for k_tile in range(prologue):
            mainloop_pipeline.consumer_release(release_state)
            release_state.advance()
        return read_state, release_state


class HopperDualProj(_WgmmaMainloop):
    """Both compressed-scoring projections in one persistent launch: ``out[0] = go P_O`` and ``out[1] = inp P_I`` over the
    flattened tokens, rounded once from fp32 to the activation dtype (what the reference's bf16 matmul does).  Tile
    64 x 64 x 64 with one TMA producer warp and one MMA warpgroup; the 64-row tiles of the two problems fill the GPU
    without a K split, so no fp32 partials are written."""

    def __init__(self, ab_dtype, max_active_clusters=132):
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.a_layout = LayoutEnum.ROW_MAJOR      # X (rows, F): K-major
        self.b_layout = LayoutEnum.COL_MAJOR      # P viewed as (k, F) with stride (1, k): N-major
        self.tile_shape_mnk = (64, PROJ_N, 64)
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.max_active_clusters = max_active_clusters
        self.num_dma_warp_groups = 1
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 256
        self.load_warp_id = 0
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, self.a_layout.sm90_mma_major_mode(), self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk, tiler_mn=(64, self.tile_shape_mnk[1]))
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = (cute.size(a_shape) + cute.size(b_shape)) * self.ab_dtype.width // 8
        self.ab_stage = min(12, (self.smem_capacity - 2048) // ab_bytes)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.a_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.b_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)

    @cute.jit
    def __call__(self, t_x0: cute.Tensor, t_p0: cute.Tensor, t_x1: cute.Tensor, t_p1: cute.Tensor, t_out: cute.Tensor,
                 stream: cuda.CUstream):
        """t_x0 (rows, F0), t_p0 (F0, k0), t_x1 (rows, F1), t_p1 (F1, k1) half; t_out (2, rows, 64) half."""
        mB0 = cute.make_tensor(t_p0.iterator, cute.select(t_p0.layout, mode=[1, 0]))
        mB1 = cute.make_tensor(t_p1.iterator, cute.select(t_p1.layout, mode=[1, 0]))
        self._setup_attributes()
        bM, bN, bK = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        atom_a0, tma_a0 = cpasync.make_tiled_tma_atom(op, t_x0, a_smem_layout, (bM, bK))
        atom_b0, tma_b0 = cpasync.make_tiled_tma_atom(op, mB0, b_smem_layout, (bN, bK))
        atom_a1, tma_a1 = cpasync.make_tiled_tma_atom(op, t_x1, a_smem_layout, (bM, bK))
        atom_b1, tma_b1 = cpasync.make_tiled_tma_atom(op, mB1, b_smem_layout, (bN, bK))
        rows = cute.size(t_x0.shape[0])
        m_tiles = cute.ceil_div(rows, bM)
        k_tiles0 = cute.ceil_div(cute.size(t_x0.shape[1]), bK)
        k_tiles1 = cute.ceil_div(cute.size(t_x1.shape[1]), bK)
        tile_sched_params = utils.PersistentTileSchedulerParams((m_tiles, 1, 2), (1, 1, 1), 1, True)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, self.max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(atom_a0, tma_a0, atom_b0, tma_b0, atom_a1, tma_a1, atom_b1, tma_b1, t_out, k_tiles0, k_tiles1,
                    self.tiled_mma, self.a_smem_layout_staged, self.b_smem_layout_staged, tile_sched_params).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, atom_a0: cute.CopyAtom, mA0: cute.Tensor, atom_b0: cute.CopyAtom, mB0: cute.Tensor,
               atom_a1: cute.CopyAtom, mA1: cute.Tensor, atom_b1: cute.CopyAtom, mB1: cute.Tensor, mOut: cute.Tensor,
               k_tiles0: cutlass.Int32, k_tiles1: cutlass.Int32, tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: utils.PersistentTileSchedulerParams):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(atom_a0)
            cpasync.prefetch_descriptor(atom_b0)
            cpasync.prefetch_descriptor(atom_a1)
            cpasync.prefetch_descriptor(atom_b1)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.ab_dtype, a_smem_layout) + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(), num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4), tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        bM, bN, bK = self.tile_shape_mnk
        gA0 = cute.local_tile(mA0, (bM, bK), (None, None))
        gB0 = cute.local_tile(mB0, (bN, bK), (None, None))
        gA1 = cute.local_tile(mA1, (bM, bK), (None, None))
        gB1 = cute.local_tile(mB1, (bN, bK), (None, None))
        one = cute.make_layout(1)
        tAsA, tAgA0 = cpasync.tma_partition(atom_a0, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA0, 0, 2))
        tBsB, tBgB0 = cpasync.tma_partition(atom_b0, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB0, 0, 2))
        _, tAgA1 = cpasync.tma_partition(atom_a1, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA1, 0, 2))
        _, tBgB1 = cpasync.tma_partition(atom_b1, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB1, 0, 2))
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(0)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.acc_dtype)
        acc_out = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.ab_dtype)
        rows = cute.size(mA0.shape[0])
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                p = tc[2]
                k_cnt = k_tiles0 + (k_tiles1 - k_tiles0) * p
                tA0 = tAgA0[(None, tc[0], None)]
                tB0 = tBgB0[(None, 0, None)]
                tA1 = tAgA1[(None, tc[0], None)]
                tB1 = tBgB1[(None, 0, None)]
                for kt in range(k_cnt):
                    mainloop_pipeline.producer_acquire(pstate)
                    if p == 0:
                        cute.copy(atom_a0, tA0[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                        cute.copy(atom_b0, tB0[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    else:
                        cute.copy(atom_a1, tA1[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                        cute.copy(atom_b1, tB1[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    mainloop_pipeline.producer_commit(pstate)
                    pstate.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(pstate)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])
            cRows = cute.make_identity_tensor((rows, PROJ_N))
            thr_mma_c = tiled_mma.get_slice(tidx - self.num_threads_per_warp_group)
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                p = tc[2]
                k_cnt = k_tiles0 + (k_tiles1 - k_tiles0) * p
                acc.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                read_state, release_state = self.consume(mainloop_pipeline, acc, k_cnt, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma)
                acc_out.store(acc.load().to(self.ab_dtype))
                gS = cute.local_tile(mOut[p, None, None], (bM, bN), (tc[0], 0))
                tCgS = thr_mma_c.partition_C(gS)
                if (tc[0] + 1) * bM <= rows:
                    cute.autovec_copy(acc_out, tCgS)
                else:
                    tCcC = thr_mma_c.partition_C(cute.local_tile(cRows, (bM, bN), (tc[0], 0)))
                    for i in cutlass.range_constexpr(cute.size(acc_out)):
                        if cute.elem_less(tCcC[i][0], rows):
                            tCgS[i] = acc_out[i]
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


class HopperOuterScore(_WgmmaMainloop):
    """Per-sample outer products of the two projections on tensor cores: ``c_b = sum_s a[b, s, :] (x) c[b, s, :]`` with
    ``a = go P_O`` and ``c = inp P_I`` viewed as ``(64, S, B)`` (M-major A, N-major B; K = the tokens of one sample).

    ``scores_mode=False``: one CTA per sample writes ``scale * c_b`` as an fp32 64 x 64 tile.
    ``scores_mode=True``:  one CTA per training sample accumulates ``c_V`` (sum over the validation rows, which follow the
    training rows in the merged batch) and its own ``c_b``, and writes ``score_b = corr * scale^2 <c_b, c_V>``; when
    ``k > 0`` the last CTA to finish (arrival counter) reads the scores back and writes the k largest indices in ascending
    order (ties -> lowest index) -- projection, outer product, scoring and selection take two launches per layer."""

    def __init__(self, ab_dtype, scores_mode: bool):
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.scores_mode = scores_mode
        self.a_layout = LayoutEnum.COL_MAJOR
        self.b_layout = LayoutEnum.COL_MAJOR
        self.tile_shape_mnk = (PROJ_N, PROJ_N, 64)
        self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.num_dma_warp_groups = 1
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 256
        self.load_warp_id = 0
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.red_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)

    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, self.a_layout.sm90_mma_major_mode(), self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk, tiler_mn=(64, self.tile_shape_mnk[1]))
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = (cute.size(a_shape) + cute.size(b_shape)) * self.ab_dtype.width // 8
        self.ab_stage = min(8, (self.smem_capacity - 8192) // ab_bytes)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.a_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.b_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)

    @cute.jit
    def argmax_step(self, best_v, best_i, off: cutlass.Constexpr):
        """One butterfly step of a warp argmax over (value, index); ties -> lowest index."""
        ov = cute.arch.shuffle_sync_bfly(best_v, offset=off)
        oi = cute.arch.shuffle_sync_bfly(best_i, offset=off)
        if ov > best_v or (ov == best_v and oi < best_i):
            best_v = ov
            best_i = oi
        return best_v, best_i

    @cute.jit
    def __call__(self, t_a: cute.Tensor, t_c: cute.Tensor, t_out: cute.Tensor, t_scores: cute.Tensor, t_sel: cute.Tensor,
                 t_counter: cute.Tensor, t_corr: cute.Tensor, n_train: cutlass.Int32, k: cutlass.Int32,
                 scale: cutlass.Float32, stream: cuda.CUstream):
        """t_a, t_c (B, S, 64) half projections.  partials mode: t_out (B, 64, 64) f32.  scores mode: t_scores (n_train,) f32,
        t_sel (>= max(k, 1),) i64, t_counter (1,) i32 (zero on entry, reset by the kernel), t_corr (1,) f32.  Unused
        outputs may be 1-element placeholders."""
        mA = cute.make_tensor(t_a.iterator, cute.select(t_a.layout, mode=[2, 1, 0]))   # (64, S, B)
        mB = cute.make_tensor(t_c.iterator, cute.select(t_c.layout, mode=[2, 1, 0]))
        self._setup_attributes()
        bM, bN, bK = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        atom_a, tma_a = cpasync.make_tiled_tma_atom(op, mA, a_smem_layout, (bM, bK))
        atom_b, tma_b = cpasync.make_tiled_tma_atom(op, mB, b_smem_layout, (bN, bK))
        s_tiles = cute.ceil_div(cute.size(mA.shape[1]), bK)
        B_total = cute.size(mA.shape[2])
        n_ctas = n_train if cutlass.const_expr(self.scores_mode) else B_total

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            red: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4], 16]
            s_sc: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, MAX_ROWS], 16]
            s_taken: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_ROWS], 16]
            s_sel: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, MAX_ROWS], 16]
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(atom_a, tma_a, atom_b, tma_b, t_out, t_scores, t_sel, t_counter, t_corr, n_train, k, scale, s_tiles,
                    self.tiled_mma, self.a_smem_layout_staged, self.b_smem_layout_staged).launch(
            grid=[n_ctas, 1, 1], block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, atom_a: cute.CopyAtom, mA: cute.Tensor, atom_b: cute.CopyAtom, mB: cute.Tensor, mOut: cute.Tensor,
               mScores: cute.Tensor, mSel: cute.Tensor, mCounter: cute.Tensor, mCorr: cute.Tensor, n_train: cutlass.Int32,
               k: cutlass.Int32, scale: cutlass.Float32, s_tiles: cutlass.Int32, tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(atom_a)
            cpasync.prefetch_descriptor(atom_b)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.ab_dtype, a_smem_layout) + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(), num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4), tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sRed = storage.red.get_tensor(cute.make_layout(4))
        s_sc = storage.s_sc.get_tensor(cute.make_layout(MAX_ROWS))
        s_taken = storage.s_taken.get_tensor(cute.make_layout(MAX_ROWS))
        s_sel = storage.s_sel.get_tensor(cute.make_layout(MAX_ROWS))
        bM, bN, bK = self.tile_shape_mnk
        gA = cute.local_tile(mA, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))   # (bM, bK, 1, s_tiles, B)
        gB = cute.local_tile(mB, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        one = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(atom_a, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(atom_b, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(0)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc_b = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.acc_dtype)
        acc_v = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.acc_dtype)
        B_total = cute.size(mA.shape[2])
        n_val = B_total - n_train
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            if cutlass.const_expr(self.scores_mode):
                for r in range(n_train, B_total):
                    tAv = tAgA[(None, 0, None, r)]
                    tBv = tBgB[(None, 0, None, r)]
                    for kt in range(s_tiles):
                        mainloop_pipeline.producer_acquire(pstate)
                        cute.copy(atom_a, tAv[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                        cute.copy(atom_b, tBv[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                        mainloop_pipeline.producer_commit(pstate)
                        pstate.advance()
            tA = tAgA[(None, 0, None, bidx)]
            tB = tBgB[(None, 0, None, bidx)]
            for kt in range(s_tiles):
                mainloop_pipeline.producer_acquire(pstate)
                cute.copy(atom_a, tA[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                cute.copy(atom_b, tB[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                mainloop_pipeline.producer_commit(pstate)
                pstate.advance()
            mainloop_pipeline.producer_tail(pstate)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])
            mma_tidx = tidx - self.num_threads_per_warp_group
            mma_warp = mma_tidx // 32
            lane = mma_tidx % 32
            thr_mma_c = tiled_mma.get_slice(mma_tidx)
            if cutlass.const_expr(self.scores_mode):
                acc_v.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                read_state, release_state = self.consume(mainloop_pipeline, acc_v, n_val * s_tiles, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma)
            acc_b.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()
            read_state, release_state = self.consume(mainloop_pipeline, acc_b, s_tiles, read_state, release_state, tCrA, tCrB, num_k_blocks, tiled_mma)
            if cutlass.const_expr(self.scores_mode):
                part = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(cute.size(acc_b)):
                    part = part + acc_b[i] * acc_v[i]
                part = cute.arch.warp_reduction_sum(part)
                if lane == 0:
                    sRed[mma_warp] = part
                self.red_barrier.arrive_and_wait()
                if mma_warp == 0:
                    total = (sRed[0] + sRed[1]) + (sRed[2] + sRed[3])
                    score = total * scale * scale * mCorr[0]
                    if lane == 0:
                        mScores[bidx] = score
                    if k > 0:
                        # arrival counter: the last CTA sees every score (release/acquire at gpu scope, L2-coherent reads)
                        is_last = cutlass.Int32(0)
                        if lane == 0:
                            cute.arch.fence_acq_rel_gpu()
                            old = cute.arch.atomic_add(mCounter.iterator, cutlass.Int32(1), sem="acq_rel", scope="gpu")
                            if old == n_train - 1:
                                is_last = cutlass.Int32(1)
                        is_last = cute.arch.shuffle_sync(is_last, 0)
                        if is_last == 1:
                            cute.arch.fence_acq_rel_gpu()
                            for b in range(lane, n_train, 32):
                                s_sc[b] = cute.arch.atomic_add(mScores.iterator + b, cutlass.Float32(0.0), sem="relaxed", scope="gpu")
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
                                mCounter[0] = cutlass.Int32(0)
            else:
                acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.acc_dtype)
                acc_s.store(acc_b.load() * scale)
                tCgO = thr_mma_c.partition_C(mOut[bidx, None, None])
                cute.autovec_copy(acc_s, tCgO)


class HopperPip:
    def __init__(self, ab_dtype, tile_mn=(128, 256), swizzle_size=4, max_active_clusters=132):
        self.ab_dtype = ab_dtype
        self.acc_dtype = cutlass.Float32
        self.ab_layout = LayoutEnum.ROW_MAJOR          # K-major operands
        self.tile_shape_mnk = (*tile_mn, 64)
        self.atom_layout_mnk = (2, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.swizzle_size = swizzle_size
        self.max_active_clusters = max_active_clusters
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = 2
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 384
        self.load_warp_id = 0
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_warps = 8
        self.red_barrier_a = pipeline.NamedBarrier(barrier_id=1, num_threads=256)
        self.red_barrier_b = pipeline.NamedBarrier(barrier_id=2, num_threads=256)

    def _setup_attributes(self):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, self.ab_layout.sm90_mma_major_mode(), self.ab_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk, tiler_mn=(64, self.tile_shape_mnk[1]))
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = (cute.size(a_shape) + cute.size(b_shape)) * self.ab_dtype.width // 8
        self.ab_stage = (self.smem_capacity - 2048) // ab_bytes
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(self.ab_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(self.ab_layout, self.tile_shape_mnk, self.ab_dtype, self.ab_stage)

    @cute.jit
    def __call__(self, t_inp: cute.Tensor, t_G: cute.Tensor, t_go: cute.Tensor, t_out: cute.Tensor, stream: cuda.CUstream):
        """t_inp (B,S,I), t_G (O,I), t_go (B,S,O) same half dtype; t_out (B, m_tiles*n_tiles) f32."""
        mA = cute.make_tensor(t_inp.iterator, cute.select(t_inp.layout, mode=[1, 2, 0]))   # (S, I, B)
        mB = t_G                                                                            # (O, I)
        mE = cute.make_tensor(t_go.iterator, cute.select(t_go.layout, mode=[1, 2, 0]))     # (S, O, B)
        self._setup_attributes()
        bM, bN, bK = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        atom_a, tma_a = cpasync.make_tiled_tma_atom(op, mA, a_smem_layout, (bM, bK))
        atom_b, tma_b = cpasync.make_tiled_tma_atom(op, mB, b_smem_layout, (bN, bK))
        m_tiles = cute.ceil_div(cute.size(mA.shape[0]), bM)
        n_tiles = cute.ceil_div(cute.size(mB.shape[0]), bN)
        L = cute.size(mA.shape[2])
        k_tiles = cute.ceil_div(cute.size(mA.shape[1]), bK)
        tile_sched_params = utils.PersistentTileSchedulerParams((m_tiles, n_tiles, L), (1, 1, 1), self.swizzle_size, True)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, self.max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            red: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.num_mma_warps], 16]
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(atom_a, tma_a, atom_b, tma_b, mE, t_out, k_tiles, m_tiles, n_tiles, self.tiled_mma,
                    self.a_smem_layout_staged, self.b_smem_layout_staged, tile_sched_params).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, atom_a: cute.CopyAtom, mA: cute.Tensor, atom_b: cute.CopyAtom, mB: cute.Tensor, mE: cute.Tensor,
               mOut: cute.Tensor, k_tiles: cutlass.Int32, m_tiles: cutlass.Int32, n_tiles: cutlass.Int32, tiled_mma: cute.TiledMma,
               a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout,
               tile_sched_params: utils.PersistentTileSchedulerParams):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(atom_a)
            cpasync.prefetch_descriptor(atom_b)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.ab_dtype, a_smem_layout) + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(), num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps), tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sRed = storage.red.get_tensor(cute.make_layout(self.num_mma_warps))
        bM, bN, bK = self.tile_shape_mnk
        gA = cute.local_tile(mA, (bM, bK), (None, None, None))     # (bM, bK, RestM, RestK, B)
        gB = cute.local_tile(mB, (bN, bK), (None, None))           # (bN, bK, RestN, RestK)
        one = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(atom_a, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(atom_b, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_wg_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(mma_wg_layout(warp_group_idx - self.num_dma_warp_groups))
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((bM, bN)), self.acc_dtype)
        S = cute.size(mA.shape[0])
        O = cute.size(mB.shape[0])
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                tA = tAgA[(None, tc[0], None, tc[2])]
                tB = tBgB[(None, tc[1], None)]
                for kt in range(k_tiles):
                    mainloop_pipeline.producer_acquire(pstate)
                    cute.copy(atom_a, tA[(None, kt)], tAsA[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    cute.copy(atom_b, tB[(None, kt)], tBsB[(None, pstate.index)], tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
                    mainloop_pipeline.producer_commit(pstate)
                    pstate.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(pstate)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])
            mma_tidx = tidx - self.num_threads_per_warp_group
            mma_warp = mma_tidx // 32
            lane = mma_tidx % 32
            thr_mma_c = tiled_mma.get_slice(mma_tidx)
            cE = cute.make_identity_tensor((S, O))
            while work_tile.is_valid_tile:
                tc = work_tile.tile_idx
                acc.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.fence()
                prologue = min(1, k_tiles)
                for k_tile in range(prologue):
                    mainloop_pipeline.consumer_wait(read_state)
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        coord = (None, None, kb, read_state.index)
                        cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
                    warpgroup.commit_group()
                    read_state.advance()
                for k_tile in range(prologue, k_tiles):
                    mainloop_pipeline.consumer_wait(read_state)
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        coord = (None, None, kb, read_state.index)
                        cute.gemm(tiled_mma, acc, tCrA[coord], tCrB[coord], acc)
                    warpgroup.commit_group()
                    warpgroup.wait_group(1)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                    read_state.advance()
                warpgroup.wait_group(0)
                for k_tile in range(prologue):
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                # epilogue: dot the accumulator tile with the go tile of this (sample, S tile, O tile)
                gE = cute.local_tile(mE[None, None, tc[2]], (bM, bN), (tc[0], tc[1]))
                tCgE = thr_mma_c.partition_C(gE)
                tCcE = thr_mma_c.partition_C(cute.local_tile(cE, (bM, bN), (tc[0], tc[1])))
                part = cutlass.Float32(0.0)
                if (tc[0] + 1) * bM <= S and (tc[1] + 1) * bN <= O:
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        part = part + acc[i] * tCgE[i].to(cutlass.Float32)
                else:
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        if cute.elem_less(tCcE[i][0], S) and cute.elem_less(tCcE[i][1], O):
                            part = part + acc[i] * tCgE[i].to(cutlass.Float32)
                part = cute.arch.warp_reduction_sum(part)
                if lane == 0:
                    sRed[mma_warp] = part
                self.red_barrier_a.arrive_and_wait()
                if mma_tidx == 0 and tc[0] < m_tiles and tc[1] < n_tiles:   # scheduler padding never writes
                    total = cutlass.Float32(0.0)
                    for w in cutlass.range_constexpr(self.num_mma_warps):
                        total = total + sRed[w]
                    mOut[tc[2], tc[0] * n_tiles + tc[1]] = total
                self.red_barrier_b.arrive_and_wait()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


# ---------------------------------------------------------------------- configuration + torch entry points
_CUTE_DT = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_COMPILED: Dict[tuple, object] = {}
_NUM_SMS: Optional[int] = None

# (tile_mn, cluster_mn, swizzle_size, raster_along_m); measured on H200 for the Qwen3-14B linears (K = 2048..8192):
# the 128x256 tile with a 2x1 cluster (B tile multicast) is within 1-3 % of cuBLAS once the problem has a few waves of
# tiles; smaller problems (k/v projections, q/o at 128x256 = 800 tiles) get more CTAs from the 128x128 tile.  Problems of
# 1-1.5 waves of 128x128 tiles (the MLP projections of sub-billion models) run 20-25 % faster as 128x256 tiles in a 2x2
# cluster (both operands multicast): half the CTAs, a single wave and half the L2 traffic instead of a nearly empty second wave.
CONFIG_LARGE = ((128, 256), (2, 1), 4, True)
CONFIG_SMALL = ((128, 128), (2, 1), 8, True)
CONFIG_WIDE = ((128, 256), (2, 2), 2, True)


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NUM_SMS


def pick_config(O: int, I: int) -> tuple:
    """Tile configuration for an ``[O, I]`` weight gradient: the large tile once it fills >= 8 waves of SMs
    (the MLP projections and the lm_head of a 14B model); below that the tail of the last wave costs more than
    the larger tile saves (q/o projections: 800 tiles of 128x256 on 132 SMs), except for problems of 1-1.5 waves of
    128x128 tiles with an even tile grid, which take the wide 2x2-cluster tile."""
    m_tiles, n_tiles_wide = -(-O // 128), -(-I // 256)
    tiles_large = m_tiles * n_tiles_wide
    sms = _num_sms()
    if tiles_large >= 8 * sms:
        return CONFIG_LARGE
    tiles_small = m_tiles * -(-I // 128)
    if sms < tiles_small <= 1.5 * sms and m_tiles % 2 == 0 and n_tiles_wide % 2 == 0:
        return CONFIG_WIDE
    return CONFIG_SMALL


def _get(torch_dtype: torch.dtype, cfg: tuple):
    key = (torch_dtype, cfg)
    fn = _COMPILED.get(key)
    if fn is None:
        tile_mn, cluster_mn, swizzle_size, raster_along_m = cfg
        dt = _CUTE_DT[torch_dtype]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_mn[0] * cluster_mn[1])
        entry = HopperSelectedWgrad(dt, tile_mn, cluster_mn, swizzle_size, raster_along_m, max_active_clusters)

        def act():
            return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=8)),
                                            stride_order=(2, 1, 0), assumed_align=16)
        out = make_fake_compact_tensor(dt, (cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)),
                                       stride_order=(1, 0), assumed_align=16)
        sel = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8)
        scl = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = cute.compile(entry, act(), act(), sel, scl, out, stream, options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def supports(go_shape, inp_shape, dtype: torch.dtype) -> bool:
    """3-D bf16/fp16 activations with 16-byte rows (``O % 8 == I % 8 == 0``); any sequence length or selection size."""
    return (len(go_shape) == 3 and len(inp_shape) == 3 and go_shape[2] % 8 == 0 and inp_shape[2] % 8 == 0
            and dtype in _CUTE_DT)


def selected_wgrad(go: torch.Tensor, inp: torch.Tensor, sel: torch.Tensor, scale, out_dtype: Optional[torch.dtype] = None,
                   cfg: Optional[tuple] = None) -> torch.Tensor:
    """``scale * sum_{k in sel} go[k]^T inp[k]`` -> ``[O, I]`` in ``go.dtype``; ``sel`` int64 ``[K]`` (may be empty)."""
    B, S, O = go.shape
    I = inp.shape[2]
    if out_dtype is not None and out_dtype != go.dtype:
        raise ValueError("hopper selected_wgrad: out_dtype must equal the activation dtype")
    if not go.is_contiguous():
        go = go.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()
    out = torch.empty(O, I, dtype=go.dtype, device=go.device)
    if sel.numel() == 0:
        return out.zero_()
    sel64 = sel if (sel.dtype == torch.int64 and sel.is_contiguous()) else sel.to(torch.int64).contiguous()
    if isinstance(scale, torch.Tensor) and scale.dtype == torch.float32 and scale.numel() == 1 and scale.is_cuda:
        scale_t = scale.reshape(1)
    else:
        scale_t = torch.as_tensor(scale, device=go.device, dtype=torch.float32).reshape(1)
    _get(go.dtype, cfg or pick_config(O, I))(go, inp, sel64, scale_t, out)
    return out


GIP_TILE = (128, 128)


def _get_gip(torch_dtype: torch.dtype, tile_mn: tuple = GIP_TILE):
    key = ("gip", torch_dtype, tile_mn)
    fn = _COMPILED.get(key)
    if fn is None:
        dt = _CUTE_DT[torch_dtype]
        entry = HopperGip(dt, tile_mn, utils.HardwareInfo().get_max_active_clusters(1))

        def act():
            return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=8)),
                                            stride_order=(2, 1, 0), assumed_align=16)
        out = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=4)
        fn = cute.compile(entry, act(), act(), act(), act(), out, make_fake_stream(use_tvm_ffi_env_stream=True),
                          options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def gip_partials(go_t: torch.Tensor, inp_t: torch.Tensor, go_v: torch.Tensor, inp_v: torch.Tensor,
                 tile_mn: tuple = GIP_TILE) -> torch.Tensor:
    """Per-CTA partial sums of the ghost inner product, fp32 ``[B, V * m_tiles * n_tiles]``; ``.sum(1)`` = scores.
    go_t ``[B, S, O]``, inp_t ``[B, S, I]``, go_v ``[V, Sv, O]``, inp_v ``[V, Sv, I]`` (same half dtype, O % 8 == I % 8 == 0)."""
    B, S, O = go_t.shape
    V, Sv, _ = go_v.shape
    go_t, inp_t, go_v, inp_v = (t if t.is_contiguous() else t.contiguous() for t in (go_t, inp_t, go_v, inp_v))
    bm, bn = tile_mn
    out = torch.empty(B * V, (-(-S // bm)) * (-(-Sv // bn)), dtype=torch.float32, device=go_t.device)
    _get_gip(go_t.dtype, tile_mn)(go_t, go_v, inp_t, inp_v, out)
    return out.view(B, -1)


def supports_proj(go_shape, inp_shape, k1: int, k2: int, dtype: torch.dtype) -> bool:
    """Projection widths <= 64 and multiples of 8 (16-byte TMA rows), half-precision 3-D activations with O % 8 == I % 8 == 0."""
    return supports(go_shape, inp_shape, dtype) and 0 < k1 <= PROJ_N and 0 < k2 <= PROJ_N and k1 % 8 == 0 and k2 % 8 == 0


def _get_dual_proj(torch_dtype: torch.dtype):
    key = ("dual_proj", torch_dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        dt = _CUTE_DT[torch_dtype]
        entry = HopperDualProj(dt, utils.HardwareInfo().get_max_active_clusters(1))

        def x():
            return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(divisibility=8)), stride_order=(1, 0), assumed_align=16)

        def pm():
            return make_fake_compact_tensor(dt, (cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)), stride_order=(1, 0), assumed_align=16)
        out = make_fake_compact_tensor(dt, (2, cute.sym_int(), PROJ_N), stride_order=(2, 1, 0), assumed_align=16)
        fn = cute.compile(entry, x(), pm(), x(), pm(), out, make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def _get_outer_score(torch_dtype: torch.dtype, scores_mode: bool):
    key = ("outer_score", torch_dtype, scores_mode)
    fn = _COMPILED.get(key)
    if fn is None:
        dt = _CUTE_DT[torch_dtype]

        def proj():
            return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(), PROJ_N), stride_order=(2, 1, 0), assumed_align=16)
        out = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), PROJ_N, PROJ_N), stride_order=(2, 1, 0), assumed_align=16)
        scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
        sel = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8)
        counter = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
        corr = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        fn = cute.compile(HopperOuterScore(dt, scores_mode), proj(), proj(), out, scores, sel, counter, corr, cutlass.Int32(1),
                          cutlass.Int32(0), cutlass.Float32(1.0), make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


_SCRATCH: Dict[tuple, torch.Tensor] = {}


def _scratch(name: str, device: torch.device) -> torch.Tensor:
    """Per-device placeholders for the outputs a mode of HopperOuterScore does not write, and the arrival counter of the
    scores mode (zero between launches: the last CTA resets it, so the launches on one stream may share it)."""
    key = (name, device.index if device.index is not None else torch.cuda.current_device())
    t = _SCRATCH.get(key)
    if t is None:
        t = {"counter": lambda: torch.zeros(1, dtype=torch.int32, device=device),
             "out": lambda: torch.empty(1, PROJ_N, PROJ_N, dtype=torch.float32, device=device),
             "scores": lambda: torch.empty(1, dtype=torch.float32, device=device),
             "sel": lambda: torch.empty(1, dtype=torch.int64, device=device),
             "corr": lambda: torch.ones(1, dtype=torch.float32, device=device)}[name]()
        _SCRATCH[key] = t
    return t


def dual_proj(go: torch.Tensor, inp: torch.Tensor, P_O: torch.Tensor, P_I: torch.Tensor) -> torch.Tensor:
    """``[2, B, S, 64]`` in the activation dtype: ``[0] = go P_O``, ``[1] = inp P_I`` (columns beyond the projection widths
    are zero).  go ``[B, S, O]``, inp ``[B, S, I]``, P_O ``[O, k1]``, P_I ``[I, k2]`` (contiguous, same dtype)."""
    B, S, O = go.shape
    I = inp.shape[2]
    out = torch.empty(2, B * S, PROJ_N, dtype=go.dtype, device=go.device)
    _get_dual_proj(go.dtype)(go.reshape(B * S, O), P_O, inp.reshape(B * S, I), P_I, out)
    return out.view(2, B, S, PROJ_N)


def outer_partials(proj: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """fp32 ``[B, 1, 64, 64]``: ``scale * proj[0, b]^T proj[1, b]`` per sample (the compressed gradients)."""
    B = proj.shape[1]
    dev = proj.device
    out = torch.empty(B, PROJ_N, PROJ_N, dtype=torch.float32, device=dev)
    _get_outer_score(proj.dtype, False)(proj[0], proj[1], out, _scratch("scores", dev), _scratch("sel", dev),
                                        _scratch("counter", dev), _scratch("corr", dev), 1, 0, float(scale))
    return out.view(B, 1, PROJ_N, PROJ_N)


def outer_scores(proj: torch.Tensor, scale: float, n_train: int, k: int, corr) -> Tuple[torch.Tensor, torch.Tensor]:
    """``scores [n_train]`` = ``corr * <c_b, sum_v c_v>`` over a merged batch whose training rows come first (``c`` the scaled
    compressed gradients), and the k largest indices in ascending order (``k = 0`` -> empty); one launch."""
    B = proj.shape[1]
    if not (0 < n_train < B <= MAX_ROWS and 0 <= k <= n_train):
        raise ValueError("outer_scores: need 0 < n_train < B <= 256 (validation rows present) and 0 <= k <= n_train")
    dev = proj.device
    scores = torch.empty(n_train, dtype=torch.float32, device=dev)
    sel = torch.empty(max(k, 1), dtype=torch.int64, device=dev)
    if isinstance(corr, torch.Tensor):
        corr_t = corr.reshape(1) if (corr.dtype == torch.float32 and corr.is_cuda) else corr.to(device=dev, dtype=torch.float32).reshape(1)
    else:
        corr_t = torch.tensor([float(corr)], dtype=torch.float32, device=dev)
    _get_outer_score(proj.dtype, True)(proj[0], proj[1], _scratch("out", dev), scores, sel, _scratch("counter", dev), corr_t,
                                       int(n_train), int(k), float(scale))
    return scores, sel[:k]


def compressed_partials(go: torch.Tensor, inp: torch.Tensor, P_O: torch.Tensor, P_I: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """fp32 ``[B, 1, 64, 64]``: ``scale * (go_b P_O)^T (inp_b P_I)`` per sample, the projections rounded to the activation
    dtype like the reference; columns beyond the projection widths are zero."""
    return outer_partials(dual_proj(go, inp, P_O, P_I), scale)


def compressed_scores(go: torch.Tensor, inp: torch.Tensor, P_O: torch.Tensor, P_I: torch.Tensor, scale: float,
                      n_train: int, k: int, corr) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed scoring of a merged batch (training rows first) and the sorted top-k in two launches; see
    :func:`outer_scores`."""
    return outer_scores(dual_proj(go, inp, P_O, P_I), scale, n_train, k, corr)


# Raster along the token tiles (M fastest) already keeps the CTAs of one G tile together; a swizzle group larger than the
# number of token tiles mis-assigns tiles when the sample dimension L > 1, so no swizzle here.
PIP_TILE, PIP_SWIZZLE = (128, 256), 1


def _get_pip(torch_dtype: torch.dtype):
    key = ("pip", torch_dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        dt = _CUTE_DT[torch_dtype]
        entry = HopperPip(dt, PIP_TILE, PIP_SWIZZLE, utils.HardwareInfo().get_max_active_clusters(1))

        def act():
            return make_fake_compact_tensor(dt, (cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=8)),
                                            stride_order=(2, 1, 0), assumed_align=16)
        G = make_fake_compact_tensor(dt, (cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)), stride_order=(1, 0), assumed_align=16)
        out = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=4)
        fn = cute.compile(entry, act(), G, act(), out, make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _COMPILED[key] = fn
    return fn


def pip_partials(go: torch.Tensor, inp: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Per-CTA partial sums of the per-token inner product, fp32 ``[B, m_tiles * n_tiles]``; ``.sum(1)`` = scores.
    go ``[B, S, O]``, inp ``[B, S, I]``, G ``[O, I]`` (same half dtype, O % 8 == I % 8 == 0)."""
    B, S, O = go.shape
    go, inp, G = (t if t.is_contiguous() else t.contiguous() for t in (go, inp, G))
    out = torch.empty(B, (-(-S // PIP_TILE[0])) * (-(-O // PIP_TILE[1])), dtype=torch.float32, device=go.device)
    _get_pip(go.dtype)(inp, G, go, out)
    return out


__all__ = ["HopperSelectedWgrad", "HopperGip", "HopperDualProj", "HopperOuterScore", "HopperPip", "CONFIG_LARGE",
           "CONFIG_SMALL", "CONFIG_WIDE", "GIP_TILE", "PIP_TILE", "PROJ_N", "MAX_ROWS", "pick_config", "supports", "supports_proj",
           "selected_wgrad", "gip_partials", "dual_proj", "outer_partials", "outer_scores", "compressed_partials",
           "compressed_scores", "pip_partials"]
