"""
Fused GPU kernels for the drpt custom Linear backward.

Besides the activation gradient, the custom backward of a hooked ``nn.Linear`` does
three GEMM-shaped things per layer: score the training samples against the validation
gradient, select, and build the weight gradient from the selected samples only.  In
plain PyTorch each is several kernels with large intermediates (``[B, S, O]`` for the
per-token inner product, ``[B, V, S, S]`` for the ghost inner product, gathered copies
of ``(grad_output, input)`` plus a separate scaling pass for the curated w.grad).  The
kernels here fuse each into a single tensor-core kernel:

- ``gip_scores``      ghost inner product, no ``[B, V, S, S]`` intermediates
- ``pip_scores``      per-token inner product, no ``[B, S, O]`` intermediate
- ``selected_wgrad``  index-mapped GEMM over the selected samples with the item-count
                      scale (and bias gradient) fused in
- ``compressed_grad`` (CuTe backend) the ``compress`` scoring projection
                      ``(go P_O)^T (inp P_I)`` per sample in one kernel, used by
                      :meth:`drpt.compressor.Compressor.forward`

Two backends implement the same three functions:

- ``cute``   (default) CuTe DSL / CUTLASS Python DSL kernels, :mod:`drpt.kernels.cute_ops`
             (needs ``nvidia-cutlass-dsl`` and ``apache-tvm-ffi``)
- ``triton`` Triton kernels, :mod:`drpt.kernels.triton_ops` (kept as a fallback)

Select with ``DRPT_KERNEL_BACKEND=cute|triton|off`` (``off`` = PyTorch reference ops;
``DRPT_FUSED_KERNELS=0`` is an alias for ``off``) or :func:`set_backend` at runtime.  The
dispatch in :mod:`drpt.selection.utils` uses the kernels only for 3-D CUDA bf16/fp16
operands whose shapes the backend supports; everything else takes the reference path.
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_BACKENDS = ("cute", "triton", "off")


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


HAS_TRITON = _importable("triton")
HAS_CUTE = _importable("cutlass.cute") and _importable("tvm_ffi")


def _default_backend() -> str:
    env = os.environ.get("DRPT_KERNEL_BACKEND", "").strip().lower()
    if os.environ.get("DRPT_FUSED_KERNELS", "1").strip().lower() in ("0", "false", "off", "no"):
        return "off"
    if env in _BACKENDS:
        return env
    if env:
        raise ValueError(f"DRPT_KERNEL_BACKEND must be one of {_BACKENDS}, got {env!r}")
    return "cute" if HAS_CUTE else ("triton" if HAS_TRITON else "off")


_BACKEND = _default_backend()
_OPS = None


def set_backend(name: str) -> None:
    """Select the kernel backend: ``"cute"``, ``"triton"`` or ``"off"`` (reference PyTorch ops)."""
    global _BACKEND, _OPS
    name = name.strip().lower()
    if name not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {name!r}")
    if name == "cute" and not HAS_CUTE:
        raise RuntimeError("CuTe backend requested but nvidia-cutlass-dsl / apache-tvm-ffi are not importable")
    if name == "triton" and not HAS_TRITON:
        raise RuntimeError("Triton backend requested but triton is not importable")
    _BACKEND = name
    _OPS = None


def set_fused_kernels(enabled: bool) -> None:
    """Backwards-compatible switch: ``False`` -> ``off``; ``True`` -> the default backend."""
    set_backend(_default_backend_ignoring_off() if enabled else "off")


def _default_backend_ignoring_off() -> str:
    env = os.environ.get("DRPT_KERNEL_BACKEND", "").strip().lower()
    if env in ("cute", "triton"):
        return env
    return "cute" if HAS_CUTE else ("triton" if HAS_TRITON else "off")


def backend() -> str:
    """Name of the active backend."""
    return _BACKEND


def fused_kernels_enabled() -> bool:
    """True when a fused-kernel backend (cute or triton) is active."""
    return _BACKEND != "off"


def kernel_ops():
    """Module providing ``pip_scores``, ``gip_scores``, ``selected_wgrad`` and ``supports`` for the active backend."""
    global _OPS
    if _OPS is None:
        if _BACKEND == "cute":
            from . import cute_ops as ops
        elif _BACKEND == "triton":
            from . import triton_ops as ops
        else:
            raise RuntimeError("fused kernels are off")
        _OPS = ops
    return _OPS


def fused_ok(*tensors: Optional[torch.Tensor], ndim: int = 3, op: Optional[str] = None) -> bool:
    """Whether the active backend can take these operands.

    Requires CUDA tensors of one half-precision dtype with ``ndim`` dims (the
    ``[B, S, features]`` layout of sequence data) and, when ``op`` is given, shapes the
    backend supports (see ``supports`` of the backend module).  2-D, fp32 and CPU inputs
    always take the reference path.
    """
    if not fused_kernels_enabled():
        return False
    dtype = None
    for t in tensors:
        if t is None or not t.is_cuda or t.dim() != ndim or t.dtype not in _SUPPORTED_DTYPES:
            return False
        if dtype is None:
            dtype = t.dtype
        elif t.dtype != dtype:
            return False
    if op is not None:
        return kernel_ops().supports(op, *(t.shape for t in tensors))
    return True


__all__ = ["HAS_TRITON", "HAS_CUTE", "backend", "set_backend", "set_fused_kernels",
           "fused_kernels_enabled", "fused_ok", "kernel_ops"]
