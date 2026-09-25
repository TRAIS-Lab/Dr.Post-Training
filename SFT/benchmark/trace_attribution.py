"""Kernel-level attribution of real training steps (torch.profiler chrome trace).

Used in two places:

* ``profile_step.py`` (synthetic harness) and the opt-in ``StepProfilerCallback`` of
  ``SFT/train/trainer.py`` call :func:`patch_annotations` to wrap the drpt custom
  backward in ``record_function`` labels (``drpt/score``, ``drpt/select``, ``drpt/wgrad``,
  ``drpt/assembly``, ...).  Without a profiler the labels cost ~1 us each.
* :func:`attribute_trace` reads the exported trace, links every CUDA kernel to the CPU
  op that launched it (correlation id) and classifies it into the rows of the paper's
  per-step tables: forward, a.grad, scoring, w.grad, autograd, optimizer.  Backward
  kernels are recognised by the ``autograd::engine::evaluate_function`` op enclosing the
  launch; inside the drpt custom Functions the labels decide; for plain ``nn.Linear``
  layers the two ``aten::mm`` of the backward are told apart by their output shape
  (weight-shaped -> w.grad, activation-shaped -> a.grad).

Enable in a real run with ``DRPT_PROFILE_STEPS=30:40 DRPT_PROFILE_TRACE=<dir>/trace.json``
(steps 30..39 are profiled after warm-up; ``profile.json`` with the per-step ms is written
next to the trace, which is deleted unless ``DRPT_PROFILE_KEEP_TRACE=1``).
"""
from __future__ import annotations

import collections
import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

CATEGORIES = ("forward", "a.grad", "scoring", "w.grad", "autograd", "optimizer")

_WGRAD_ANY = {"drpt/wgrad", "drpt/assembly", "drpt/store_update"}
_SCORING = {"drpt/score", "drpt/score_bias", "drpt/compress", "drpt/select", "drpt/split", "drpt/augment_bias",
            "drpt/LW.full", "drpt/LW.compressed", "drpt/GS.accum_full", "drpt/GS.accum_compressed",
            "drpt/LW.embedding_bwd", "drpt/GS.embedding_bwd"}
_ACTGRAD_ONLY = {"drpt/LW.linear_bwd", "drpt/GS.linear_bwd", "drpt/compressed_linear_bwd"}
_AUTOGRAD = {"drpt/rmsnorm_bwd", "drpt/retain"}

_PATCHED = False


def _wrap_attr(owner, name: str, label: str, static: bool = False) -> bool:
    """Wrap ``owner.name`` in ``record_function(label)`` if it exists."""
    orig = getattr(owner, name, None)
    if orig is None:
        return False
    if static:
        @staticmethod
        def w(*a, **k):
            with record_function(label):
                return orig(*a, **k)
    else:
        def w(*a, **k):
            with record_function(label):
                return orig(*a, **k)
    setattr(owner, name, w)
    return True


def _wrap_layer_bwd(cls, label: str) -> None:
    orig = getattr(cls, "backward")

    @staticmethod
    def w(ctx, grad_output):
        hm = ctx.hook_manager_ref() if hasattr(ctx, "hook_manager_ref") else None
        lt = hm.layer_names[ctx.layer_idx].split(".")[-1] if (hm is not None and hasattr(ctx, "layer_idx")) else "?"
        with record_function(label), record_function(f"layer/{lt}"):
            return orig(ctx, grad_output)
    setattr(cls, "backward", w)


def patch_annotations() -> None:
    """Annotate the drpt backward with ``record_function`` labels (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    import drpt.selection.backward as _bwd
    from drpt.compressor import Compressor
    from drpt.hook import GradientHook
    from drpt.selection import state as _state

    _wrap_layer_bwd(_bwd.LayerWiseSubsetLinearBackward, "drpt/LW.linear_bwd")
    _wrap_attr(_bwd.LayerWiseSubsetLinearBackward, "_backward_full", "drpt/LW.full", static=True)
    _wrap_attr(_bwd.LayerWiseSubsetLinearBackward, "_backward_compressed", "drpt/LW.compressed", static=True)
    _wrap_attr(_bwd.LayerWiseSubsetEmbeddingBackward, "backward", "drpt/LW.embedding_bwd", static=True)
    _wrap_layer_bwd(_bwd.GlobalSubsetLinearBackward, "drpt/GS.linear_bwd")
    _wrap_attr(_bwd.GlobalSubsetLinearBackward, "_accumulate_full", "drpt/GS.accum_full", static=True)
    _wrap_attr(_bwd.GlobalSubsetLinearBackward, "_accumulate_compressed", "drpt/GS.accum_compressed", static=True)
    _wrap_attr(_bwd.GlobalSubsetEmbeddingBackward, "backward", "drpt/GS.embedding_bwd", static=True)
    _wrap_attr(_bwd.TrainOnlyRMSNormBackward, "backward", "drpt/rmsnorm_bwd", static=True)
    if hasattr(_bwd, "CompressedLinearBackward"):   # MeSO baseline: a.grad + compressed update gradient
        _wrap_attr(_bwd.CompressedLinearBackward, "backward", "drpt/compressed_linear_bwd", static=True)
    for fn, label in (("_dispatch_scoring", "drpt/score"), ("exact_scores_fused", "drpt/score"),
                      ("compressed_scores_fused", "drpt/score"), ("_add_bias_scores", "drpt/score_bias"),
                      ("_do_selection", "drpt/select"), ("_produce_gradient_update", "drpt/wgrad"),
                      ("_store_update_grad", "drpt/store_update"), ("augment_input_for_bias", "drpt/augment_bias"),
                      ("split_train_val_batch", "drpt/split")):
        _wrap_attr(_bwd, fn, label)
    _wrap_attr(Compressor, "forward", "drpt/compress")
    _wrap_attr(GradientHook, "assemble_gradients_from_retained", "drpt/assembly")
    _wrap_attr(GradientHook, "retain_layer_data", "drpt/retain")
    # Global Subset: selection after (or, grouped, inside) backward; the nested assembly keeps its own label
    for cls_name in ("SelectionState", "GlobalSubsetState", "GroupedSelectionState"):
        cls = getattr(_state, cls_name, None)
        if cls is None:
            continue
        for fn in ("_select_indices", "_select_from_accumulators", "get_final_selection", "_finalize_group",
                   "_compute_scale_factor_for_assembly"):
            if fn in cls.__dict__:          # wrap where defined, not inherited copies
                _wrap_attr(cls, fn, "drpt/select")


def _mm_out_shape(name: str, dims) -> Optional[Tuple[int, int]]:
    try:
        if name == "aten::mm" and len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
            return (dims[0][0], dims[1][1])
        if name == "aten::addmm" and len(dims) >= 3 and len(dims[1]) == 2 and len(dims[2]) == 2:
            return (dims[1][0], dims[2][1])
    except (TypeError, IndexError):
        pass
    return None


def classify(stack, weight_shapes) -> str:
    """Category of a kernel from the CPU ops / annotations enclosing its launch (outer -> inner)."""
    names = [e["name"] for e in stack]
    if any(n.startswith("Optimizer.") for n in names):
        return "optimizer"
    in_bwd = any(n.startswith("autograd::engine::evaluate_function") or n == "torch::autograd::AccumulateGrad" for n in names)
    if not in_bwd and any(n.startswith("aten::_foreach_") for n in names):
        return "optimizer"      # gradient clipping (foreach norm + scale) runs between backward and optimizer.step
    if any(n in _WGRAD_ANY for n in names):
        return "w.grad"
    if "drpt/compressed_linear_bwd" in names:            # MeSO baseline layer
        return "w.grad" if "drpt/compress" in names else "a.grad"
    drpt = [n for n in names if n.startswith("drpt/")]
    if drpt:
        inner = drpt[-1]
        if inner in _SCORING:
            return "scoring"
        if inner in _AUTOGRAD:
            return "autograd"
        if inner in _ACTGRAD_ONLY:
            return "a.grad"
        return "scoring"
    if in_bwd:
        gemm = [e for e in stack if e["name"] in ("aten::mm", "aten::addmm")]
        if gemm:
            out = _mm_out_shape(gemm[-1]["name"], gemm[-1].get("args", {}).get("Input Dims") or [])
            if out is not None and (out in weight_shapes or (out[1], out[0]) in weight_shapes):
                return "w.grad"
            return "a.grad"
        return "autograd"
    return "forward"


def attribute_trace(trace_path: str, nsteps: int, weight_shapes: Iterable[Tuple[int, int]] = ()) -> Dict:
    """Per-step GPU-busy ms per category (see module docstring) from a chrome trace of ``nsteps`` steps."""
    weight_shapes = {tuple(s) for s in weight_shapes}
    ev = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = {e["args"]["correlation"]: e for e in ev
               if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    intervals = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cpu_op", "user_annotation") and "dur" in e and "ts" in e:
            intervals[e["tid"]].append(e)
    launches = collections.defaultdict(list)
    unattributed = 0.0
    for k in kernels:
        r = runtime.get(k["args"].get("correlation"))
        if r is None:
            unattributed += k["dur"]
        else:
            launches[r["tid"]].append((r["ts"], k))
    busy = collections.Counter(); count = collections.Counter()
    top = collections.defaultdict(collections.Counter)
    for tid, ls in launches.items():
        ivs = sorted(intervals.get(tid, []), key=lambda e: (e["ts"], -e["dur"]))
        ls.sort(key=lambda x: x[0])
        stack, i = [], 0
        for ts, k in ls:
            while i < len(ivs) and ivs[i]["ts"] <= ts:
                stack.append(ivs[i]); i += 1
            stack = [e for e in stack if e["ts"] + e["dur"] >= ts]     # drop finished ops
            cat = classify(stack, weight_shapes)
            busy[cat] += k["dur"]; count[cat] += 1
            top[cat][k["name"][:100]] += k["dur"]
    scale = 1.0 / nsteps / 1000.0   # us over nsteps -> ms per step
    ts0 = min(e["ts"] for e in ev if "ts" in e); ts1 = max(e["ts"] + e.get("dur", 0) for e in ev if "ts" in e)
    out = {
        "nsteps": nsteps,
        "busy_ms": {c: busy[c] * scale for c in CATEGORIES},
        "kernels_per_step": {c: count[c] / nsteps for c in CATEGORIES},
        "backward_ms": sum(busy[c] for c in ("a.grad", "scoring", "w.grad", "autograd")) * scale,
        "busy_total_ms": sum(busy.values()) * scale,
        "unattributed_ms": unattributed * scale,
        "trace_wall_ms": (ts1 - ts0) * scale,
        "top_kernels": {c: [(n, v * scale) for n, v in top[c].most_common(8)] for c in CATEGORIES},
    }
    return out


try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:  # the callback is only instantiated inside the trainer process
    class _TrainerCallback:  # pragma: no cover
        pass


class StepProfilerCallback(_TrainerCallback):
    """HF ``TrainerCallback`` profiling steps ``[start, end)`` of a real run (see module docstring)."""

    def __init__(self, trainer, steps: str, trace_path: str, keep_trace: bool = False):
        a, b = steps.split(":")
        self.start, self.end = int(a), int(b)
        self.trainer, self.trace_path, self.keep_trace = trainer, trace_path, keep_trace
        self.prof = None
        self.weight_shapes = set()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.prof is None and state.global_step == self.start:
            patch_annotations()
            model = kwargs.get("model") or self.trainer.model
            self.weight_shapes = {tuple(p.shape) for p in model.parameters() if p.dim() == 2}
            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            self.prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None and state.global_step >= self.end:
            torch.cuda.synchronize()
            self.prof.__exit__(None, None, None)
            os.makedirs(os.path.dirname(os.path.abspath(self.trace_path)), exist_ok=True)
            self.prof.export_chrome_trace(self.trace_path)
            nsteps = state.global_step - self.start
            res = attribute_trace(self.trace_path, nsteps, self.weight_shapes)
            res["steps"] = [self.start, state.global_step]
            res["weight_shapes"] = sorted(self.weight_shapes)
            with open(os.path.join(os.path.dirname(os.path.abspath(self.trace_path)), "profile.json"), "w") as f:
                json.dump(res, f, indent=1)
            if not self.keep_trace:
                os.remove(self.trace_path)
            self.prof = None
            b = res["busy_ms"]
            print(f"[drpt profile] steps {res['steps']}: " + "  ".join(f"{c} {b[c]:.1f}" for c in CATEGORIES)
                  + f"  | busy {res['busy_total_ms']:.1f} ms, trace wall {res['trace_wall_ms']:.1f} ms/step", flush=True)


def reattribute(root: str, delete_traces: bool = False) -> None:
    """Recompute ``profile.json`` from every kept ``<root>/*/trace.json`` (after a rule change)."""
    import glob
    for tr in sorted(glob.glob(os.path.join(root, "*", "trace.json"))):
        pj = os.path.join(os.path.dirname(tr), "profile.json")
        if not os.path.exists(pj):
            continue
        old = json.load(open(pj))
        res = attribute_trace(tr, old["nsteps"], [tuple(s) for s in old.get("weight_shapes", [])])
        res["steps"], res["weight_shapes"] = old.get("steps"), old.get("weight_shapes", [])
        json.dump(res, open(pj, "w"), indent=1)
        b = res["busy_ms"]
        print(os.path.basename(os.path.dirname(tr)) + ": " + "  ".join(f"{c} {b[c]:.1f}" for c in CATEGORIES))
        if delete_traces:
            os.remove(tr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="attribute an exported chrome trace")
    ap.add_argument("trace", nargs="?"); ap.add_argument("--nsteps", type=int)
    ap.add_argument("--weight-shapes", default="", help="comma list like 2048x8192,512x2048")
    ap.add_argument("--reattribute", metavar="DIR", help="recompute every DIR/*/profile.json from its kept trace.json")
    ap.add_argument("--delete-traces", action="store_true")
    a = ap.parse_args()
    if a.reattribute:
        reattribute(a.reattribute, a.delete_traces)
        raise SystemExit(0)
    if not a.trace or not a.nsteps:
        ap.error("trace and --nsteps are required")
    ws = [tuple(int(x) for x in s.split("x")) for s in a.weight_shapes.split(",") if s]
    r = attribute_trace(a.trace, a.nsteps, ws)
    print(json.dumps({k: v for k, v in r.items() if k != "top_kernels"}, indent=1))
    for c in CATEGORIES:
        print(f"[{c}]"); [print(f"   {v:7.2f} ms  {n}") for n, v in r["top_kernels"][c][:5]]
