#!/usr/bin/env python
"""
Kernel-level profile of one curated training step.

Runs a few Standard / LayerWiseSubset / GlobalSubset-one-pass steps on a real model
under torch.profiler and attributes every CUDA kernel (via record_function
annotations + chrome-trace correlation ids) to the phase of the custom Linear
backward that launched it: act_grad, score, compress, select, w.grad, assembly,
plus the per-layer-type split (q/k/v/o/gate/up/down/lm_head).  Reports GPU-busy
vs GPU-wall per phase (how launch-bound the step is), kernel counts and the top
kernels per phase.  This is what motivated the fused kernels in drpt/kernels.

    python SFT/benchmark/profile_step.py --model Qwen/Qwen3-1.7B --method layerwise --scoring gip \
        --batch-size 8 --seq-length 512
    DRPT_FUSED_KERNELS=0 python SFT/benchmark/profile_step.py ...   # reference ops
"""
import argparse, json, os, sys, collections
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from SFT.benchmark.utils import (BenchmarkConfig, set_seed, setup_model, setup_grad_hook,
                                 create_dataloaders, get_batches, pad_and_merge_batches)
import drpt.selection.backward as _bwd
from drpt.compressor import Compressor
from drpt.hook import GradientHook


def _wrap_static(cls, name, label):
    orig = getattr(cls, name)
    @staticmethod
    def w(*a, **k):
        with record_function(label):
            return orig(*a, **k)
    setattr(cls, name, w)

def _wrap_fn(mod, name, label):
    orig = getattr(mod, name)
    def w(*a, **k):
        with record_function(label):
            return orig(*a, **k)
    setattr(mod, name, w)

def _wrap_layer_bwd(cls, label):
    orig = getattr(cls, "backward")
    @staticmethod
    def w(ctx, grad_output):
        hm = ctx.hook_manager_ref()
        lt = hm.layer_names[ctx.layer_idx].split(".")[-1] if hm is not None else "?"
        with record_function(label), record_function(f"layer/{lt}"):
            return orig(ctx, grad_output)
    setattr(cls, "backward", w)

from SFT.benchmark.trace_attribution import patch_annotations  # noqa: E402  (shared with the trainer's StepProfilerCallback)


def build(args):
    cfg = BenchmarkConfig(model_name=args.model, batch_size=args.batch_size, seq_length=args.seq_length,
                          val_batch_size=args.val_batch_size, dataset="dummy", num_warmup=args.warmup,
                          num_iterations=args.steps, scoring_method=args.scoring,
                          score_compression=("normal-64*64" if args.scoring == "compress" else "none"),
                          direct_batch_size=args.direct_batch_size,
                          gradient_checkpointing=args.ckpt)
    set_seed(cfg.seed)
    model, tok = setup_model(cfg)
    hook = setup_grad_hook(model, cfg, tok, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    tl, vl = create_dataloaders(cfg, tok)
    tb, vb = get_batches(tl, vl, cfg.num_warmup + cfg.num_iterations, cfg.device)
    return cfg, model, tok, hook, opt, tb, vb


def make_step(args, cfg, model, tok, hook, opt):
    pad = tok.pad_token_id or 0
    if args.method == "standard":
        hook.disable_hooks()
        def step(batch, val_batch):
            opt.zero_grad()
            with record_function("phase/forward"), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            with record_function("phase/backward"):
                loss.backward()
            with record_function("phase/optimizer"):
                opt.step()
        return step
    if args.method == "layerwise":
        def step(batch, val_batch):
            bs = batch["input_ids"].shape[0]
            merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad)
            hook.setup_selection(train_batch_size=bs, selection_method="LayerWiseSubset", frac=0.5,
                                 lr=5e-5, selection_mode="topk", use_second_order=False,
                                 scoring_method=args.scoring, direct_batch_size=cfg.direct_batch_size)
            hook.set_token_counts(merged["labels"], bs)
            opt.zero_grad()
            with record_function("phase/forward"), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**merged).loss
            with record_function("phase/backward"):
                loss.backward()
            hook.clear_selection(); hook.clear_token_counts()
            with record_function("phase/optimizer"):
                opt.step()
        return step
    if args.method == "onepass":
        if args.scoring != "compress":
            hook.score_compressors = [None] * len(hook.score_compressors)
        def step(batch, val_batch):
            bs = batch["input_ids"].shape[0]
            merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad)
            hook.setup_selection(train_batch_size=bs, selection_method="GlobalSubset", frac=0.5,
                                 lr=5e-5, selection_mode="topk", use_second_order=False,
                                 scoring_method=args.scoring, one_pass=True,
                                 direct_batch_size=cfg.direct_batch_size)
            hook.set_token_counts(merged["labels"], bs)
            opt.zero_grad()
            with record_function("phase/forward"), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**merged).loss
            with record_function("phase/backward"):
                loss.backward()
            with record_function("phase/selection"):
                st = hook.selection_state
                sel = st.get_final_selection().sort()[0]
            with record_function("phase/assembly"):
                sf = st._compute_scale_factor_for_assembly(sel)
                hook.assemble_gradients_from_retained(sel, sf)
            hook.clear_selection(); hook.clear_token_counts()
            with record_function("phase/optimizer"):
                opt.step()
        return step
    raise ValueError(args.method)


def analyze(trace_path, nsteps):
    ev = json.load(open(trace_path))["traceEvents"]
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = {e["args"]["correlation"]: e for e in ev
               if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    ann = [e for e in ev if e.get("cat") == "user_annotation"]
    # per-thread sorted annotation list
    by_tid = collections.defaultdict(list)
    for a in ann:
        by_tid[a["tid"]].append(a)
    for t in by_tid: by_tid[t].sort(key=lambda a: a["ts"])
    phase_ann = sorted([a for a in ann if a["name"].startswith("phase/")], key=lambda a: a["ts"])
    def enclosing(tid, ts):
        out = [a["name"] for a in phase_ann if a["ts"] <= ts <= a["ts"] + a.get("dur", 0)]
        for a in by_tid.get(tid, []):
            if a["name"].startswith("phase/"): continue
            if a["ts"] <= ts <= a["ts"] + a.get("dur", 0):
                out.append(a["name"])
        return out  # outer→inner (sorted by ts, nested ones start later)
    phase_busy = collections.Counter(); phase_cnt = collections.Counter()
    phase_kmin = {}; phase_kmax = {}
    inner_busy = collections.Counter(); inner_cnt = collections.Counter()
    topk_by = collections.defaultdict(collections.Counter)
    unattributed = 0.0
    layer_busy = collections.Counter(); layer_cnt = collections.Counter(); layer_inner = collections.Counter()
    for k in kernels:
        corr = k["args"].get("correlation"); r = runtime.get(corr)
        if r is None:
            unattributed += k["dur"]; continue
        labels = enclosing(r["tid"], r["ts"])
        phases = [l for l in labels if l.startswith("phase/")]
        drpt = [l for l in labels if l.startswith("drpt/")]
        lay = [l for l in labels if l.startswith("layer/")]
        if lay:
            layer_busy[lay[-1]] += k["dur"]; layer_cnt[lay[-1]] += 1
            layer_inner[(lay[-1], drpt[-1] if drpt else "?")] += k["dur"]
        ph = phases[-1] if phases else "phase/other"
        phase_busy[ph] += k["dur"]; phase_cnt[ph] += 1
        phase_kmin[ph] = min(phase_kmin.get(ph, 1e30), k["ts"]); phase_kmax[ph] = max(phase_kmax.get(ph, 0), k["ts"] + k["dur"])
        inner = drpt[-1] if drpt else (ph + "/rest")
        key = (ph, inner)
        inner_busy[key] += k["dur"]; inner_cnt[key] += 1
        topk_by[inner][k["name"][:90]] += k["dur"]
    # also cumulative time of top-level drpt Function (LW.linear_bwd etc.)
    top_busy = collections.Counter(); top_cnt = collections.Counter()
    for k in kernels:
        corr = k["args"].get("correlation"); r = runtime.get(corr)
        if r is None: continue
        labels = enclosing(r["tid"], r["ts"])
        for l in labels:
            if l in ("drpt/LW.linear_bwd", "drpt/GS.linear_bwd", "drpt/LW.embedding_bwd", "drpt/GS.embedding_bwd", "drpt/rmsnorm_bwd", "drpt/assembly"):
                top_busy[l] += k["dur"]; top_cnt[l] += 1
    # annotation CPU durations
    ann_cpu = collections.Counter(); ann_n = collections.Counter()
    for a in ann:
        ann_cpu[a["name"]] += a.get("dur", 0); ann_n[a["name"]] += 1
    s = 1.0 / nsteps / 1000.0  # us → ms per step
    print(f"\n=== per-step summary ({nsteps} profiled steps) ===")
    print(f"{'phase':<18}{'gpu_wall ms':>12}{'gpu_busy ms':>12}{'idle ms':>10}{'#kernels':>10}{'cpu_wall ms':>12}")
    for ph in sorted(phase_busy):
        wall = (phase_kmax[ph] - phase_kmin[ph]) * s if ph in phase_kmin else 0
        # gpu_wall across steps is wrong if computed over whole trace; compute per-step by summing per-step spans instead
        print(f"{ph:<18}{'':>12}{phase_busy[ph]*s:>12.1f}{'':>10}{phase_cnt[ph]/nsteps:>10.0f}{ann_cpu[ph]*s:>12.1f}")
    print(f"unattributed kernel time: {unattributed*s:.2f} ms/step")
    print(f"\n--- custom Function totals (kernels launched inside) ---")
    for l, v in top_busy.most_common():
        print(f"{l:<28}{v*s:>10.1f} ms   {top_cnt[l]/nsteps:>7.0f} kernels   cpu {ann_cpu[l]*s:>8.1f} ms  ({ann_n[l]/nsteps:.0f} calls)")
    print(f"\n--- innermost attribution (GPU busy ms/step, #kernels/step, avg us/kernel) ---")
    for (ph, inner), v in sorted(inner_busy.items(), key=lambda x: -x[1]):
        n = inner_cnt[(ph, inner)]
        print(f"{ph:<18}{inner:<26}{v*s:>9.1f} ms {n/nsteps:>8.0f} k  {v/n:>8.1f} us")
    print(f"\n--- per layer type (GPU busy ms/step inside the custom Linear backward) ---")
    for l, v in layer_busy.most_common():
        parts = "  ".join(f"{i.split('/')[-1]}={layer_inner[(l,i)]*s:.1f}" for (ll,i) in layer_inner if ll == l)
        print(f"{l:<16}{v*s:>9.1f} ms  {layer_cnt[l]/nsteps:>6.0f} k   [{parts}]")
    print(f"\n--- top kernels per drpt annotation ---")
    for inner in sorted(topk_by, key=lambda i: -sum(topk_by[i].values())):
        if not inner.startswith("drpt/"): continue
        print(f"[{inner}]")
        for nm, v in topk_by[inner].most_common(6):
            print(f"    {v*s:>8.2f} ms  {nm}")
    # GPU wall per phase, per step (span of kernels inside each phase instance)
    print(f"\n--- GPU wall per phase (avg over steps; span of kernels launched in that phase instance) ---")
    phase_inst = [a for a in ann if a["name"].startswith("phase/")]
    spans = collections.defaultdict(list)
    for a in phase_inst:
        ks = [k for k in kernels if runtime.get(k["args"].get("correlation")) is not None
              and a["ts"] <= runtime[k["args"]["correlation"]]["ts"] <= a["ts"] + a["dur"]]
        if ks:
            spans[a["name"]].append((max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks), sum(k["dur"] for k in ks), a["dur"]))
    for ph, lst in spans.items():
        w = sum(x[0] for x in lst) / len(lst) / 1000; b = sum(x[1] for x in lst) / len(lst) / 1000; c = sum(x[2] for x in lst) / len(lst) / 1000
        print(f"{ph:<18} gpu_wall {w:>8.1f} ms   gpu_busy {b:>8.1f} ms   idle {w-b:>7.1f} ms ({100*(w-b)/max(w,1e-9):4.1f}%)   cpu_wall {c:>8.1f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--method", default="layerwise", choices=["standard", "layerwise", "onepass"])
    p.add_argument("--scoring", default="pip", choices=["pip", "gip", "direct", "compress"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--direct-batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--trace", default=None, help="chrome trace output path")
    p.add_argument("--trace-dir", default=os.path.join(REPO, "SFT", "benchmark", "results", "profiles"),
                   help="directory for the chrome trace when --trace is not given")
    args = p.parse_args()
    from drpt.kernels import fused_kernels_enabled
    print(f"fused kernels: {'on' if fused_kernels_enabled() else 'off'}")
    patch_annotations()
    cfg, model, tok, hook, opt, tb, vb = build(args)
    step = make_step(args, cfg, model, tok, hook, opt)
    for i in range(args.warmup):
        step(tb[i], vb[i])
    torch.cuda.synchronize()
    # plain CUDA-event timing (no profiler) for reference
    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for i in range(args.steps):
        step(tb[args.warmup + i], vb[args.warmup + i])
    ev1.record(); torch.cuda.synchronize()
    print(f"\nplain step time: {ev0.elapsed_time(ev1)/args.steps:.1f} ms  (peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB)")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(args.steps):
            step(tb[args.warmup + i], vb[args.warmup + i])
        torch.cuda.synchronize()
    trace = args.trace or os.path.join(
        args.trace_dir, f"trace_{args.method}_{args.scoring}_{os.path.basename(args.model)}_n{args.batch_size}_T{args.seq_length}.json")
    os.makedirs(os.path.dirname(trace) or ".", exist_ok=True)
    prof.export_chrome_trace(trace)
    print(f"\nchrome trace: {trace}")
    analyze(trace, args.steps)

if __name__ == "__main__":
    main()
