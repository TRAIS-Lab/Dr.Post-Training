#!/usr/bin/env python3
"""Stage 1 of the rewrite-target pipeline: sample candidate answers for a target's rows.

For every row of ``eval/<source_target>/<source_target>_<split>_data.jsonl`` a
generator model (by default a much stronger Qwen3 than the model being trained)
writes ``num_samples`` candidate assistant answers. Two modes:

``solve``   (tag ``gen``)  the generator sees only the user prompt (math prompts are
                           wrapped in the MATH500 evaluation template so the answer
                           ends in ``\\boxed{}``) and solves the task from scratch;
``rewrite`` (tag ``rw``)   the generator sees the prompt AND the reference answer and
                           rewrites the answer into the requested format, keeping
                           its content / final answer.

Nothing is verified here. ``build_rewrite_target.py`` (stage 2) scores every
candidate with the domain verifier and assembles the new target, so this stage
can run in a vLLM environment while stage 2 runs in the training environment.
The model is loaded once for all ``--jobs``.

Output (one file per job): one line per source row::

    {"id": ..., "source_target": ..., "split": ..., "mode": "solve|rewrite",
     "generator": ..., "sampling": {...}, "candidates": ["...", ...]}

Example
-------
  python SFT/data/gen_target_candidates.py --data_dir $DATA --generator /path/to/Qwen3-32B \\
      --jobs precise_if:solve precise_if:rewrite math_persona:solve math_persona:rewrite \\
      --out_dir $DATA/candidates --tag 32b --backend vllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from SFT.data.target_common import (  # noqa: E402
    MODE_TAGS, assistant_reference, canonical_mode, infer_domain, read_jsonl, target_split_path,
    user_prompt, write_jsonl,
)

# Same template as SFT/eval/tasks/math500.py (duplicated so this stage does not import the eval stack).
MATH_SOLVE_TEMPLATE = (
    "Solve the following mathematics problem. Show your reasoning, then put only "
    "the final answer inside \\boxed{{}}.\n\n{problem}"
)

# Only the generator sees these; the stored row keeps the target's own user message.
SYSTEM_PROMPTS = {
    "if": "Follow every instruction in the request exactly, including all formatting and length constraints.",
    "math": None,
    "code": ("You are an expert Python programmer. Return one complete, self-contained solution in a single "
             "```python code block. Do not add prose outside the block."),
}

REWRITE_TEMPLATES = {
    "if": (
        "Below is a user request and a draft response to it. Rewrite the draft so that it satisfies every "
        "instruction in the request exactly (format, length, keywords, letter case, punctuation, language and "
        "structure), stays helpful, natural and well written, and contains nothing that violates the request. "
        "Output only the rewritten response, with no preamble or commentary.\n\n"
        "### Request\n{prompt}\n\n### Draft response\n{reference}"
    ),
    "math": (
        "Below is a mathematics problem and a reference solution. Rewrite the solution as a clear, "
        "well-organized step-by-step solution in your own words. Keep every final answer exactly the same as "
        "in the reference. End by putting the final answer inside \\boxed{{}}; if the problem has several "
        "parts, give each part's final answer in its own \\boxed{{}}. Output only the rewritten solution.\n\n"
        "### Problem\n{prompt}\n\n### Reference solution\n{reference}"
    ),
    "code": (
        "Below is a programming task and a reference solution. Rewrite the solution as one complete, "
        "self-contained Python function that passes the tests, in a single ```python code block with no "
        "prose outside it.\n\n### Task\n{prompt}\n\n### Reference solution\n{reference}"
    ),
}

DEFAULT_MAX_NEW_TOKENS = {"if": 2048, "math": 2048, "code": 1024}


def build_messages(domain: str, mode: str, row: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = user_prompt(row)
    messages: List[Dict[str, str]] = []
    if mode == "solve":
        system = SYSTEM_PROMPTS.get(domain)
        if system:
            messages.append({"role": "system", "content": system})
        content = MATH_SOLVE_TEMPLATE.format(problem=prompt) if domain == "math" else prompt
    else:
        content = REWRITE_TEMPLATES[domain].format(prompt=prompt, reference=assistant_reference(row))
    messages.append({"role": "user", "content": content})
    return messages


def render(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


class VllmBackend:
    def __init__(self, args):
        from vllm import LLM

        self.args = args
        self.llm = LLM(
            model=args.generator, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel,
            max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed, enable_prefix_caching=True, trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: Sequence[str], num_samples: int, max_new_tokens: int) -> List[List[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=num_samples, temperature=self.args.temperature, top_p=self.args.top_p, top_k=self.args.top_k,
            max_tokens=max_new_tokens, seed=self.args.seed,
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=True)
        return [[o.text for o in out.outputs] for out in outputs]


class HfBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cuda = torch.cuda.is_available()
        kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if cuda else torch.float32}
        if cuda:
            kwargs["device_map"] = "cuda"
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
        self.model = AutoModelForCausalLM.from_pretrained(args.generator, **kwargs).eval()
        stop = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        for marker in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(marker)
            if isinstance(token_id, int) and token_id >= 0:
                stop.add(token_id)
        self.stop_ids = sorted(stop)

    def generate(self, prompts: Sequence[str], num_samples: int, max_new_tokens: int) -> List[List[str]]:
        torch = self.torch
        results: List[List[str]] = [[] for _ in prompts]
        batch_size = self.args.batch_size
        for sample_index in range(num_samples):
            torch.manual_seed(self.args.seed + sample_index)
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start:start + batch_size])
                encoded = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                with torch.no_grad():
                    out = self.model.generate(
                        **encoded, do_sample=self.args.temperature > 0,
                        temperature=self.args.temperature if self.args.temperature > 0 else None,
                        top_p=self.args.top_p if self.args.temperature > 0 else None,
                        top_k=self.args.top_k if self.args.temperature > 0 else None,
                        max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.stop_ids or None,
                    )
                prompt_length = encoded["input_ids"].shape[1]
                texts = self.tokenizer.batch_decode(out[:, prompt_length:], skip_special_tokens=True)
                for offset, text in enumerate(texts):
                    results[start + offset].append(text)
                print(f"    sample {sample_index + 1}/{num_samples}: {min(start + batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)
        return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="dataset root containing eval/<target>/")
    parser.add_argument("--jobs", nargs="+", required=True,
                        help="<source_target>:<solve|rewrite|gen|rw> pairs, e.g. precise_if:solve math_persona:rewrite")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--generator", required=True, help="HF id or local path of the generator model")
    parser.add_argument("--generator_name", default=None, help="label stored in the output (default: basename of --generator)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--out_dir", required=True, help="where <target>_<gen|rw><tag>.jsonl files are written")
    parser.add_argument("--tag", default="", help="suffix identifying the generator, e.g. 32b -> precise_if_gen32b.jsonl")
    parser.add_argument("--num_samples", type=int, default=8, help="candidates per validation (D*) row")
    parser.add_argument("--num_samples_test", type=int, default=2, help="candidates per held-out test row")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=None, help="default: per domain (if/math 2048, code 1024)")
    parser.add_argument("--max_model_len", type=int, default=8192, help="vLLM context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16, help="prompts per generate call (hf backend)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test row cap per split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    jobs = []
    for spec in args.jobs:
        target, _, mode = spec.partition(":")
        if not mode:
            raise SystemExit(f"--jobs entries must look like <target>:<mode>, got {spec!r}")
        jobs.append((target, canonical_mode(mode)))
    generator_name = args.generator_name or os.path.basename(os.path.normpath(args.generator))
    os.makedirs(args.out_dir, exist_ok=True)

    pending = []
    for target, mode in jobs:
        out_path = os.path.join(args.out_dir, f"{target}_{MODE_TAGS[mode]}{args.tag}.jsonl")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip {target}:{mode}: {out_path} exists (--overwrite to redo)")
            continue
        pending.append((target, mode, out_path))
    if not pending:
        return 0

    print(f"loading generator {args.generator} with backend {args.backend}", flush=True)
    backend = VllmBackend(args) if args.backend == "vllm" else HfBackend(args)
    tokenizer = backend.tokenizer

    for target, mode, out_path in pending:
        domain = infer_domain(target)
        max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS[domain]
        out_rows: List[Dict[str, Any]] = []
        for split in args.splits:
            path = target_split_path(args.data_dir, target, split)
            if not os.path.exists(path):
                print(f"  {target}/{split}: {path} not found, skipping")
                continue
            rows = read_jsonl(path)
            if args.limit:
                rows = rows[: args.limit]
            num_samples = args.num_samples if split == "validation" else args.num_samples_test
            prompts = [render(tokenizer, build_messages(domain, mode, row)) for row in rows]
            started = time.time()
            print(f"  {target}:{mode}/{split}: {len(rows)} rows x {num_samples} samples, max_new_tokens {max_new_tokens}", flush=True)
            samples = backend.generate(prompts, num_samples, max_new_tokens)
            sampling = {"num_samples": num_samples, "temperature": args.temperature, "top_p": args.top_p,
                        "top_k": args.top_k, "max_new_tokens": max_new_tokens, "seed": args.seed,
                        "backend": args.backend, "enable_thinking": False}
            for row, texts in zip(rows, samples):
                out_rows.append({
                    "id": row["id"], "source_target": target, "split": split, "mode": mode,
                    "generator": generator_name, "generator_path": args.generator, "sampling": sampling,
                    "candidates": [text.strip() for text in texts],
                })
            print(f"  {target}:{mode}/{split}: done in {time.time() - started:.0f}s", flush=True)
        write_jsonl(out_path, out_rows)
        print(f"wrote {len(out_rows)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
