"""Quick qualitative inspection: load a checkpoint, generate on a few test examples,
print prompt / gold / prediction. Use to sanity-check whether high-metric checkpoints
are producing meaningful output or gibberish that happens to overlap.

Usage:
    python -m SFT.eval.inspect_generations <model_path> <task> [n_examples]
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    adapter = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter):
        with open(adapter) as f:
            base = json.load(f)["base_model_name_or_path"]
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
        m = PeftModel.from_pretrained(m, model_path).merge_and_unload()
    else:
        m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    m.eval()
    return m, tok


def gen(model, tok, prompt, max_new):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def inspect_tydiqa(model, tok, n, data_dir):
    from SFT.eval.tasks.tydiqa import get_tydiqa_dataset_df, load_oneshot_examples
    one_shot = load_oneshot_examples(data_dir)
    examples = get_tydiqa_dataset_df(
        data_dir=data_dir, split="test", use_chat_format=True,
        chat_format="tulu", k=n, oneshot_examples=one_shot,
    )
    for i, (prompt, answer, lang) in enumerate(examples):
        out = gen(model, tok, prompt, 50)
        # Trim at next user turn marker if present
        for stop in ["<|user|>", "\n\n"]:
            if stop in out:
                out = out[:out.find(stop)]
        print(f"\n[{i}] lang={lang}")
        print(f"  GOLD: {answer!r}")
        print(f"  PRED: {out.strip()!r}")


def inspect_triviaqa(model, tok, n, data_dir):
    """Print model generations on TriviaQA closed-book test prompts."""
    from SFT.data.get_val_dataset import load_unified_jsonl
    examples = load_unified_jsonl(data_dir, "triviaqa", split="test", k=n)
    nl = "\n"
    q_prefix = "Answer the following question." + nl + "Question: "
    for i, ex in enumerate(examples):
        user = ex["messages"][0]["content"]
        gold = ex["metadata"].get("primary_answer", "")
        prompt = f"<|user|>{nl}{user}{nl}<|assistant|>{nl}"
        out = gen(model, tok, prompt, 32)
        for stop in ["<|user|>", "<|assistant|>", nl]:
            if stop in out:
                out = out[:out.find(stop)]
        out = out.strip()
        question_only = user.replace(q_prefix, "")
        print(f"\n[{i}]")
        print(f"  Q:     {question_only!r}")
        print(f"  GOLD:  {gold!r}")
        print(f"  PRED:  {out!r}")


def inspect_hhrlhf(model, tok, n, data_dir):
    """Print model generations on CategoricalHarmfulQA prompts."""
    from SFT.data.get_val_dataset import load_unified_jsonl
    examples = load_unified_jsonl(data_dir, "hhrlhf", split="test", k=n)
    for i, ex in enumerate(examples):
        user = ex["messages"][0]["content"]
        cat = ex.get("metadata", {}).get("category", "?")
        prompt = f"<|user|>\n{user}\n<|assistant|>\n"
        out = gen(model, tok, prompt, 200)
        for stop in ["<|user|>"]:
            if stop in out:
                out = out[:out.find(stop)]
        print(f"\n[{i}] cat={cat}")
        print(f"  PROMPT: {user[:120]!r}")
        print(f"  PRED:   {out.strip()[:300]!r}")


def inspect_samsum(model, tok, n, data_dir):
    from SFT.data.get_val_dataset import load_unified_jsonl
    examples = load_unified_jsonl(data_dir, "samsum", split="test", k=n)
    for i, ex in enumerate(examples):
        user = ex["messages"][0]["content"]
        gold = ex["messages"][1]["content"]
        prompt = f"<|user|>\n{user}\n<|assistant|>\n"
        out = gen(model, tok, prompt, 128)
        for stop in ["<|user|>"]:
            if stop in out:
                out = out[:out.find(stop)]
        print(f"\n[{i}]")
        print(f"  GOLD: {gold!r}")
        print(f"  PRED: {out.strip()!r}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m SFT.eval.inspect_generations <model_path> <task> [n]")
        sys.exit(1)
    model_path, task = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    data_dir = os.environ.get(
        "SFT_DATA_DIR",
        "/work/hdd/bfwm/phu1/Project/Dr.Post-Training/SFT/data",
    )

    print(f"\n{'='*70}\nMODEL: {os.path.basename(model_path)}\nTASK: {task}\n{'='*70}")
    model, tok = load(model_path)

    if task == "tydiqa":
        inspect_tydiqa(model, tok, n, data_dir)
    elif task == "samsum":
        inspect_samsum(model, tok, n, data_dir)
    elif task == "hhrlhf":
        inspect_hhrlhf(model, tok, n, data_dir)
    elif task == "triviaqa":
        inspect_triviaqa(model, tok, n, data_dir)
    else:
        print(f"Unknown task: {task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
