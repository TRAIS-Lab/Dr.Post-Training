"""Token-level parity of the GREATS-grounded encodings with GREATS's own code (CPU; needs the Mistral-7B-v0.1 and Meta-Llama-3-8B
tokenizers, the official LESS zip, and the files of SFT/data/prepare_greats_data.py).

(A) LESS pool rows: GREATS / LESS encode_with_messages_format (tokenizer default special tokens -> BOS) vs SFT encode_data with
    chat_template less_tulu + add_bos, through the multi-process datasets.map the trainer uses.
(B) TyDiQA target rows: GREATS get_tydiqa_dataset + tokenize, rebuilt from the LESS zip, vs get_dataset("tydiqa_less") with add_bos.
(C) torchtune Llama-3 rows: tokenize_messages (BOS + header / role / "\\n\\n" / body / <|eot_id|> pieces, assistant message
    supervised, masked EOS) vs the alpaca_tt pool and get_dataset("samsum_tt") with add_bos. The masked trailing EOS is dropped from
    the reference: it is the last token and carries no label, so it changes no loss or gradient.
(D) add_bos off leaves the repo encoding unchanged (no BOS).

    TORCHDYNAMO_DISABLE=1 python tests/test_greats_encoding.py --less_zip <less-data.zip> [--data_dir <SFT data dir>]
"""
import argparse
import json
import os
import random
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from datasets import Dataset, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from SFT.data.chat_format import CHAT_TEMPLATES, encode_prompt_completion, ensure_chat_template, set_add_bos  # noqa: E402
from SFT.data.get_train_dataset import encode_data  # noqa: E402
from SFT.data.get_val_dataset import get_dataset  # noqa: E402

MAX_LEN = 512


# ---- GREATS / LESS reference (less/data_selection/get_training_dataset.py, get_validation_dataset.py), verbatim logic ----
def less_concat_messages(messages, tokenizer):
    text = ""
    for m in messages:
        if m["role"] == "system":
            text += "<|system|>\n" + m["content"].strip() + "\n"
        elif m["role"] == "user":
            text += "<|user|>\n" + m["content"].strip() + "\n"
        elif m["role"] == "assistant":
            text += "<|assistant|>\n" + m["content"].strip() + tokenizer.eos_token + "\n"
    return text


def less_encode_with_messages_format(example, tokenizer, max_seq_length):
    messages = example["messages"]
    text = less_concat_messages(messages, tokenizer)
    input_ids = tokenizer(text, max_length=max_seq_length, truncation=True).input_ids
    labels = list(input_ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            start = 0 if i == 0 else len(tokenizer(less_concat_messages(messages[:i], tokenizer), max_length=max_seq_length,
                                                   truncation=True).input_ids)
            so_far = less_concat_messages(messages[:i + 1], tokenizer)
            if i < len(messages) - 1 and messages[i + 1]["role"] == "assistant":
                so_far += "<|assistant|>\n"
            end = len(tokenizer(so_far, max_length=max_seq_length, truncation=True).input_ids)
            labels[start:end] = [-100] * (end - start)
            if end >= max_seq_length:
                break
    return input_ids, labels


def less_tokenize(tokenizer, query, completion, max_length):
    full = tokenizer.encode(query + completion, max_length=max_length, truncation=True)
    prompt_len = min(len(tokenizer.encode(query)), len(full))
    return full, [-100] * prompt_len + full[prompt_len:]


TYDIQA_TEMPLATES = {
    "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
    "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
    "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
    "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
    "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
    "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
    "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
    "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
    "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:"),
}


def greats_tydiqa(tokenizer, squad_json, k):
    random.seed(42)
    data = squad_json["data"]
    random.shuffle(data)
    out = []
    for i, ex in enumerate(data):
        if i == k:
            break
        qa = ex["paragraphs"][0]["qas"][0]
        lang = qa["id"].split("-")[0]
        prompt, p, q, a = TYDIQA_TEMPLATES[lang]
        prompt += p + " " + ex["paragraphs"][0]["context"] + "\n" + q + " " + qa["question"] + "\n"
        prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a
        out.append(less_tokenize(tokenizer, prompt, " " + qa["answers"][0]["text"], MAX_LEN))
    return out


# ---- torchtune v0.1 reference (AlpacaInstructTemplate, SummarizeTemplate, TikTokenTokenizer.tokenize_messages) ----
ALPACA_IN = ("Below is an instruction that describes a task, paired with an input that provides further context. "
             "Write a response that appropriately completes the request.\n\n"
             "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
ALPACA_NO_IN = ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n")
SUMMARIZE = "Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"


def torchtune_tokenize(tok, user, assistant, max_seq_len):
    sid, eid, eot = (tok.convert_tokens_to_ids(t) for t in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731

    def message(role, content):
        return [sid] + enc(role) + [eid] + enc("\n\n") + enc(content.strip()) + [eot]

    u, a = message("user", user), message("assistant", assistant)
    tokens = [tok.bos_token_id] + u + a + [tok.eos_token_id]
    mask = [True] + [True] * len(u) + [False] * len(a) + [True]
    if max_seq_len and len(tokens) > max_seq_len:          # torchtune truncate(): cut, force the last token to EOS (masked)
        tokens, mask = tokens[:max_seq_len], mask[:max_seq_len]
        tokens[-1], mask[-1] = tok.eos_token_id, True
        truncated = True
    else:
        tokens, mask, truncated = tokens[:-1], mask[:-1], False   # drop the masked trailing EOS (no loss, last position)
    return tokens, [-100 if m else t for t, m in zip(tokens, mask)], truncated


def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def check_less_pool(model, less_zip, n_per_source):
    tok = AutoTokenizer.from_pretrained(model)
    ensure_chat_template(tok)
    rows = []
    with zipfile.ZipFile(less_zip) as z:
        for name in ("flan_v2", "cot", "dolly", "oasst1"):
            lines = z.open(f"data/train/processed/{name}/{name}_data.jsonl").read().decode().splitlines()
            rows += [json.loads(line) for line in random.Random(0).sample(lines, n_per_source)]
    rows = [{"messages": r["messages"]} for r in rows]
    tok.chat_template = CHAT_TEMPLATES["less_tulu"]   # chat_template: less_tulu, as train.py sets it
    set_add_bos(tok, True)
    ours = encode_data(Dataset.from_list(rows), tok, MAX_LEN)
    bad = 0
    for i, r in enumerate(rows):
        ref_ids, ref_labels = less_encode_with_messages_format(r, tok, MAX_LEN)
        if as_list(ours[i]["input_ids"]) != ref_ids or as_list(ours[i]["labels"]) != ref_labels:
            bad += 1
    multi = sum(len(r["messages"]) > 2 for r in rows)
    print(f"(A) {model}: LESS pool {len(rows)} rows ({multi} multi-turn): {len(rows) - bad} identical, {bad} differ")
    set_add_bos(tok, False)
    plain = encode_data(Dataset.from_list(rows[:20]), tok, MAX_LEN)
    no_bos = all(as_list(plain[i]["input_ids"])[0] != tok.bos_token_id for i in range(20))
    print(f"(D) {model}: add_bos off -> no BOS on any row: {no_bos}")
    return bad == 0 and no_bos


def check_tydiqa(model, less_zip, data_dir):
    tok = AutoTokenizer.from_pretrained(model)
    ensure_chat_template(tok)
    set_add_bos(tok, True)
    ok = True
    with zipfile.ZipFile(less_zip) as z:
        for split, member, k in (("validation", "dev", 9), ("test", "test", 500)):
            ref = greats_tydiqa(tok, json.load(z.open(f"data/eval/tydiqa/{member}/tydiqa-goldp-v1.1-dev.json")), k)
            ours = get_dataset(task="tydiqa_less", data_dir=data_dir, tokenizer=tok, max_length=MAX_LEN, split=split, k=k,
                               seed=None, max_seq_length_threshold=None)   # file order = GREATS's seed-42 order
            same = sum(as_list(ours[i]["input_ids"]) == ref[i][0] and as_list(ours[i]["labels"]) == ref[i][1]
                       for i in range(len(ref)))
            print(f"(B) {model}: tydiqa_less/{split}: {same}/{len(ref)} rows identical to GREATS get_tydiqa_dataset + tokenize")
            ok &= same == len(ref) == len(ours)
    return ok


def check_torchtune(model, data_dir):
    tok = AutoTokenizer.from_pretrained(model)
    ensure_chat_template(tok)
    set_add_bos(tok, True)
    ok = True
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    pool = [json.loads(line) for line in open(os.path.join(data_dir, "train", "alpaca_tt", "alpaca_tt_data.jsonl"), encoding="utf-8")]
    assert len(pool) == len(alpaca)
    bad = trunc_bad = 0
    for ex, row in zip(alpaca, pool):
        user = (ALPACA_IN if ex["input"] else ALPACA_NO_IN).format(instruction=ex["instruction"], input=ex["input"])
        ref_ids, ref_labels, truncated = torchtune_tokenize(tok, user, ex["output"], MAX_LEN)
        enc = encode_prompt_completion(row, tok, MAX_LEN)
        if as_list(enc["input_ids"]) != ref_ids or as_list(enc["labels"]) != ref_labels:
            trunc_bad += truncated
            bad += not truncated
    print(f"(C) {model}: alpaca_tt {len(pool)} rows: {len(pool) - bad - trunc_bad} identical to torchtune, {bad} differ, "
          f"{trunc_bad} differ only by torchtune's forced EOS on a >{MAX_LEN}-token row")
    ok &= bad == 0
    samsum = load_dataset("knkarthick/samsum")
    for split, k, src in (("validation", 16, samsum["validation"]), ("test", 819, samsum["test"])):
        ours = get_dataset(task="samsum_tt", data_dir=data_dir, tokenizer=tok, max_length=10 ** 6, split=split, k=k, seed=None,
                           max_seq_length_threshold=None)
        same = 0
        for i in range(k):
            ref_ids, ref_labels, _ = torchtune_tokenize(tok, SUMMARIZE.format(dialogue=src[i]["dialogue"]), src[i]["summary"], None)
            same += as_list(ours[i]["input_ids"]) == ref_ids and as_list(ours[i]["labels"]) == ref_labels
        print(f"(C) {model}: samsum_tt/{split}: {same}/{k} rows identical to torchtune samsum_dataset (no max_seq_len)")
        ok &= same == k == len(ours)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--less_zip", required=True)
    ap.add_argument("--data_dir", default=os.path.join(REPO, "SFT", "data"))
    ap.add_argument("--n_per_source", type=int, default=100)
    args = ap.parse_args()
    ok = check_less_pool("mistralai/Mistral-7B-v0.1", args.less_zip, args.n_per_source)
    ok &= check_tydiqa("mistralai/Mistral-7B-v0.1", args.less_zip, args.data_dir)
    ok &= check_torchtune("meta-llama/Meta-Llama-3-8B", args.data_dir)
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
