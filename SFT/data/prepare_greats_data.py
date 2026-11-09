#!/usr/bin/env python
"""Data of the GREATS-grounded settings (Wang et al., "GREATS: Online Selection of High-Quality Data for LLM Training in Every
Iteration", NeurIPS 2024; github.com/Jiachen-T-Wang/GREATS), rendered with the exact strings GREATS's code feeds its tokenizer.
Rows are pre-rendered {"prompt", "completion"} pairs: the harness tokenizes prompt + completion verbatim, masks the prompt and
prepends BOS when the config sets add_bos (SFT/data/chat_format.py encode_prompt_completion, get_val_dataset.tokenize).

LESS -> TyDiQA (GREATS's LESS-based code, less/data_selection/get_validation_dataset.py):
  train/less_official/<name>_data.jsonl    the official LESS pool (princeton-nlp/less_data), messages format (tulu template)
  eval/tydiqa_less/tydiqa_less_validation_data.jsonl   D*: LESS's 9-row TyDiQA dev file (one per language; GREATS asks for 10)
  eval/tydiqa_less/tydiqa_less_test_data.jsonl         TyDiQA-GoldP dev (5,077 rows) shuffled with random.seed(42): its first
                                                       500 are GREATS's test set, its first 110 GREATS's test-perplexity rows
  eval/tydiqa_less/tydiqa_less_fewshot.json            LESS's one-shot demonstrations (1-shot F1, SFT/eval/tasks/greats.py)
  Row strings: "<|user|>\\n" + instruction + "Passage: <context>\\nQuestion: <question>\\n" + "<|assistant|>\\nAnswer:" and
  " <answer>" (no EOS), the language-specific open-instruct templates.
Alpaca -> SAMSum (GREATS's torchtune recipe; torchtune v0.1 AlpacaInstructTemplate / SummarizeTemplate and the Llama-3 tiktoken
tokenize_messages, which wraps every message in header tokens and supervises the whole assistant message):
  train/alpaca_tt/alpaca_tt_data.jsonl     tatsu-lab/alpaca (52,002 rows)
  eval/samsum_tt/samsum_tt_validation_data.jsonl  D*: the first 16 rows of the SAMSum validation split
  eval/samsum_tt/samsum_tt_test_data.jsonl        the SAMSum test split in order (GREATS's test perplexity: first 432 rows)
  Row strings: "<|start_header_id|>user<|end_header_id|>\\n\\n" + template.strip() + "<|eot_id|>" and
  "<|start_header_id|>assistant<|end_header_id|>\\n\\n" + output.strip() + "<|eot_id|>". torchtune also appends a masked EOS,
  which changes no loss or gradient and is left out. SAMSum comes from knkarthick/samsum, a copy of the original `samsum`
  dataset (Samsung/samsum is no longer on the Hub) with its split sizes (818 / 819) and row order (validation starts with
  13817023, test with 13862856).

    python SFT/data/prepare_greats_data.py --data_dir $SCRATCH_DIR/Dr.Post-Training/SFT/data [--less_zip less-data.zip]
"""
import argparse
import json
import os
import random
import shutil
import zipfile

LESS_POOL = ["flan_v2", "cot", "dolly", "oasst1"]
TYDIQA_TEMPLATES = {   # open-instruct eval/tydiqa/run_eval.py, as in GREATS's get_tydiqa_dataset
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
ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
SUMMARIZE_TEMPLATE = "Summarize this dialogue:\n{dialogue}\n---\nSummary:\n"
SAMSUM_N_VAL = 16


def llama3_turns(user, assistant):
    """torchtune Llama-3 tokenize_messages: header + content.strip() + <|eot_id|> per message."""
    return ("<|start_header_id|>user<|end_header_id|>\n\n" + user.strip() + "<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n" + assistant.strip() + "<|eot_id|>")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path}: {len(rows)} rows")


def tydiqa_rows(squad_json, tag):
    """GREATS get_tydiqa_dataset: random.seed(42); random.shuffle(data); one (prompt, ' answer') per row."""
    data = list(squad_json["data"])
    random.Random(42).shuffle(data)
    rows = []
    for ex in data:
        para = ex["paragraphs"][0]
        qa = para["qas"][0]
        lang = qa["id"].split("-")[0]
        instruction, p, q, a = TYDIQA_TEMPLATES[lang]
        user = instruction + p + " " + para["context"] + "\n" + q + " " + qa["question"] + "\n"
        rows.append({
            "dataset": "tydiqa_less", "id": qa["id"],
            "prompt": "<|user|>\n" + user + "<|assistant|>\n" + a,
            "completion": " " + qa["answers"][0]["text"],
            "context": para["context"], "question": qa["question"], "answers": qa["answers"],
            "metadata": {"language": lang, "source": tag},
        })
    return rows


def prepare_less(zip_path, data_dir):
    with zipfile.ZipFile(zip_path) as z:
        for name in LESS_POOL:
            dst = os.path.join(data_dir, "train", "less_official", f"{name}_data.jsonl")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(f"data/train/processed/{name}/{name}_data.jsonl") as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            print(f"  {dst}")
        dev = json.load(z.open("data/eval/tydiqa/dev/tydiqa-goldp-v1.1-dev.json"))
        test = json.load(z.open("data/eval/tydiqa/test/tydiqa-goldp-v1.1-dev.json"))
        oneshot = json.load(z.open("data/eval/tydiqa/dev/tydiqa-one-shot.json"))
    out = os.path.join(data_dir, "eval", "tydiqa_less")
    write_jsonl(os.path.join(out, "tydiqa_less_validation_data.jsonl"), tydiqa_rows(dev, "less dev/tydiqa-goldp-v1.1-dev.json"))
    write_jsonl(os.path.join(out, "tydiqa_less_test_data.jsonl"), tydiqa_rows(test, "less test/tydiqa-goldp-v1.1-dev.json"))
    with open(os.path.join(out, "tydiqa_less_fewshot.json"), "w", encoding="utf-8") as f:
        json.dump(oneshot, f, ensure_ascii=False, indent=1)


def prepare_alpaca(data_dir):
    from datasets import load_dataset
    rows = []
    for i, ex in enumerate(load_dataset("tatsu-lab/alpaca", split="train")):
        template = ALPACA_PROMPT_INPUT if ex["input"] else ALPACA_PROMPT_NO_INPUT
        prompt, completion = llama3_turns(template.format(instruction=ex["instruction"], input=ex["input"]), ex["output"])
        rows.append({"dataset": "alpaca_tt", "id": f"alpaca_{i}", "prompt": prompt, "completion": completion})
    write_jsonl(os.path.join(data_dir, "train", "alpaca_tt", "alpaca_tt_data.jsonl"), rows)


def prepare_samsum(data_dir):
    from datasets import load_dataset
    ds = load_dataset("knkarthick/samsum")
    assert (len(ds["validation"]), len(ds["test"])) == (818, 819), "not the original SAMSum splits"
    assert ds["validation"][0]["id"] == "13817023" and ds["test"][0]["id"] == "13862856", "not the original SAMSum row order"

    def rows(split):
        out = []
        for ex in split:
            prompt, completion = llama3_turns(SUMMARIZE_TEMPLATE.format(dialogue=ex["dialogue"]), ex["summary"])
            out.append({"dataset": "samsum_tt", "id": ex["id"], "prompt": prompt, "completion": completion,
                        "dialogue": ex["dialogue"], "summary": ex["summary"]})
        return out

    out = os.path.join(data_dir, "eval", "samsum_tt")
    write_jsonl(os.path.join(out, "samsum_tt_validation_data.jsonl"), rows(ds["validation"].select(range(SAMSUM_N_VAL))))
    write_jsonl(os.path.join(out, "samsum_tt_test_data.jsonl"), rows(ds["test"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--less_zip", default=None, help="less-data.zip of princeton-nlp/less_data (downloaded if omitted)")
    ap.add_argument("--parts", default="less,alpaca,samsum")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    if "less" in parts:
        zip_path = args.less_zip
        if zip_path is None:
            from huggingface_hub import hf_hub_download
            zip_path = hf_hub_download("princeton-nlp/less_data", "less-data.zip", repo_type="dataset")
        print("LESS pool + TyDiQA target")
        prepare_less(zip_path, args.data_dir)
    if "alpaca" in parts:
        print("Alpaca pool (torchtune Llama-3 format)")
        prepare_alpaca(args.data_dir)
    if "samsum" in parts:
        print("SAMSum target (torchtune Llama-3 format)")
        prepare_samsum(args.data_dir)


if __name__ == "__main__":
    main()
