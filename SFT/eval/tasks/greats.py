"""Downstream metrics of the GREATS-grounded settings (Wang et al., "GREATS: Online Selection of High-Quality Data for LLM
Training in Every Iteration", NeurIPS 2024), on the target files written by SFT/data/prepare_greats_data.py.

tydiqa_less: 1-shot TyDiQA-GoldP F1 with the LESS / open-instruct protocol that GREATS App. B says it follows: GREATS's 500
  test rows (LESS's TyDiQA-GoldP dev file shuffled with random.seed(42), the order of tydiqa_less_test_data.jsonl), the
  language's one-shot demonstration from LESS's tydiqa-one-shot.json, open-instruct's tulu prompt ending in
  "<|assistant|>\\nAnswer:", contexts capped at 512 tokens (LESS's evaluation script), greedy decoding of at most 50 new
  tokens stopped at a newline, SQuAD-v1.1 F1 / EM per language and their macro average. GREATS's own in-training F1 code builds
  a malformed prompt (instruction and tulu markers repeated), so the stated protocol is followed instead of that code.
samsum_tt: ROUGE of greedy summaries (128 new tokens, stop at <|eot_id|>) for GREATS's torchtune Llama-3 prompt
  (<|start_header_id|>user<|end_header_id|> "Summarize this dialogue:" ... <|eot_id|> then the assistant header). GREATS reports
  only SAMSum perplexity; ROUGE is this harness's downstream number.
Prompts are tokenized as in training: the BOS id is prepended when the run was trained with add_bos (generate_completions).
"""
import json
import os
import re
import string
from collections import Counter, defaultdict

from SFT.data.get_val_dataset import load_unified_jsonl
from ..utils import dump_generations, generate_completions, get_eos_token_ids

# open-instruct eval/tydiqa/run_eval.py (the templates of the LESS / GREATS target rows)
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
MAX_CONTEXT_TOKENS = 512
SAMSUM_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


# SQuAD v1.1 scoring (the `squad` metric open-instruct uses): lowercase, drop ASCII punctuation and English articles
def _squad_normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def _squad_f1(prediction, truth):
    p, t = _squad_normalize(prediction).split(), _squad_normalize(truth).split()
    common = sum((Counter(p) & Counter(t)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(t)
    return 2 * precision * recall / (precision + recall)


def _cap_context(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= MAX_CONTEXT_TOKENS else tokenizer.decode(ids[:MAX_CONTEXT_TOKENS])


def tydiqa_less_prompt(tokenizer, row, demo):
    """open-instruct's 1-shot tulu prompt (create_prompt_with_tulu_chat_format keeps the user content's trailing newline)."""
    lang = row["metadata"]["language"]
    instruction, p, q, a = TYDIQA_TEMPLATES[lang]
    prompt = instruction + "\n\n"
    prompt += (p + " " + _cap_context(tokenizer, demo["context"]) + "\n" + q + " " + demo["question"] + "\n" + a + " "
               + demo["answers"][0]["text"] + "\n\n")
    prompt += p + " " + _cap_context(tokenizer, row["context"]) + "\n" + q + " " + row["question"] + "\n"
    return "<|user|>\n" + prompt + "\n" + "<|assistant|>\n" + a


def compute_tydiqa_less(args, model, tokenizer, batch_size=16):
    n_test = args.n_test if args.n_test and args.n_test > 0 else 10 ** 9
    rows = load_unified_jsonl(args.data_dir, "tydiqa_less", "test", n_test)
    with open(os.path.join(args.data_dir, "eval", "tydiqa_less", "tydiqa_less_fewshot.json"), encoding="utf-8") as f:
        demos = json.load(f)
    prompts = [tydiqa_less_prompt(tokenizer, r, demos[r["metadata"]["language"]][0]) for r in rows]
    newline = tokenizer.encode("\n", add_special_tokens=False)[-1]   # open-instruct: last id, the tokenizer may add a space piece
    raw = generate_completions(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=50, do_sample=False,
                               stop_id_sequences=[[newline]], eos_token_id=get_eos_token_ids(tokenizer))
    outputs = [g.strip() for g in raw]

    per_lang = defaultdict(lambda: {"f1": [], "em": []})
    dumped = []
    for r, out in zip(rows, outputs):
        truths = [ans["text"] for ans in r["answers"]]
        f1 = 100 * max(_squad_f1(out, t) for t in truths)
        em = 100 * max(float(_squad_normalize(out) == _squad_normalize(t)) for t in truths)
        lang = r["metadata"]["language"]
        per_lang[lang]["f1"].append(f1)
        per_lang[lang]["em"].append(em)
        dumped.append({"id": r["id"], "language": lang, "prediction": out, "references": truths, "f1": f1})
    dump_generations(args, "tydiqa_less", dumped, tokenizer=tokenizer)

    langs = sorted(per_lang)
    per_language = {lang: {"f1": sum(v["f1"]) / len(v["f1"]), "exact_match": sum(v["em"]) / len(v["em"]), "n": len(v["f1"])}
                    for lang, v in ((lang, per_lang[lang]) for lang in langs)}
    all_f1 = [x for lang in langs for x in per_lang[lang]["f1"]]
    return {
        "f1_score": sum(d["f1"] for d in per_language.values()) / len(langs),            # macro over languages (open-instruct)
        "exact_match": sum(d["exact_match"] for d in per_language.values()) / len(langs),
        "f1_micro": sum(all_f1) / len(all_f1),
        "per_language": per_language,
        "n_test": len(rows),
        "protocol": "LESS/open-instruct 1-shot, tulu prompt, 512-token contexts, greedy <=50 tokens stop at newline, SQuAD F1",
    }


def compute_samsum_tt(args, model, tokenizer, batch_size=16, max_new_tokens=128):
    import evaluate
    n_test = args.n_test if args.n_test and args.n_test > 0 else 10 ** 9
    rows = load_unified_jsonl(args.data_dir, "samsum_tt", "test", n_test)
    prompts = [r["prompt"] + SAMSUM_ASSISTANT_HEADER for r in rows]
    raw = generate_completions(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=False,
                               eos_token_id=get_eos_token_ids(tokenizer))
    predictions = [g.strip() for g in raw]
    references = [r["summary"] for r in rows]
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    per = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)["rougeL"]
    dump_generations(args, "samsum_tt", [{"id": r["id"], "prediction": p, "raw": g, "references": [ref], "rougeL": 100 * s}
                                         for r, p, g, ref, s in zip(rows, predictions, raw, references, per)], tokenizer=tokenizer)
    return {
        "rouge1": scores["rouge1"], "rouge2": scores["rouge2"], "rougeL": scores["rougeL"],
        "rougeLsum": scores.get("rougeLsum", scores["rougeL"]),
        "n_test": len(rows),
        "protocol": "torchtune Llama-3 Summarize prompt, greedy <=128 tokens stop at <|eot_id|>, ROUGE (stemmed)",
    }
