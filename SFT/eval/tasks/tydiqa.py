import json
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from SFT.data.get_val_dataset import load_unified_jsonl, render_chat
from ..utils import generate_completions, get_eos_token_ids


# TyDiQA-style scoring (replaces evaluate.load("squad")):
# - Lowercase (Unicode-aware) and strip Unicode punctuation (any category P*)
# - Whitespace-tokenize (matches the official TyDiQA-GoldP convention)
# - Do NOT strip English articles a/an/the — that bias is harmful for the 8
#   non-English TyDiQA languages (arabic, bengali, finnish, indonesian, korean,
#   russian, swahili, telugu) and was the main concern with the prior SQuAD scorer.
def _normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return " ".join(s.split())


def _get_tokens(s: str) -> List[str]:
    return _normalize_answer(s).split() if s else []


def _f1(prediction: str, ground_truth: str) -> float:
    pt, gt = _get_tokens(prediction), _get_tokens(ground_truth)
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pt), n / len(gt)
    return 2 * p * r / (p + r)


def _em(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))

# Language-specific templates for TyDiQA
# Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
ENCODING_TEMPLATES_WITH_CONTEXT = {
    "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
    "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
    "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
    "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
    "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
    "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
    "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
    "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
    "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
}


def load_oneshot_examples(data_dir: str) -> Dict:
    """Load one-shot examples from the fewshot JSON file."""
    # Try different possible file names
    possible_files = [
        os.path.join(data_dir, "eval", "tydiqa", "tydiqa_fewshot.json"),
        os.path.join(data_dir, "eval", "tydiqa", "tydiqa-one-shot.json"),
    ]

    for fewshot_file in possible_files:
        if os.path.exists(fewshot_file):
            with open(fewshot_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    print(f"Warning: No one-shot file found. Tried: {possible_files}")
    return {}


def format_oneshot_prompt(example: Dict, language: str) -> str:
    """Format a one-shot example into a prompt string with answer."""
    templates = ENCODING_TEMPLATES_WITH_CONTEXT.get(language, ENCODING_TEMPLATES_WITH_CONTEXT["english"])
    task_prompt, p_template, q_template, a_template = templates

    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", [])
    answer = answers[0]["text"] if answers else ""

    prompt = f"{task_prompt}\n{p_template} {context}\n{q_template} {question}\n{a_template} {answer}\n\n"
    return prompt


def get_tydiqa_dataset_df(
        tokenizer,
        data_dir: str,
        split: str = "test",
        k: int = 100,
        oneshot_examples: Dict = None
    ) -> List[Tuple[str, str, str]]:
    """Get TyDiQA dataset as a list of (prompt, answer, language) tuples.

    Prompts are rendered with the tokenizer's native chat template; if a
    one-shot example is available for the question's language, it is
    prepended to the user content (so it sits inside the user turn, before
    the question being asked).
    """
    examples = load_unified_jsonl(data_dir, "tydiqa", split, k)

    results = []
    for example in examples:
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        answer = messages[1]['content']
        language = example.get('metadata', {}).get('language', 'unknown')

        if oneshot_examples:
            lang_examples = oneshot_examples.get(language) or oneshot_examples.get("english", [])
            if lang_examples:
                oneshot_prompt = format_oneshot_prompt(lang_examples[0], language)
                full_content = oneshot_prompt + user_content
            else:
                full_content = user_content
        else:
            full_content = user_content

        prompt = render_chat(tokenizer, full_content)
        results.append((prompt, answer, language))

    return results

def compute_accuracy(args, model, tokenizer):
    """
    Compute F1 and exact match scores for TyDiQA evaluation (1-shot).

    Args:
        args: Arguments containing n_test and data_dir
        model: The model to evaluate
        tokenizer: The tokenizer for the model

    Returns:
        dict: Dictionary with f1_score, exact_match, and n_test
    """
    # Get data_dir from args, default to ./data
    data_dir = getattr(args, 'data_dir', './data')
    n_test = getattr(args, 'n_test', 100)
    if n_test <= 0:
        n_test = 10000  # Load all available

    # Load one-shot examples
    oneshot_examples = load_oneshot_examples(data_dir)
    if oneshot_examples:
        print(f"Loaded one-shot examples for {len(oneshot_examples)} languages")
    else:
        print("Warning: No one-shot examples found, using 0-shot evaluation")

    test_dataset = get_tydiqa_dataset_df(
        tokenizer=tokenizer,
        data_dir=data_dir,
        split="test",
        k=n_test,
        oneshot_examples=oneshot_examples
    )

    print(f'Loaded {len(test_dataset)} test examples (1-shot evaluation)')

    prompts = [prompt for prompt, answer, lang in test_dataset]

    print("Generating completions...")
    generations = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=1,
        max_new_tokens=50,
        eos_token_id=get_eos_token_ids(tokenizer),
    )

    # 2K-char cap is well above any legitimate TyDiQA short-form answer and
    # avoids feeding pathological generations into the scorer.
    MAX_PRED_CHARS = 2048
    outputs = [g.strip().replace("\n", "")[:MAX_PRED_CHARS] for g in generations]

    by_lang: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"f1": [], "em": []})

    for i, (_, answer, lang) in enumerate(test_dataset):
        output = outputs[i]
        for stop in ("<|im_end|>", "<|im_start|>"):
            if stop in output:
                output = output[:output.find(stop)]

        clean_answer = answer.strip()[:MAX_PRED_CHARS]
        clean_output = output.strip()[:MAX_PRED_CHARS]

        by_lang[lang]["f1"].append(_f1(clean_output, clean_answer) * 100.0)
        by_lang[lang]["em"].append(_em(clean_output, clean_answer) * 100.0)

    if not by_lang:
        print("Warning: No valid scores computed")
        return {"f1_score": 0.0, "exact_match": 0.0, "n_test": len(test_dataset)}

    per_language = {
        lang: {
            "f1": float(sum(d["f1"]) / len(d["f1"])),
            "em": float(sum(d["em"]) / len(d["em"])),
            "n":  len(d["f1"]),
        }
        for lang, d in by_lang.items()
    }
    macro_f1 = sum(p["f1"] for p in per_language.values()) / len(per_language)
    macro_em = sum(p["em"] for p in per_language.values()) / len(per_language)
    all_f1 = [v for d in by_lang.values() for v in d["f1"]]
    all_em = [v for d in by_lang.values() for v in d["em"]]
    micro_f1 = sum(all_f1) / len(all_f1)
    micro_em = sum(all_em) / len(all_em)

    print(f"\nTyDiQA Results (TyDiQA-style scoring, {len(per_language)} languages):")
    print(f"  Macro F1 (avg across languages): {macro_f1:.4f}")
    print(f"  Macro EM (avg across languages): {macro_em:.4f}")
    print(f"  Micro F1 (avg across examples):  {micro_f1:.4f}")
    print(f"  Micro EM (avg across examples):  {micro_em:.4f}")
    print(f"  Examples evaluated: {len(all_f1)}/{len(test_dataset)}")
    print(f"  Per-language:")
    for lang in sorted(per_language):
        p = per_language[lang]
        print(f"    {lang:12s} n={p['n']:3d}  F1={p['f1']:6.2f}  EM={p['em']:6.2f}")

    # Keep f1_score / exact_match as micro-avg for backward compatibility with
    # existing notebooks; expose macro and per-language as additional fields.
    return {
        "f1_score": micro_f1,
        "exact_match": micro_em,
        "f1_macro": macro_f1,
        "exact_match_macro": macro_em,
        "per_language": per_language,
        "n_test": len(test_dataset),
    }