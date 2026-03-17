import json
import os
from typing import Dict, List, Tuple

import evaluate
metric = evaluate.load("squad")

from SFT.data.get_val_dataset import load_unified_jsonl
from ..utils import generate_completions

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

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
        data_dir: str,
        split: str = "test",
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        k: int = 100,
        oneshot_examples: Dict = None
    ) -> List[Tuple[str, str, str]]:
    """
    Get TyDiQA dataset as a list of tuples for evaluation.

    This function returns data in the format expected by the TyDiQA evaluation:
    - List of (formatted_prompt, answer, language) tuples

    Args:
        data_dir: The main data directory.
        split: Which split to load ("validation", "test", or "lr").
        use_chat_format: Whether to format prompts with chat template.
        chat_format: The chat format to use ("tulu" or "llama2").
        k: Number of examples to load.
        oneshot_examples: Dictionary of one-shot examples by language.

    Returns:
        List of (formatted_prompt, answer, language) tuples
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

        # Build prompt with one-shot example if available
        if oneshot_examples:
            # Get one-shot example for this language, fallback to English
            lang_examples = oneshot_examples.get(language) or oneshot_examples.get("english", [])
            if lang_examples:
                oneshot_prompt = format_oneshot_prompt(lang_examples[0], language)
                # Prepend one-shot example to the user content
                full_content = oneshot_prompt + user_content
            else:
                full_content = user_content
        else:
            full_content = user_content

        # Format the prompt with chat template
        if use_chat_format:
            if chat_format == "tulu":
                prompt = f"<|user|>\n{full_content}\n<|assistant|>\n"
            else:
                prompt = f"<s> {B_INST} {full_content} {E_INST} "
        else:
            prompt = f"{full_content}\nAnswer: "

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

    # Load test dataset with one-shot prompting
    # Note: get_tydiqa_dataset_df returns list of tuples: (formatted_prompt, answer, lang)
    # where formatted_prompt already contains the full prompt with context and question
    test_dataset = get_tydiqa_dataset_df(
        data_dir=data_dir,
        split="test",
        use_chat_format=True,  # This controls the format returned by get_tydiqa_dataset_df
        chat_format="tulu",
        k=n_test,
        oneshot_examples=oneshot_examples
    )

    print(f'Loaded {len(test_dataset)} test examples (1-shot evaluation)')

    # Extract prompts from the dataset (they're already formatted)
    prompts = [prompt for prompt, answer, lang in test_dataset]

    # Generate completions
    print("Generating completions...")
    generations = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=1,
        max_new_tokens=50
    )

    # Clean outputs
    outputs = [g.strip().replace("\n", "") for g in generations]

    # Compute F1 and exact match scores
    total_f1, total_em, valid_count = 0, 0, 0
    for i, (prompt, answer, lang) in enumerate(test_dataset):
        # Clean up the output by removing any special tokens
        if "<|user|>" in outputs[i]:
            index = outputs[i].find("<|user|>")
            output = outputs[i][:index]
        else:
            output = outputs[i]

        try:
            # Strip whitespace from both answer and output for fair comparison
            clean_answer = answer.strip()
            clean_output = output.strip()
            lang_predictions = [{"id": str(i), "prediction_text": clean_output}]
            # SQuAD metric expects answers in format: {'text': [list], 'answer_start': [list]}
            lang_references = [{"id": str(i), "answers": {"text": [clean_answer], "answer_start": [0]}}]
            res = metric.compute(predictions=lang_predictions, references=lang_references)
            if isinstance(res["f1"], float):
                total_f1 += res["f1"]
                total_em += res["exact_match"]
                valid_count += 1
        except Exception as e:
            # Skip examples where computation fails
            continue

    if valid_count == 0:
        print("Warning: No valid scores computed")
        return {"f1_score": 0.0, "exact_match": 0.0, "n_test": len(test_dataset)}

    avg_f1 = total_f1 / valid_count
    avg_em = total_em / valid_count
    print(f"\nTyDiQA Results:")
    print(f"  F1 Score: {avg_f1:.4f}")
    print(f"  Exact Match: {avg_em:.4f}")
    print(f"  Examples evaluated: {valid_count}/{len(test_dataset)}")

    return {
        "f1_score": avg_f1,
        "exact_match": avg_em,
        "n_test": len(test_dataset)
    }