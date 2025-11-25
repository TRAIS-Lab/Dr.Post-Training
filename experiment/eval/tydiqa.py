from typing import List, Tuple

import evaluate
metric = evaluate.load("squad")

from experiment.data.get_val_dataset import load_unified_jsonl
from utils import generate_completions

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def get_tydiqa_dataset_df(
        data_dir: str,
        validation: bool = False,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        k: int = 100
    ) -> List[Tuple[str, str, str]]:
    """
    Get TyDiQA dataset as a list of tuples for evaluation.

    This function returns data in the format expected by the TyDiQA evaluation:
    - List of (formatted_prompt, answer, language) tuples

    Args:
        data_dir: The main data directory.
        validation: If True, load validation split; otherwise load test split.
        use_chat_format: Whether to format prompts with chat template.
        chat_format: The chat format to use ("tulu" or "llama2").
        k: Number of examples to load.

    Returns:
        List of (formatted_prompt, answer, language) tuples
    """
    examples = load_unified_jsonl(data_dir, "tydiqa", validation, k)

    results = []
    for example in examples:
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        answer = messages[1]['content']
        language = example.get('metadata', {}).get('language', 'unknown')

        # Format the prompt
        if use_chat_format:
            if chat_format == "tulu":
                prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
            else:
                prompt = f"<s> {B_INST} {user_content} {E_INST} "
        else:
            prompt = f"{user_content}\nAnswer: "

        results.append((prompt, answer, language))

    return results

def compute_accuracy(args, model, tokenizer):
    """
    Compute F1 score for TyDiQA evaluation.

    Args:
        args: Arguments containing n_test
        model: The model to evaluate
        tokenizer: The tokenizer for the model

    Returns:
        float: Average F1 score across all examples
    """
    # Load test dataset
    # Note: get_tydiqa_dataset_df returns list of tuples: (formatted_prompt, answer, lang)
    # where formatted_prompt already contains the full prompt with context and question
    test_dataset = get_tydiqa_dataset_df(
        data_dir="./data",
        validation=False,
        use_chat_format=True,  # This controls the format returned by get_tydiqa_dataset_df
        chat_format="tulu",
        k=args.n_test
    )

    print(f'Loaded {len(test_dataset)} test examples')

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

    # Compute F1 scores
    acc, length = 0, 0
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
                acc += res["f1"]
                length += 1
        except Exception as e:
            # Skip examples where F1 computation fails
            continue

    if length == 0:
        print("Warning: No valid F1 scores computed")
        return 0.0

    avg_f1 = acc / length
    print(f"\nAverage F1 Score: {avg_f1:.4f} ({length}/{len(test_dataset)} examples)")
    return avg_f1