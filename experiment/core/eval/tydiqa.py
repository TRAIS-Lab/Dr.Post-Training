import evaluate
metric = evaluate.load("squad")

from core.data.get_validation_dataset import get_tydiqa_dataset_df
from experiment.core.eval.utils import generate_completions


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

        # Strip leading/trailing whitespace from answer
        answer = answer.strip()
        output = output.strip()

        print(f"Example {i} ({lang}):")
        print(f"  Output: {output}")
        print(f"  Answer: {answer}")

        try:
            lang_predictions = [{"id": str(i), "prediction_text": output}]
            lang_references = [{"id": str(i), "answers": [{"text": answer, "answer_start": 0}]}]
            res = metric.compute(predictions=lang_predictions, references=lang_references)
            if isinstance(res["f1"], float):
                acc += res["f1"]
                length += 1
                print(f"  F1: {res['f1']:.4f}")
        except Exception as e:
            print(f'  Error in f1 computation: {e}')
            continue

    if length == 0:
        print("Warning: No valid F1 scores computed")
        return 0.0

    avg_f1 = acc / length
    print(f"\nAverage F1 Score: {avg_f1:.4f} ({length}/{len(test_dataset)} examples)")
    return avg_f1