"""
MMLU evaluation module for multiple-choice question answering.

Uses accuracy as the primary metric - measures next token prediction
for answer choices (A, B, C, D).
"""

import torch
import pandas as pd
from tqdm import tqdm

from SFT.data.get_val_dataset import load_unified_jsonl
from ..utils import get_next_word_predictions, create_prompt_with_tulu_chat_format

# Answer choices for MMLU
CHOICES = ["A", "B", "C", "D"]


def get_mmlu_dataset_df(data_dir: str, split: str = "test", k: int = 5, subject: str = None):
    """
    Get MMLU dataset as a pandas DataFrame for evaluation.

    This function returns data in the format expected by the MMLU evaluation:
    - Columns: [question, A, B, C, D, answer]

    Args:
        data_dir: The main data directory.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to load.
        subject: Optional MMLU subject to filter by.

    Returns:
        pandas DataFrame with columns [question, A, B, C, D, answer]
    """
    examples = load_unified_jsonl(data_dir, "mmlu", split, k, subject)

    rows = []
    for example in examples:
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        answer = messages[1]['content']  # Single letter: A, B, C, or D

        # Parse the question and choices from the formatted prompt
        # Format: "The following is a multiple choice question...\n\n{question}\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n\nAnswer:"
        lines = user_content.split('\n')

        # Find question and choices
        question = ""
        choices = {"A": "", "B": "", "C": "", "D": ""}
        in_question = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for choice lines
            if line.startswith("A. "):
                choices["A"] = line[3:]
            elif line.startswith("B. "):
                choices["B"] = line[3:]
            elif line.startswith("C. "):
                choices["C"] = line[3:]
            elif line.startswith("D. "):
                choices["D"] = line[3:]
            elif line.startswith("Answer:"):
                continue
            elif "multiple choice question" in line.lower():
                in_question = True
            elif in_question and not line.startswith(("A.", "B.", "C.", "D.")):
                if question:
                    question += " " + line
                else:
                    question = line

        rows.append({
            "question": question,
            "A": choices["A"],
            "B": choices["B"],
            "C": choices["C"],
            "D": choices["D"],
            "answer": answer
        })

    df = pd.DataFrame(rows)
    return df

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(CHOICES[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model_generate_ICL_prompts(args, model, tokenizer, dev_df, test_df, batch_size=1):

    subject = args.subject

    args.use_chat_format = True
    k = args.n_val

    prompts = []

    chat_formatting_function = create_prompt_with_tulu_chat_format

    for i in range(0, test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids

        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 512:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    return prompts


@torch.no_grad()
def compute_accuracy(args, model, tokenizer, batch_size=1):
    """
    Compute accuracy for MMLU evaluation.

    Args:
        args: Arguments containing data_dir, n_test (or n_eval), n_val, subject
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        batch_size: Batch size for evaluation

    Returns:
        dict: Dictionary with accuracy, n_test
    """
    # Get data_dir from args, default to ./data
    data_dir = getattr(args, 'data_dir', './data')
    n_val = getattr(args, 'n_val', 5)  # Number of few-shot examples
    n_test = getattr(args, 'n_test', -1)
    if n_test <= 0:
        n_test = 10000  # Load all available
    subject = getattr(args, 'subject', None)

    # For compatibility with args that use n_eval instead of n_test
    n_eval = getattr(args, 'n_eval', n_test)

    # Store original args values for ICL prompt generation
    args.n_val = n_val
    args.n_eval = n_eval
    if subject is None:
        args.subject = "general"  # Default subject name for prompts
    else:
        args.subject = subject

    # Load validation set for few-shot examples
    dev_df = get_mmlu_dataset_df(
        data_dir=data_dir,
        split="validation",
        k=n_val,
        subject=subject
    )

    # Load test set for evaluation
    eval_df = get_mmlu_dataset_df(
        data_dir=data_dir,
        split="test",
        k=n_eval,
        subject=subject
    )

    print(f"Loaded {len(eval_df)} MMLU test examples (with {len(dev_df)} few-shot examples)")

    # Generate ICL prompts
    prompts = eval_hf_model_generate_ICL_prompts(args, model, tokenizer, dev_df, eval_df, batch_size=1)

    # Get answer choice token IDs
    # Adding a prefix space as expected from the prompt format
    answer_choice_ids = [
        tokenizer.encode(" " + choice, add_special_tokens=False)[-1]
        for choice in CHOICES
    ]

    # Get predictions
    pred_indices, _ = get_next_word_predictions(
        model, tokenizer, prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=batch_size
    )

    # Compute accuracy
    correct = 0
    ground_truths = eval_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = CHOICES[pred_indices[i]]
        ground_truth = ground_truths[i]
        if prediction == ground_truth:
            correct += 1

    accuracy = correct / len(pred_indices) if len(pred_indices) > 0 else 0.0

    print(f"\nMMLU Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Examples evaluated: {len(pred_indices)}")

    return {
        "accuracy": accuracy,
        "n_test": len(pred_indices),
        "n_correct": correct,
    }