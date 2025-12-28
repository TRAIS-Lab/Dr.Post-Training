import json
import os
from typing import List, Tuple

import torch
from torch import Tensor
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        completion: str,
        max_length: int,
        print_ex: bool = False
    ) -> Tuple[Tensor, Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def load_unified_jsonl(
        data_dir: str,
        task: str,
        validation: bool,
        k: int = 5,
        subject: str = None
    ) -> List[dict]:
    """
    Load examples from unified JSONL format.

    File naming convention:
    - Validation: {data_dir}/eval/{task}/{task}_validation_data.jsonl
    - Test: {data_dir}/eval/{task}/{task}_test_data.jsonl

    Args:
        data_dir: Base data directory
        task: Task name (mmlu, bbh, tydiqa, gsm8k, math500, samsum)
        validation: If True, load validation split; otherwise load test split
        k: Number of examples to load
        subject: Optional subject filter (for MMLU and BBH with multiple subtasks)

    Returns:
        List of example dictionaries with 'messages' field
    """
    split = "validation" if validation else "test"
    file_path = os.path.join(data_dir, "eval", task, f"{task}_{split}_data.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            f"Please run: python SFT/data/prepare_datasets.py --datasets {task}"
        )

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            # Filter by subject/task if specified (for BBH and MMLU)
            if subject is not None:
                # Check 'task' field (BBH) or 'subject' field (MMLU)
                example_subject = example.get('task') or example.get('subject')
                if example_subject != subject:
                    continue
            examples.append(example)
            if len(examples) >= k:
                break

    return examples


def get_bbh_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        validation: bool = False,
        k: int = 5,
        subject: str = None,
        **kwargs
    ) -> Dataset:
    """
    Get the BBH dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        use_chat_format: Whether to use chat format for the input.
        chat_format: The chat format to use ("tulu" or "llama2").
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to load.
        subject: Optional BBH task name to filter by (e.g., "boolean_expressions").

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "bbh", validation, k, subject)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        # Format the prompt
        if use_chat_format:
            if chat_format == "tulu":
                prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
            else:
                prompt = f"<s> {B_INST} {user_content} {E_INST} "
        else:
            prompt = f"{user_content}\nAnswer: "

        answer = assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        validation: bool = False,
        k: int = 5,
        **kwargs
    ) -> Dataset:
    """
    Get the TyDiQA dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        use_chat_format: Whether to use chat format for the input.
        chat_format: The chat format to use.
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to load.

    Returns:
        Dataset: The TyDiQA dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "tydiqa", validation, k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        # Format the prompt
        if use_chat_format:
            if chat_format == "tulu":
                prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
            else:
                prompt = f"<s> {B_INST} {user_content} {E_INST} "
        else:
            prompt = f"{user_content}\nAnswer: "

        answer = assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_gsm8k_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        validation: bool = False,
        k: int = 5,
        **kwargs
    ) -> Dataset:
    """
    Get the GSM8K dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to use.

    Returns:
        Dataset: The GSM8K dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "gsm8k", validation, k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_math500_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        validation: bool = False,
        k: int = 5,
        **kwargs
    ) -> Dataset:
    """
    Get the MATH500 dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to use.

    Returns:
        Dataset: The MATH500 dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "math500", validation, k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_samsum_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        validation: bool = False,
        k: int = 5,
        **kwargs
    ) -> Dataset:
    """
    Get the SamSUM dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to use.

    Returns:
        Dataset: The SamSUM dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "samsum", validation, k)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


# =============================================================================
# MMLU Dataset (uses unified JSONL format)
# =============================================================================

def get_mmlu_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        validation: bool = False,
        k: int = 5,
        subject: str = None,
        **kwargs
    ) -> Dataset:
    """
    Get the MMLU dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        use_chat_format: Whether to use chat format for the prompts.
        chat_format: The chat format to use.
        validation: If True, load validation split; otherwise load test split.
        k: Number of examples to load.
        subject: Optional MMLU subject to filter by (e.g., "sociology").

    Returns:
        Dataset: The MMLU dataset containing input_ids, attention_mask, and labels.
    """
    examples = load_unified_jsonl(data_dir, "mmlu", validation, k, subject)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, example in enumerate(examples):
        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        # Format the prompt
        if use_chat_format:
            if chat_format == "tulu":
                prompt = f"<|user|>\n{user_content}\n<|assistant|>\nThe answer is:"
            else:
                prompt = f"<s> {B_INST} {user_content} {E_INST} The answer is:"
        else:
            prompt = f"{user_content} The answer is:"

        answer = " " + assistant_content + tokenizer.eos_token

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if i == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


# =============================================================================
# Main Interface
# =============================================================================

def get_dataset(task: str, **kwargs) -> Dataset:
    """
    Get the dataset for the given task.

    Args:
        task: The name of the task (bbh, tydiqa, mmlu, samsum, gsm8k, math500).
        **kwargs: Additional arguments passed to the task-specific function.

    Returns:
        Dataset: The dataset for the task.

    Raises:
        ValueError: If the task name is not valid.
    """
    task_functions = {
        "bbh": get_bbh_dataset,
        "tydiqa": get_tydiqa_dataset,
        "mmlu": get_mmlu_dataset,
        "samsum": get_samsum_dataset,
        "gsm8k": get_gsm8k_dataset,
        "math500": get_math500_dataset,
    }

    if task not in task_functions:
        raise ValueError(f"Invalid task name: {task}. Valid tasks: {list(task_functions.keys())}")

    return task_functions[task](**kwargs)


def get_dataloader(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, batch_size: int = 1) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset: The dataset to create a DataLoader for.
        tokenizer: The tokenizer to use for padding.
        batch_size: The batch size.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="longest"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    print(f"There are {len(dataset)} examples in the dataset")
    return dataloader


