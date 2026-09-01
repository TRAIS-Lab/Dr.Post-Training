import json
import logging
import os
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

# Default multiplier for max sequence length threshold (relative to avg train seq length)
DEFAULT_SEQ_LENGTH_MULTIPLIER = 1.2


def estimate_token_length(
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        completion: str,
    ) -> int:
    """
    Estimate the token length of a query-completion pair without truncation.

    Args:
        tokenizer: The tokenizer to use.
        query: The query/prompt string.
        completion: The completion/answer string.

    Returns:
        The number of tokens in the full sequence.
    """
    full_prompt = query + completion
    # Use encode without truncation to get the true length
    tokens = tokenizer.encode(full_prompt, add_special_tokens=True)
    return len(tokens)


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

    # Encode query without truncation to find the prompt/completion boundary
    prompt_input_ids = tokenizer.encode(query)
    # Encode full prompt and truncate to max_length
    full_tokens = tokenizer.encode(full_prompt, max_length=max_length, truncation=True)

    # Mask prompt tokens; cap at full sequence length so completion tokens
    # (if any survive truncation) get real labels
    prompt_len = min(len(prompt_input_ids), len(full_tokens))

    full_input_ids = torch.tensor(full_tokens)
    labels = torch.tensor(full_tokens)
    labels[:prompt_len] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def load_unified_jsonl(
        data_dir: str,
        task: str,
        split: str = "test",
        k: int = 5,
        subject: str = None,
        seed: Optional[int] = None
    ) -> List[dict]:
    """
    Load examples from unified JSONL format.

    File naming convention:
    - Validation: {data_dir}/eval/{task}/{task}_validation_data.jsonl
    - LR sweep: {data_dir}/eval/{task}/{task}_lr_data.jsonl
    - Test: {data_dir}/eval/{task}/{task}_test_data.jsonl

    Args:
        data_dir: Base data directory
        task: Task name (mmlu, bbh, tydiqa, gsm8k, math500, samsum)
        split: Which split to load ("validation", "test", or "lr")
        k: Number of examples to load
        subject: Optional subject filter (for MMLU and BBH with multiple subtasks)
        seed: Optional seed for shuffling examples before selecting first k

    Returns:
        List of example dictionaries with 'messages' field
    """
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
                example_subject = example.get('task') or example.get('subject')
                if example_subject != subject:
                    continue
            examples.append(example)
            if seed is None and len(examples) >= k:
                break

    if seed is not None:
        random.Random(seed).shuffle(examples)

    if k is not None and k > 0 and len(examples) < k:
        logger.warning(
            f"{task}/{split}: requested {k} examples but the file only has {len(examples)} "
            f"({file_path}); using all {len(examples)}."
        )

    return examples[:k]


def get_bbh_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        split: str = "test",
        k: int = 5,
        subject: str = None,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
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
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to load.
        subject: Optional BBH task name to filter by (e.g., "boolean_expressions").
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "bbh", split, load_k, subject, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

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

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"BBH: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        use_chat_format: bool = True,
        chat_format: str = "tulu",
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
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
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to load.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The TyDiQA dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "tydiqa", split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

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

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"TyDiQA: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_gsm8k_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """
    Get the GSM8K dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to use.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The GSM8K dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "gsm8k", split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"GSM8K: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_math500_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """
    Get the MATH500 dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to use.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The MATH500 dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "math500", split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"MATH500: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_samsum_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """
    Get the SamSUM dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to use.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The SamSUM dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "samsum", split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"SamSUM: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_truthfulqa_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """
    Get the TruthfulQA dataset in unified JSONL format.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to use.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The TruthfulQA dataset containing input_ids, attention_mask, and labels.
    """
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "truthfulqa", split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

        messages = example.get('messages', [])
        if len(messages) < 2:
            continue

        user_content = messages[0]['content']
        assistant_content = messages[1]['content']

        prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
        answer = assistant_content + tokenizer.eos_token

        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"TruthfulQA: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

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
        split: str = "test",
        k: int = 5,
        subject: str = None,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
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
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to load.
        subject: Optional MMLU subject to filter by (e.g., "sociology").
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.

    Returns:
        Dataset: The MMLU dataset containing input_ids, attention_mask, and labels.
    """
    # Load more examples than needed if rejection sampling is enabled
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "mmlu", split, load_k, subject, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0

    for i, example in enumerate(examples):
        if accepted_count >= k:
            break

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

        # Rejection sampling based on sequence length
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"MMLU: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")

    dataset = Dataset.from_dict(dataset)
    return dataset


# =============================================================================
# Main Interface
# =============================================================================

# =============================================================================
# Generic messages-format loader (Dolci targets: precise_if, math, mbpp, ...)
# =============================================================================

# Tasks whose validation/test files are plain single-turn ``messages`` JSONL and
# need no task-specific prompt formatting. Any other task whose files exist under
# ``eval/<task>/`` also falls back to this loader (see ``get_dataset``).
MESSAGES_FORMAT_TASKS = ("precise_if", "math", "mbpp")


def get_messages_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        task: str,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """
    Load any unified ``messages`` JSONL split and render it with the tokenizer's
    chat template (native template if the tokenizer has one, tulu fallback
    otherwise). Loss labels cover the assistant answer plus end-of-turn marker.

    The first user turn and the first assistant turn after it are used; an
    optional leading system turn is kept.

    Args:
        data_dir: The main data directory.
        tokenizer: The tokenizer used to tokenize the input text.
        max_length: The maximum length of the input sequence.
        task: Task name; files live at ``{data_dir}/eval/{task}/{task}_{split}_data.jsonl``.
        split: Which split to load ("validation", "test", or "lr").
        k: Number of examples to use.
        seed: Optional seed for shuffling examples before selection.
        max_seq_length_threshold: If provided, reject samples longer than this threshold.
    """
    from SFT.data.chat_format import render_prompt_and_answer

    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, task, split, load_k, seed=seed)

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0
    skipped_count = 0

    for example in examples:
        if accepted_count >= k:
            break

        messages = example.get("messages", [])
        system_content = None
        user_content = None
        assistant_content = None
        for message in messages:
            role = message.get("role")
            if role == "system" and user_content is None and system_content is None:
                system_content = message.get("content", "")
            elif role == "user" and user_content is None:
                user_content = message.get("content", "")
            elif role == "assistant" and user_content is not None:
                assistant_content = message.get("content", "")
                break
        if not user_content or not assistant_content:
            skipped_count += 1
            continue

        prompt, answer = render_prompt_and_answer(
            tokenizer, user_content, assistant_content, system_content=system_content
        )

        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1

    if rejected_count > 0:
        logger.info(f"{task}/{split}: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")
    if skipped_count > 0:
        logger.info(f"{task}/{split}: Skipped {skipped_count} rows without a user/assistant pair")
    if accepted_count < k:
        logger.warning(
            f"{task}/{split}: requested {k} examples but only {accepted_count} were usable "
            f"(after length rejection and format filtering)."
        )

    return Dataset.from_dict(dataset)


def get_dataset(task: str, **kwargs) -> Dataset:
    """
    Get the dataset for the given task.

    Args:
        task: The name of the task (bbh, tydiqa, mmlu, samsum, gsm8k, math500).
        **kwargs: Additional arguments passed to the task-specific function.
            Common kwargs:
            - data_dir: Base data directory
            - tokenizer: Tokenizer for encoding
            - max_length: Maximum sequence length
            - split: Which split to load ("validation", "test", or "lr")
            - k: Number of examples to load
            - seed: Optional seed for shuffling examples before selection
            - max_seq_length_threshold: If provided, reject samples longer than
              this threshold (for rejection sampling based on train avg length)

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
        "truthfulqa": get_truthfulqa_dataset,
    }
    for _messages_task in MESSAGES_FORMAT_TASKS:
        task_functions.setdefault(_messages_task, get_messages_dataset)

    if task not in task_functions:
        # Unknown task: accept it if its unified JSONL exists, using the generic
        # messages-format loader. This is how new Dolci-style targets are added
        # without touching code.
        data_dir = kwargs.get("data_dir")
        split = kwargs.get("split", "test")
        candidate = os.path.join(str(data_dir), "eval", task, f"{task}_{split}_data.jsonl") if data_dir else None
        if candidate and os.path.exists(candidate):
            logger.info(f"Task {task!r} is not registered; using the generic messages-format loader ({candidate})")
            return get_messages_dataset(task=task, **kwargs)
        raise ValueError(
            f"Invalid task name: {task}. Valid tasks: {list(task_functions.keys())}. "
            f"Unregistered tasks are accepted if {task}/{task}_{split}_data.jsonl exists under {data_dir}/eval/."
        )

    if task_functions[task] is get_messages_dataset:
        return get_messages_dataset(task=task, **kwargs)
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
