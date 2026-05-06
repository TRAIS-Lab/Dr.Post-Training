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

# Default multiplier for max sequence length threshold (relative to avg train seq length)
DEFAULT_SEQ_LENGTH_MULTIPLIER = 1.2


# Open-instruct/Tulu-style fallback chat template — used when the tokenizer
# doesn't ship its own (e.g., base models like meta-llama/Llama-3.2-1B that
# weren't fine-tuned on chat data). The role markers (`<|user|>`, `<|assistant|>`)
# are plaintext (not special tokens), tokenized by BPE; that's the same convention
# the open-instruct / LESS / Tulu codebases use.
_DEFAULT_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'].strip() }}\n"
    "{%- elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'].strip() }}\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'].strip() }}{{ eos_token }}\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def ensure_chat_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Set a default open-instruct chat template if the tokenizer doesn't have one.

    Idempotent — safe to call multiple times. Lets `apply_chat_template` work
    uniformly across base models (no native template, e.g. Llama-3.2-1B-Base)
    and instruct/chat models (Llama-3.2-1B-Instruct, Qwen3 chat models, ...).
    """
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE
    return tokenizer


def render_chat(
        tokenizer: PreTrainedTokenizerBase,
        user_content: str,
        assistant_content: Optional[str] = None,
    ):
    """Render a single (user[, assistant]) turn with the tokenizer's chat template.

    Returns the prompt string when `assistant_content is None`, otherwise
    `(prompt, answer)` such that `prompt + answer == full_render`. Falls back
    to the open-instruct template via `ensure_chat_template` when the tokenizer
    doesn't have one. For Qwen3, `enable_thinking=False` is passed so the
    prompt includes the auto-injected empty `<think></think>` block; for
    other tokenizers the kwarg is silently ignored.
    """
    ensure_chat_template(tokenizer)
    user_msg = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        user_msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if assistant_content is None:
        return prompt
    full = tokenizer.apply_chat_template(
        user_msg + [{"role": "assistant", "content": assistant_content}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "Chat template inconsistency: prompt is not a prefix of full render.\n"
            f"  prompt: {prompt!r}\n  full: {full!r}"
        )
    return prompt, full[len(prompt):]


def estimate_token_length(
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        completion: str,
    ) -> int:
    """Estimate the token length of a chat-template-rendered (query, completion).

    Uses `add_special_tokens=False` because the chat template already emits
    its own special tokens (`<|im_start|>` etc.), so we don't want any
    automatic BOS/EOS injection on top.
    """
    full_prompt = query + completion
    tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    return len(tokens)


def tokenize(
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        completion: str,
        max_length: int,
        print_ex: bool = False
    ) -> Tuple[Tensor, Tensor, List[int]]:
    """Tokenize a chat-template-rendered (query, completion) pair.

    Both pieces are expected to come from `render_chat(...)` (or equivalent),
    which means they already contain the model's native special tokens.
    Tokenization uses `add_special_tokens=False` to avoid double-prepending
    BOS or similar.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = tokenizer.encode(query, add_special_tokens=False)
    full_tokens = tokenizer.encode(
        full_prompt, max_length=max_length, truncation=True, add_special_tokens=False
    )

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
        seed: Optional[int] = None
    ) -> List[dict]:
    """
    Load examples from unified JSONL format.

    File naming convention:
    - Validation: {data_dir}/eval/{task}/{task}_validation_data.jsonl
    - lr (extra dev split): {data_dir}/eval/{task}/{task}_lr_data.jsonl
    - Test: {data_dir}/eval/{task}/{task}_test_data.jsonl

    Args:
        data_dir: Base data directory
        task: Task name (tydiqa, samsum, nq_open)
        split: Which split to load ("validation", "test", or "lr")
        k: Number of examples to load
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
            examples.append(example)
            if seed is None and len(examples) >= k:
                break

    if seed is not None:
        random.Random(seed).shuffle(examples)

    return examples[:k]


def get_tydiqa_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """Get the TyDiQA dataset rendered with the tokenizer's native chat template."""
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

        prompt, answer = render_chat(tokenizer, user_content, assistant_content)

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

        prompt, answer = render_chat(tokenizer, user_content, assistant_content)

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


def get_squad_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """SQuAD closed-book QA loader (Q->A messages format, no context)."""
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, "squad", split, load_k, seed=seed)
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0
    for example in examples:
        if accepted_count >= k:
            break
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]
        if not assistant_content:
            continue
        prompt, answer = render_chat(tokenizer, user_content, assistant_content)
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False,
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1
    if rejected_count > 0:
        logger.info(f"SQuAD: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")
    return Dataset.from_dict(dataset)


def get_nq_open_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """NaturalQuestions-open closed-book QA loader (Q->A messages format)."""
    return _qa_messages_dataset(
        data_dir, tokenizer, max_length, "nq_open", split, k, seed, max_seq_length_threshold,
    )


def get_triviaqa_dataset(
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        split: str = "test",
        k: int = 5,
        seed: Optional[int] = None,
        max_seq_length_threshold: Optional[int] = None,
        **kwargs
    ) -> Dataset:
    """TriviaQA closed-book loader (Q->A messages format, identical to nq_open)."""
    return _qa_messages_dataset(
        data_dir, tokenizer, max_length, "triviaqa", split, k, seed, max_seq_length_threshold,
    )


def _qa_messages_dataset(data_dir, tokenizer, max_length, dataset_name, split, k, seed, max_seq_length_threshold):
    load_k = k * 3 if max_seq_length_threshold is not None else k
    examples = load_unified_jsonl(data_dir, dataset_name, split, load_k, seed=seed)
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    rejected_count = 0
    accepted_count = 0
    for example in examples:
        if accepted_count >= k:
            break
        messages = example.get("messages", [])
        if len(messages) < 2:
            continue
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]
        if not assistant_content:
            continue
        prompt, answer = render_chat(tokenizer, user_content, assistant_content)
        if max_seq_length_threshold is not None:
            token_length = estimate_token_length(tokenizer, prompt, answer)
            if token_length > max_seq_length_threshold:
                rejected_count += 1
                continue
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length,
            print_ex=True if accepted_count == 0 else False,
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
        accepted_count += 1
    if rejected_count > 0:
        logger.info(f"{dataset_name}: Rejected {rejected_count} samples exceeding length threshold {max_seq_length_threshold}")
    return Dataset.from_dict(dataset)


def get_dataset(task: str, **kwargs) -> Dataset:
    """
    Get the dataset for the given task.

    Args:
        task: The name of the task (tydiqa, samsum, nq_open).
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
        "tydiqa": get_tydiqa_dataset,
        "samsum": get_samsum_dataset,
        "nq_open": get_nq_open_dataset,
        "squad": get_squad_dataset,
        "triviaqa": get_triviaqa_dataset,
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
