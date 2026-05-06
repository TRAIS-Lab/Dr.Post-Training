import contextlib
from functools import partial
from typing import List, Union

import torch
from datasets import load_dataset


@contextlib.contextmanager
def temp_seed(seed):
    torch_state = torch.get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)


def get_train_files_for_dataset(data_dir: str, dataset_name: str) -> List[str]:
    """
    Map a training dataset name to its file path(s).

    Args:
        data_dir: Base directory containing training data
        dataset_name: Name of the training dataset

    Returns:
        List of file paths for the training dataset
    """
    dataset_mapping = {
        # Single dataset files
        "alpaca":   [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "dolly":    [f"{data_dir}/train/dolly/dolly_data.jsonl"],
        "flan_v2":  [f"{data_dir}/train/flan_v2/flan_v2_data.jsonl"],
        "cot":      [f"{data_dir}/train/cot/cot_data.jsonl"],
        "oasst1":   [f"{data_dir}/train/oasst1/oasst1_data.jsonl"],
        "tulu3":    [f"{data_dir}/train/tulu3/tulu3_data.jsonl"],
        "samsum":   [f"{data_dir}/train/samsum/samsum_train_data.jsonl"],
        "nq_open":  [f"{data_dir}/train/nq_open/nq_open_data.jsonl"],
        "triviaqa": [f"{data_dir}/train/triviaqa/triviaqa_data.jsonl"],
        "squad":    [f"{data_dir}/train/squad/squad_data.jsonl"],
        # LESS mixture (flan_v2 + cot + dolly + oasst1)
        "less": [
            f"{data_dir}/train/flan_v2/flan_v2_data.jsonl",
            f"{data_dir}/train/cot/cot_data.jsonl",
            f"{data_dir}/train/dolly/dolly_data.jsonl",
            f"{data_dir}/train/oasst1/oasst1_data.jsonl",
        ],
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unknown training dataset: {dataset_name}. "
                        f"Available: {list(dataset_mapping.keys())}")

    return dataset_mapping[dataset_name]


def _get_default_train_files(data_dir: str, task: str) -> List[str]:
    """
    Get default training files based on task.

    Args:
        data_dir: Base directory containing training data
        task: Evaluation task name

    Returns:
        List of default training file paths for the task
    """
    # LESS mixture for general instruction tuning evaluation tasks
    less_mixture = [
        f"{data_dir}/train/flan_v2/flan_v2_data.jsonl",
        f"{data_dir}/train/cot/cot_data.jsonl",
        f"{data_dir}/train/dolly/dolly_data.jsonl",
        f"{data_dir}/train/oasst1/oasst1_data.jsonl"
    ]

    task_defaults = {
        "samsum":   [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "tydiqa":   less_mixture,
        "triviaqa": [f"{data_dir}/train/nq_open/nq_open_data.jsonl"],
        "nq_open":  [f"{data_dir}/train/triviaqa/triviaqa_data.jsonl"],
    }

    return task_defaults.get(task, less_mixture)


def get_training_dataset(data_dir: str, task: str, tokenizer, max_seq_length,
                         sample_percentage=1.0, seed=0, train_files: List[str] = None,
                         train_dataset_names: List[str] = None):
    """
    Get training dataset with a specified seed.

    Args:
        data_dir: Base directory containing training data
        task: Evaluation task name (mmlu, samsum, tydiqa, bbh, gsm8k, math500)
        tokenizer: Tokenizer to use for encoding
        max_seq_length: Maximum sequence length
        sample_percentage: Percentage of data to sample
        seed: Random seed for sampling
        train_files: Optional explicit list of training files (overrides all other selection)
        train_dataset_names: Optional list of training dataset names (e.g., ['wizardlm', 'alpaca'])

    Returns:
        Encoded training dataset
    """
    # Priority: train_files > train_dataset_names > task-based default
    if train_files is None:
        if train_dataset_names is not None:
            # Use explicitly specified training datasets
            train_files = []
            for name in train_dataset_names:
                train_files.extend(get_train_files_for_dataset(data_dir, name))
        else:
            # Fall back to task-based defaults
            train_files = _get_default_train_files(data_dir, task)

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = torch.randperm(len(processed_datasets))[:sample_size].tolist()

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False):
    """Encode messages-format examples with the tokenizer's native chat template."""
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    if "messages" not in raw_datasets.column_names:
        raise ValueError(
            "Training data must have a 'messages' column. Got columns: "
            f"{raw_datasets.column_names}"
        )
    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''Render `example['messages']` with the tokenizer's chat template and
    produce labels that supervise only each assistant *answer* span.

    Template-agnostic: works for any tokenizer whose `apply_chat_template`
    is prefix-stable — i.e. partial-prefix renders are byte-prefixes of the
    full render. This holds for Llama (open-instruct/Tulu fallback, Llama-Instruct)
    and for Qwen3 single-turn data. For Qwen3 multi-turn the auto-injected
    empty `<think></think>` block on the last assistant means the partial render
    of an *intermediate* assistant turn won't be a strict prefix; this isn't
    a concern for the active settings (alpaca/samsum is single-turn) but
    re-evaluate when re-enabling tulu3 / oasst1.

    For each assistant message, the label-bearing span is
    `[len(prefix_to_assistant), len(prefix_through_assistant))` where:

      prefix_to_assistant       = render(messages[:i], add_generation_prompt=True)
      prefix_through_assistant  = render(messages[:i+1], add_generation_prompt=False)

    Token positions are recovered by re-encoding each prefix with
    `add_special_tokens=False` (same setting `apply_chat_template(tokenize=True)`
    uses internally, so the encodings agree at boundary positions).
    '''
    from SFT.data.get_val_dataset import ensure_chat_template
    ensure_chat_template(tokenizer)

    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(
        full_text, add_special_tokens=False, max_length=max_seq_length, truncation=True
    )
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)

    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        prefix_to_text = tokenizer.apply_chat_template(
            messages[:message_idx], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix_through_text = tokenizer.apply_chat_template(
            messages[:message_idx + 1], tokenize=False, add_generation_prompt=False,
        )
        if not full_text.startswith(prefix_through_text):
            # Multi-turn Qwen edge case (see docstring). Skip this turn rather
            # than risk mislabeled positions.
            continue

        start_len = min(
            len(tokenizer.encode(prefix_to_text, add_special_tokens=False)),
            len(input_ids),
        )
        end_len = min(
            len(tokenizer.encode(prefix_through_text, add_special_tokens=False)),
            len(input_ids),
        )

        if start_len < end_len:
            labels[start_len:end_len] = input_ids[start_len:end_len]
        if end_len >= len(input_ids):
            break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }
