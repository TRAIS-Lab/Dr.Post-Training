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
        "alpaca": [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "dolly": [f"{data_dir}/train/dolly/dolly_data.jsonl"],
        "flan_v2": [f"{data_dir}/train/flan_v2/flan_v2_data.jsonl"],
        "cot": [f"{data_dir}/train/cot/cot_data.jsonl"],
        "oasst1": [f"{data_dir}/train/oasst1/oasst1_data.jsonl"],
        "gsm8k": [f"{data_dir}/train/gsm8k/gsm8k_train_data.jsonl"],
        "vicuna": [f"{data_dir}/train/vicuna/vicuna_data.jsonl"],
        "wizardlm": [f"{data_dir}/train/wizardlm/wizardlm_data.jsonl"],
        "openhermes": [f"{data_dir}/train/openhermes/openhermes_data.jsonl"],
        "tulu3": [f"{data_dir}/train/tulu3/tulu3_data.jsonl"],
        "samsum": [f"{data_dir}/train/samsum/samsum_train_data.jsonl"],
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
        "samsum": [f"{data_dir}/train/alpaca/alpaca_data.jsonl"],
        "gsm8k": [f"{data_dir}/train/gsm8k/gsm8k_train_data.jsonl"],
        # Test-only tasks use LESS mixture by default
        "mmlu": less_mixture,
        "bbh": less_mixture,
        "tydiqa": less_mixture,
        "math500": less_mixture,
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


def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False, func_name="encode_with_messages_format"):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    return encode_function


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/truncated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
