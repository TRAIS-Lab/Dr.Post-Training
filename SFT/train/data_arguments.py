import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def none_or_str(value):
    """Convert string 'None' to Python None, otherwise return the value as-is."""
    if value == "None":
        return None
    return value


@dataclass
class DataArguments:
    data_dir: str = field(
        default="data",
        metadata={"help": "The directory containing training and evaluation data."}
    )
    train_files: List[str] = field(default_factory=list, metadata={
                                   "help": "The input training data files (multiple files in glob format)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    sample_data_seed: int = field(
        default=42, metadata={"help": ("The seed used for data sampling.")},
    )
    percentage: float = field(
        default=0.05,
        metadata={"help": "Sampling percentage for each dataset"},
    )


def get_data_statistics(lm_datasets, return_avg_length: bool = False):
    """
    Get the data statistics of the dataset.

    Args:
        lm_datasets: The dataset(s) to compute statistics for.
        return_avg_length: If True, return the average sequence length.

    Returns:
        If return_avg_length is True, returns the average sequence length (float).
        Otherwise, returns None.
    """
    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]

        completion_lens = []
        for labels in examples["labels"]:
            com_len = (torch.tensor(labels) > -1).sum()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    avg_length = None
    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(
            f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}")

        # Store avg length for the first (or only) dataset
        if avg_length is None:
            avg_length = length

    if return_avg_length:
        return avg_length
