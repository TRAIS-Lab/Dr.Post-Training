"""
Task-specific evaluation modules.

Each module provides a compute_accuracy function that evaluates
a model on that specific task.
"""

from .samsum import compute_accuracy as compute_samsum_accuracy
from .tydiqa import compute_accuracy as compute_tydiqa_accuracy
from .mmlu import compute_accuracy as compute_mmlu_accuracy
from .bbh import compute_accuracy as compute_bbh_accuracy
from .gsm8k import compute_accuracy as compute_gsm8k_accuracy
from .math500 import compute_accuracy as compute_math500_accuracy

__all__ = [
    "compute_samsum_accuracy",
    "compute_tydiqa_accuracy",
    "compute_mmlu_accuracy",
    "compute_bbh_accuracy",
    "compute_gsm8k_accuracy",
    "compute_math500_accuracy",
]
