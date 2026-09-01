"""
Task-specific evaluation modules.

Each module provides a compute_accuracy function that evaluates
a model on that specific task.
"""

from .samsum import compute_accuracy as compute_samsum_accuracy
from .tydiqa import compute_accuracy as compute_tydiqa_accuracy
from .nq_open import compute_accuracy as compute_nq_open_accuracy
from .squad import compute_accuracy as compute_squad_accuracy

__all__ = [
    "compute_samsum_accuracy",
    "compute_tydiqa_accuracy",
    "compute_nq_open_accuracy",
    "compute_squad_accuracy",
]
