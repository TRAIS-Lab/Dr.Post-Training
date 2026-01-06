"""
Adaptive difficulty prediction module for RLVR.

This module provides difficulty prediction models for data selection:
- FewShotRegressor: Attention-based few-shot regression using activation similarity
- TextEncoder: Frozen LLM encoder for generating embeddings
- Various scaling methods for calibration (platt, temperature, group_logit_temp)

Based on: https://arxiv.org/pdf/2506.05316
"""

from .model import FewShotRegressor, TextEncoder, RegressionHead, ResidualHead
from .load_data import TeacherDataset
from .utils import compute_metrics, linear_calibrate

__all__ = [
    "FewShotRegressor",
    "TextEncoder",
    "RegressionHead",
    "ResidualHead",
    "TeacherDataset",
    "compute_metrics",
    "linear_calibrate",
]
