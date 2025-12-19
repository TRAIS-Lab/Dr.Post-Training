"""
Training arguments for RLHF experiments.

Following SFT conventions:
- `method`: Controls data selection (NA, Streaming, GREATS)
- `sparsification`/`projection`: Controls compression (implies MeSO optimizer)
- PPO-specific arguments for RLHF
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as TA


@dataclass
class TrainingArguments(TA):
    """
    Training arguments for RLHF with gradient streaming.

    Inherits from transformers.TrainingArguments and adds:
    - Data selection arguments (method, selection_frac)
    - Gradient compression arguments (sparsification, projection)
    - PPO-specific arguments
    """

    # ===================
    # Task Configuration
    # ===================
    task: str = field(
        default="toxicity",
        metadata={"help": "Task name: toxicity, imdb, etc."},
    )

    # ===================
    # Data Selection (following SFT conventions)
    # ===================
    method: str = field(
        default="NA",
        metadata={
            "help": (
                "Training method: "
                "'NA' (baseline, no selection), "
                "'Streaming' (per-layer selection, single-pass), "
                "'GREATS' (global selection, two-pass)"
            )
        },
    )
    selection_frac: float = field(
        default=0.5,
        metadata={"help": "Fraction of samples to select (0-1)"},
    )
    n_val: int = field(
        default=16,
        metadata={"help": "Number of validation samples for data selection"},
    )
    val_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Batch size for validation data during selection. "
                "If None, uses all n_val samples."
            )
        },
    )
    val_strategy: str = field(
        default="random",
        metadata={"help": "Validation strategy: 'random' or 'top' (most toxic)"},
    )
    use_second_order: bool = field(
        default=False,
        metadata={
            "help": (
                "Use second-order selection (greedy with similarity matrix). "
                "Slower but more accurate."
            )
        },
    )

    # ===================
    # Gradient Compression (following SFT conventions)
    # ===================
    sparsification: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Sparsification method and dimension: 'METHOD-DIM*DIM'. "
                "Examples: 'Rademacher-64*64', 'Gaussian-32*32'. "
                "None to disable. Implies MeSO optimizer."
            )
        },
    )
    projection: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Projection method and dimension: 'METHOD-DIM'. "
                "Examples: 'Gaussian-256', 'Rademacher-512'. "
                "None to disable. Implies MeSO optimizer."
            )
        },
    )
    update_compressor_freq: int = field(
        default=200,
        metadata={"help": "Steps between compressor refreshes"},
    )

    # ===================
    # PPO-specific Arguments
    # ===================
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of PPO epochs per batch"},
    )
    kl_coef: float = field(
        default=0.04,
        metadata={"help": "KL penalty coefficient"},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clipping range for policy"},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "PPO clipping range for value"},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient"},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor"},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "GAE lambda parameter"},
    )
    target_kl: Optional[float] = field(
        default=None,
        metadata={"help": "Target KL divergence for early stopping (None to disable)"},
    )

    # ===================
    # Generation Arguments
    # ===================
    max_new_tokens: int = field(
        default=30,
        metadata={"help": "Maximum new tokens to generate"},
    )
    min_new_tokens: int = field(
        default=10,
        metadata={"help": "Minimum new tokens to generate"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling (0 to disable)"},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling"},
    )

    # ===================
    # Validation Loss Type
    # ===================
    val_loss_type: str = field(
        default="logprob",
        metadata={
            "help": (
                "Validation loss type for gradient capture: "
                "'logprob' (NLL on validation), "
                "'reward_weighted' (NLL weighted by reward), "
                "'advantage_weighted' (NLL weighted by advantage)"
            )
        },
    )

    def __post_init__(self):
        # Validate method
        valid_methods = ["NA", "Streaming", "GREATS"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

        # Validate selection_frac
        if not 0 < self.selection_frac <= 1:
            raise ValueError(f"selection_frac must be in (0, 1], got {self.selection_frac}")

        super().__post_init__()

    @property
    def has_compression(self) -> bool:
        """Whether gradient compression is enabled."""
        return self.sparsification is not None or self.projection is not None

    @property
    def has_selection(self) -> bool:
        """Whether data selection is enabled."""
        return self.method != "NA"
