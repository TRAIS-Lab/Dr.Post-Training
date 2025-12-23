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
    # Override: LR Scheduler
    # ===================
    # Reference implementation uses constant LR (no scheduler)
    # See: archive/LDA-ORL-main/rlhf-toxicity/scripts/ppo_config.py
    # The default in transformers is "linear" which decays LR over training
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": (
                "Learning rate scheduler type. Default 'constant' matches reference "
                "implementation. Use 'linear' for decay, 'cosine' for cosine annealing."
            )
        },
    )

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
        default=128,
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
        metadata={"help": "Number of PPO epochs per batch (default: 4)"},
    )
    mini_batch_size: int = field(
        default=1,
        metadata={
            "help": (
                "Mini-batch size for PPO backward passes (optimizer.step()). "
                "Reference uses 1, meaning one optimizer.step() per sample. "
                "This prevents ratio explosion by making small updates."
            )
        },
    )
    forward_batch_size: int = field(
        default=256,
        metadata={
            "help": (
                "Batch size for forward passes during PPO updates (computing new logprobs). "
                "Reference uses tracin_batch_size=256 for efficient GPU utilization. "
                "0 means use full batch size. This is separate from mini_batch_size "
                "which controls backward passes."
            )
        },
    )
    ratio_threshold: float = field(
        default=10.0,
        metadata={"help": "Skip batches where avg ratio exceeds this (prevents divergence)"},
    )

    # KL Control
    kl_coef: float = field(
        default=0.2,
        metadata={"help": "KL penalty coefficient (default: 0.2). Alias for init_kl_coef."},
    )
    init_kl_coef: float = field(
        default=0.2,
        metadata={
            "help": (
                "Initial KL penalty coefficient (default: 0.2). "
                "With adaptive KL control, this value changes during training."
            )
        },
    )
    kl_penalty: str = field(
        default="full",
        metadata={
            "help": (
                "KL penalty mode (reference: ppo_config.py:80-85): "
                "'kl': model_logp - ref_logp (can be negative), "
                "'abs': abs(model_logp - ref_logp) (always positive, prevents reward bonus), "
                "'mse': 0.5 * (model_logp - ref_logp)^2 (always positive), "
                "'full': true KL divergence using F.kl_div"
            )
        },
    )
    adap_kl_ctrl: bool = field(
        default=True,
        metadata={
            "help": (
                "Use adaptive KL control (default: True). "
                "Dynamically adjusts KL coefficient based on observed KL divergence."
            )
        },
    )
    target_kl: float = field(
        default=0.1,
        metadata={
            "help": (
                "Target KL divergence for adaptive KL control (default: 0.1). "
                "Also used for early stopping if enabled."
            )
        },
    )
    horizon: int = field(
        default=10000,
        metadata={
            "help": (
                "Horizon for adaptive KL control (default: 10000). "
                "Larger values = slower adaptation."
            )
        },
    )

    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clipping range for policy (default: 0.2)"},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "PPO clipping range for value (default: 0.2)"},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient (default: 0.1)"},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor (default: 1.0)"},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "GAE lambda parameter (default: 0.95)"},
    )

    # ===================
    # Generation Arguments
    # ===================
    max_new_tokens: int = field(
        default=30,
        metadata={"help": "Maximum new tokens to generate"},
    )
    min_new_tokens: int = field(
        default=20,
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
        default="reward_weighted",
        metadata={
            "help": (
                "Validation loss type for gradient capture: "
                "'logprob' (NLL on validation), "
                "'reward_weighted' (NLL weighted by reward - sequence-level attribution), "
                "'advantage_weighted' (NLL weighted by advantage)"
            )
        },
    )

    # ===================
    # Toxicity Evaluation (during training)
    # ===================
    enable_toxicity_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable toxicity evaluation during training. "
                "Uses a DIFFERENT classifier (DaNLP/da-electra-hatespeech-detection) "
                "than the reward model to provide unbiased evaluation."
            )
        },
    )
    eval_interval: int = field(
        default=0,
        metadata={
            "help": (
                "Steps between toxicity evaluations. "
                "0 = evaluate at the end of each epoch only. "
                "N > 0 = evaluate every N steps."
            )
        },
    )
    eval_n_samples: int = field(
        default=500,
        metadata={"help": "Number of samples for toxicity evaluation (default: 500)"},
    )
    eval_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for generation during evaluation"},
    )
    eval_on_step_generations: bool = field(
        default=True,
        metadata={
            "help": (
                "Evaluate toxicity on the generations produced during each PPO step. "
                "This provides per-step toxicity metrics without extra generation cost."
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

        # Validate kl_penalty (matching reference: ppo_config.py:217)
        valid_kl_penalties = ["kl", "abs", "mse", "full"]
        if self.kl_penalty not in valid_kl_penalties:
            raise ValueError(f"kl_penalty must be one of {valid_kl_penalties}, got {self.kl_penalty}")

        super().__post_init__()

    @property
    def has_compression(self) -> bool:
        """Whether gradient compression is enabled."""
        return self.sparsification is not None or self.projection is not None

    @property
    def has_selection(self) -> bool:
        """Whether data selection is enabled."""
        return self.method != "NA"
