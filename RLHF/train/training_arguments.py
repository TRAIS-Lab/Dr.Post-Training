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
    - Data selection arguments (method, filter_frac)
    - Gradient compression arguments (sparsification, projection)
    - PPO-specific arguments
    """

    # ===================
    # Learning Rate and Scheduler
    # ===================
    learning_rate: float = field(
        default=1e-5,
        metadata={
            "help": (
                "Learning rate for Adam optimizer (default: 1e-5, matching reference). "
                "The transformers default is 5e-5 which is too high for PPO."
            )
        },
    )
    learning_rate_vhead: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate for the value head (default: None, uses same as learning_rate). "
                "The value head is randomly initialized while the main model is pretrained, "
                "so it may benefit from a different (often higher) learning rate."
            )
        },
    )
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
        metadata={"help": "Task name: 'toxicity'"},
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
                "'IIF' (pre-filter entire rollout before PPO epochs), "
                "'Streaming' (per-layer selection, single-pass), "
                "'GREATS' (global selection, two-pass)"
            )
        },
    )
    filter_frac: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fraction of negative-influence samples to drop (0-1). "
                "1.0 = drop all negative samples, 0.5 = drop bottom 50% of negative samples. "
                "All positive-influence samples are always kept."
            )
        },
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
    n_val: int = field(
        default=8,
        metadata={
            "help": (
                "Number of validation samples for data selection. "
                "Uses a fixed validation set (separate from training) for computing "
                "validation gradients. Set to 0 to use self-referencing validation "
                "(training buffer as validation set)."
            )
        },
    )
    val_batch_size: int = field(
        default=1,
        metadata={
            "help": (
                "Batch size for validation gradient computation. "
                "Controls how many validation samples are processed per gradient capture pass. "
                "Smaller values use less memory but require more passes."
            )
        },
    )
    val_generation_interval: int = field(
        default=0,
        metadata={
            "help": (
                "Steps between regenerating validation completions. "
                "0 = generate once at start and reuse throughout training. "
                "N > 0 = regenerate every N steps (more compute but fresher validation)."
            )
        },
    )
    val_loss_type: str = field(
        default="seqloss-lastadv",
        metadata={
            "help": (
                "Validation loss type for data selection gradient computation. "
                "Options: "
                "'seqloss-lastadv' (default) - sequence log prob * last-token GAE advantage, "
                "'seqloss-reward' - sequence log prob * normalized raw reward. "
                "Matches LDA-ORL reference implementation options."
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
    score_compression_dim: int = field(
        default=64,
        metadata={
            "help": (
                "Dimension for score-only compression (factorized, so actual dim is score_compression_dim^2). "
                "Used when data selection is enabled but no explicit compression is specified. "
                "Set to 0 to disable auto score compression. Default: 64 (i.e., 64*64)"
            )
        },
    )

    # ===================
    # PPO-specific Arguments
    # ===================
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of PPO epochs per batch (default: 4)"},
    )
    mini_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Mini-batch size for PPO updates (default: 8). "
                "Larger values reduce number of updates per rollout "
                "but improve computational efficiency."
            )
        },
    )
    ratio_threshold: float = field(
        default=10.0,
        metadata={"help": "Skip batches where avg ratio exceeds this (prevents divergence)"},
    )

    # KL Control
    init_kl_coef: float = field(
        default=0.04,
        metadata={
            "help": (
                "Initial KL penalty coefficient (default: 0.04, matching reference). "
                "With adaptive KL control, this value adjusts during training."
            )
        },
    )
    kl_estimator: str = field(
        default="k1",
        metadata={
            "help": (
                "KL divergence estimator (http://joschu.net/blog/kl-approx.html): "
                "'k1': -log(r) = policy_logp - ref_logp (unbiased, higher variance, can be negative), "
                "'k2': 0.5 * log(r)^2 (biased, low variance, for logging only), "
                "'k3': (r - 1) - log(r) (unbiased, low variance, always positive)."
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
    target: float = field(
        default=1.0,
        metadata={
            "help": (
                "Target KL divergence for adaptive KL control (default: 1.0). "
                "The AdaptiveKLController adjusts kl_coef to maintain KL near this value. "
                "Higher values allow more policy divergence from reference."
            )
        },
    )
    target_kl: float = field(
        default=0.1,
        metadata={
            "help": (
                "KL threshold for early stopping (default: 0.1). "
                "If policy KL exceeds 1.5 * target_kl, the optimization step is skipped. "
                "This is separate from `target` which controls the adaptive KL coefficient."
            )
        },
    )
    early_stopping: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable early stopping based on KL divergence (default: False). "
                "If True, optimization steps are skipped when policy KL exceeds 1.5 * target_kl."
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
    max_grad_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Maximum gradient norm for gradient clipping (default: None). "
                "Set to None or 0 to disable gradient clipping. "
                "Helps prevent training instability from gradient explosions."
            )
        },
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
        default=0,
        metadata={
            "help": (
                "Minimum new tokens to generate (ONLY used for evaluation, not training). "
                "During training rollouts, min_length=-1 is used to avoid negative KL exploitation. "
                "See: https://huggingface.co/docs/trl/main/en/how_to_train"
            )
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"},
    )
    top_k: float = field(
        default=0.0,
        metadata={"help": "Top-k sampling (0.0 to disable)"},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling"},
    )

    # ===================
    # Evaluation (during training)
    # ===================
    enable_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable evaluation during training. "
                "Uses DaNLP classifier (different from reward model)."
            )
        },
    )
    eval_interval: int = field(
        default=1,
        metadata={
            "help": (
                "Steps between evaluations. "
                "0 = evaluate at the end of each epoch only. "
                "N > 0 = evaluate every N steps."
            )
        },
    )
    n_eval: int = field(
        default=500,
        metadata={"help": "Number of samples for evaluation (default: 500)"},
    )
    eval_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for generation during evaluation"},
    )
    eval_on_step_generations: bool = field(
        default=True,
        metadata={
            "help": (
                "Evaluate on the generations produced during each PPO step. "
                "This provides per-step metrics without extra generation cost."
            )
        },
    )

    # ===================
    # Debugging
    # ===================
    debug_n_samples: int = field(
        default=3,
        metadata={
            "help": (
                "Number of samples to print for debugging rollouts (default: 3). "
                "Set to 0 to disable debug printing. "
                "Prints prompt, response, and reward for first N samples in each batch."
            )
        },
    )

    def __post_init__(self):
        # Validate task
        valid_tasks = ["toxicity"]
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got {self.task}")

        # Validate method
        valid_methods = ["NA", "IIF", "Streaming", "GREATS"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")

        # Validate filter_frac
        if not 0 <= self.filter_frac <= 1:
            raise ValueError(f"filter_frac must be in [0, 1], got {self.filter_frac}")

        # Validate kl_estimator (matching TRL experimental PPO)
        valid_kl_estimators = ["k1", "k2", "k3"]
        if self.kl_estimator not in valid_kl_estimators:
            raise ValueError(f"kl_estimator must be one of {valid_kl_estimators}, got {self.kl_estimator}")

        super().__post_init__()

    @property
    def has_compression(self) -> bool:
        """Whether gradient compression is enabled."""
        return self.sparsification is not None or self.projection is not None

    @property
    def has_selection(self) -> bool:
        """Whether data selection is enabled."""
        return self.method != "NA"

    @property
    def use_validation_set(self) -> bool:
        """Whether to use a separate validation set (vs self-referencing with training buffer)."""
        return self.n_val > 0
