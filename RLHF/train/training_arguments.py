"""
Training arguments for RLHF experiments.

Following SFT conventions:
- `method`: Controls data curation (NA, IIF, LayerWiseSubset, GlobalSubset, GroupWiseSubset)
- `sparsification`/`projection`: Controls compression (implies MeSO optimizer)
- PPO-specific arguments for RLHF
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as TA


@dataclass
class TrainingArguments(TA):
    """
    Training arguments for RLHF with layer_wise_subset data curation.

    Inherits from transformers.TrainingArguments and adds:
    - Data curation arguments (method, filter_frac)
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
    # Data Curation (following SFT conventions)
    # ===================
    method: str = field(
        default="NA",
        metadata={
            "help": (
                "Training method: "
                "'NA' (baseline, no curation), "
                "'IIF' (pre-filter entire rollout before PPO epochs), "
                "'LayerWiseSubset' (per-layer curation, single-pass), "
                "'GlobalSubset' (global curation, two-pass), "
                "'GroupWiseSubset' (per-layer-group curation, single-pass; see "
                "--selection_granularity / --selection_groups). 'BlockWiseSubset' and "
                "'SublayerWiseSubset' are aliases for GroupWiseSubset with granularity block / sublayer."
            )
        },
    )
    selection_granularity: str = field(
        default="block",
        metadata={
            "help": (
                "GroupWiseSubset only. Layer grouping preset: 'layer' (== LayerWiseSubset), "
                "'sublayer' (per block: attention vs MLP), 'block' (per decoder block, default), "
                "'global' (== GlobalSubset one_pass), 'custom' (rules from --selection_groups)."
            )
        },
    )
    selection_groups: str = field(
        default=None,
        metadata={
            "help": (
                "GroupWiseSubset only. Custom per-block grouping rules "
                "'<name>=<member>[,<member>..];<name>=..' matched against the layer name after "
                "'...layers.N.' (works with PEFT LoRA names). Implies selection_granularity=custom."
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
                "Use second-order curation (greedy with similarity matrix). "
                "Slower but more accurate."
            )
        },
    )
    n_val: int = field(
        default=8,
        metadata={
            "help": (
                "Number of validation samples for data curation. "
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
    train_on_val: bool = field(
        default=False,
        metadata={
            "help": (
                "Target-only baseline: use the n_val held-out validation prompts as the PPO "
                "training set (method must be NA). Compares 'train directly on the target "
                "prompts' against curating the full pool toward them."
            )
        },
    )
    val_loss_type: str = field(
        default="reward",
        metadata={
            "help": (
                "Validation loss type for data curation gradient computation. "
                "Options: "
                "'reward' (default) - per-token loss -normalized_reward * log_prob_t, "
                "'token-pg' - token-level REINFORCE, per-token loss -A_t * log_prob_t, "
                "'train-loss' - actual training objective (clipped surrogate + value loss). "
                "All are reduced per --loss_reduction (per-example token mean, averaged over the "
                "validation batch, under 'sample_mean')."
            )
        },
    )
    loss_reduction: str = field(
        default="sample_mean",
        metadata={
            "help": (
                "Batch-loss convention shared by the PPO training loss, the validation target and the "
                "curation scale factors (see drpt.losses). "
                "'sample_mean' (default): every response is one item; policy and value losses are "
                "per-example token means averaged over the (mini-)batch, so per-sample gradients are "
                "gradients of per-example losses and curated updates are plain means over the kept "
                "samples, as in the paper. "
                "'token_mean': token mean over the micro-batch (policy/value) and per-sequence token "
                "sums for the 'reward'/'token-pg' targets (length-weighted items)."
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
    score_compression: str = field(
        default=None,
        metadata={
            "help": (
                "Score-only compression for influence score computation. "
                "Same format as sparsification: 'METHOD-DIM*DIM'. "
                "Examples: 'normal-64*64' (Gaussian, 64x64 factorized). "
                "Set to None to disable (use exact scoring). Default: None"
            )
        },
    )

    scoring_method: str = field(
        default="pip",
        metadata={
            "help": (
                "Scoring method for influence score computation: "
                "'pip' (per-token inner product, default), 'gip' (ghost inner product), "
                "'direct', 'compress'."
            )
        },
    )
    subset_mode: str = field(
        default="two_pass",
        metadata={
            "help": (
                "GlobalSubset descent mode: "
                "'one_pass' (Algorithm 4.2, single backward + post-hoc assembly) or "
                "'two_pass' (Algorithm 4.3, scoring pass + gradient pass on selected subset). "
                "Default: two_pass for RLHF."
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
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling (0 to disable)"},
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

        # Method aliases -> GroupWiseSubset + granularity preset
        _aliases = {"BlockWiseSubset": "block", "SublayerWiseSubset": "sublayer"}
        if self.method in _aliases:
            self.selection_granularity = _aliases[self.method]
            self.method = "GroupWiseSubset"
        if self.selection_groups is not None and self.selection_groups.strip() == "":
            self.selection_groups = None
        if self.selection_groups is not None:
            self.selection_granularity = "custom"

        # Validate method
        valid_methods = ["NA", "IIF", "LayerWiseSubset", "GlobalSubset", "GroupWiseSubset"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        if self.method == "GroupWiseSubset":
            from drpt.selection.grouping import GRANULARITIES
            if self.selection_granularity not in GRANULARITIES:
                raise ValueError(
                    f"selection_granularity must be one of {GRANULARITIES}, got {self.selection_granularity!r}"
                )
            if self.selection_granularity == "custom" and self.selection_groups is None:
                raise ValueError("selection_granularity='custom' requires --selection_groups")

        # Validate filter_frac
        if not 0 <= self.filter_frac <= 1:
            raise ValueError(f"filter_frac must be in [0, 1], got {self.filter_frac}")

        # Validate val_loss_type (names shared with train.sh: rew | tpg | tloss)
        valid_val_loss_types = ["reward", "token-pg", "train-loss"]
        if self.val_loss_type not in valid_val_loss_types:
            raise ValueError(
                f"val_loss_type must be one of {valid_val_loss_types}, got {self.val_loss_type}"
            )

        from drpt.losses import validate_loss_reduction
        validate_loss_reduction(self.loss_reduction)

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
        """Whether data curation is enabled."""
        return self.method != "NA"

    @property
    def use_validation_set(self) -> bool:
        """Whether to use a separate validation set (vs self-referencing with training buffer)."""
        return self.n_val > 0
