"""
Training arguments for SFT experiments.
"""

from dataclasses import dataclass, field

from transformers import TrainingArguments as TA


fsdp_config = {
    "mpt7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "opt125m_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "mpt7b_lora": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_13b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "mistral_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
}


@dataclass
class TrainingArguments(TA):
    analysis_mode: float = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in analysis mode. "
            )
        },
    )
    analysis_dataset: str = field(
        default="bbh",
        metadata={
            "help": (
                "The dataset to use for analysis mode. "
            )
        },
    )
    train_dataset_names: str = field(
        default=None,
        metadata={
            "help": (
                "The dataset to use for training. "
            )
        },
    )


    # Data Curation Arguments
    method: str = field(
        default="NA",
        metadata={
            "help": (
                "Data curation method: 'NA' (no curation), "
                "'Layerwise' (per-layer curation), or 'Subset' (global curation)"
            )
        },
    )
    selection_frac: float = field(
        default=0.5,
        metadata={"help": "Fraction of samples to select (0-1)"},
    )
    selection_mode: str = field(
        default="topk",
        metadata={
            "help": (
                "Selection mode: 'topk' (select top frac samples by score) or "
                "'filtering' (drop bottom frac of negative-score samples)."
            )
        },
    )
    subject: str = field(
        default="sociology",
        metadata={"help": "Subject for MMLU/BBH evaluation"},
    )
    n_val: int = field(
        default=8,
        metadata={"help": "Number of validation samples for data curation"},
    )
    n_eval: int = field(
        default=500,
        metadata={"help": "Number of evaluation samples for generalization testing"},
    )
    val_batch_size_for_selection: int = field(
        default=1,
        metadata={
            "help": (
                "Batch size for validation data used during training for data curation. "
                "If None, defaults to per_device_train_batch_size. "
                "This allows independent control of batch size for data curation during training."
            )
        },
    )

    # Gradient Compression Arguments
    sparsification: str = field(
        default=None,
        metadata={
            "help": (
                "Sparsification method and dimension in format 'METHOD-DIM' or 'METHOD-DIM*DIM' for factorized. "
                "Examples: 'Rademacher-512', 'Gaussian-256*256'. Set to None to disable sparsification."
            )
        },
    )
    projection: str = field(
        default=None,
        metadata={
            "help": (
                "Projection method and dimension in format 'METHOD-DIM' or 'METHOD-DIM*DIM' for factorized. "
                "Examples: 'Gaussian-256', 'Rademacher-128*128'. Set to None to use identity (no projection)."
            )
        },
    )
    update_compressor_freq: int = field(
        default=200,
        metadata={
            "help": (
                "Number of steps between projector refreshes. "
                "Set to a large value (e.g., 1000000) to effectively disable refresh. Default: 200"
            )
        },
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
        default="reduced_ghost",
        metadata={
            "help": (
                "Scoring method for influence score computation: "
                "'reduced_ghost' (default, our ghost inner product, never materializes per-sample grads), "
                "'full_ghost' (GREATS-style ghost IP, materializes for 3D but true ghost for 2D), "
                "'direct' (explicit per-sample gradient materialization), "
                "'compress' (compressed per-sample gradients, requires score_compression to be set)."
            )
        },
    )
    subset_mode: str = field(
        default="one_pass",
        metadata={
            "help": (
                "Subset descent mode (only used when method='Subset'): "
                "'one_pass' (default, Algorithm 4.2): Single forward+backward pass, "
                "retains activations for post-hoc gradient assembly. "
                "'two_pass' (Algorithm 4.3): First pass for scoring, second pass on selected subset."
            )
        },
    )
    use_second_order: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use second-order interactions for data curation. "
                "If True, uses greedy curation considering sample similarities (O(k*n) complexity). "
                "If False (default), uses simple top-k curation based on scores."
            )
        },
    )
    val_strategy: str = field(
        default="separate_batch_factorized",
        metadata={
            "help": (
                "Validation gradient strategy for data curation: "
                "'separate_batch_factorized' (default): Separate val pass, store [V,S,O] and [V,S,I] factors. "
                "'separate_batch': Separate val pass, store mean gradient [O,I] per layer. "
                "'merged_batch': Merge train+val into single batch, compute val grad in same pass. "
                "All modes should produce identical gradients when selection_frac=1.0."
            )
        },
    )

    # Curation Recording (Case Study)
    record_selections: bool = field(
        default=False,
        metadata={
            "help": (
                "Record selected sample indices and scores per step for case study analysis. "
                "For Layerwise: records per-layer curation. For Subset: records global curation. "
                "Saves to output_dir/selection_records.json."
            )
        },
    )
    record_selections_freq: int = field(
        default=1,
        metadata={
            "help": (
                "Record curation decisions every N steps. Default: 1 (every step). "
                "Increase to reduce file size for long training runs."
            )
        },
    )

    # Profiling
    profile: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable PyTorch profiler. Profiles first 10 steps and saves trace to output_dir/profile/. "
                "View with: tensorboard --logdir=output_dir/profile/ or chrome://tracing"
            )
        },
    )
    profile_steps: int = field(
        default=10,
        metadata={"help": "Number of steps to profile (default: 10)"},
    )

    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        super().__post_init__()
