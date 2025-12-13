from dataclasses import asdict, dataclass, field, fields

from transformers import TrainingArguments as TA
from transformers.utils import logging

logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

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


    ### added ###
    method: str = field(default='Regular', metadata={"help": "training method"})
    selection_frac: float = field(default=1.0, metadata={"help": "training method"})
    subject: str = field(default='abstract_algebra', metadata={"help": "subject for evaluation"})
    n_val: int = field(default=5, metadata={"help": "number of validation data (for data selection)"})
    n_eval: int = field(default=500, metadata={"help": "number of evaluation data (for generalization testing)"})
    val_batch_size_for_selection: int = field(
        default=None,
        metadata={
            "help": (
                "Batch size for validation data used during training for data selection. "
                "If None, defaults to per_device_train_batch_size. "
                "This allows independent control of batch size for data selection during training."
            )
        },
    )

    ### Gradient compression arguments ###
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
    use_compressed_optimizer: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the compressed optimizer (MeSOAdamW). "
                "When True, uses compressed optimizer states for memory efficiency. "
                "Requires gradient compression to be enabled (sparsification or projection)."
            )
        },
    )
    selection_mode: str = field(
        default="global",
        metadata={
            "help": (
                "Data selection granularity for GREATS/GradNorm with MeSO. "
                "Options: 'global' (default) or 'per_layer'. "
                "global: Accumulates scores across all layers, selects globally, then does second pass for gradients. "
                "per_layer: Each layer independently selects samples (single pass, uses block-diagonal approximation)."
            )
        },
    )
    use_second_order: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use second-order interactions for data selection. "
                "If True, uses greedy selection considering sample similarities (O(k*n) complexity). "
                "If False (default), uses simple top-k selection based on scores (~200x faster)."
            )
        },
    )

    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        super().__post_init__()
