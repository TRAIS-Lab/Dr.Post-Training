import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, LlamaTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    # LoRA arguments
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for training"},
    )
    lora_r: int = field(
        default=32,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: float = field(
        default=1,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Target modules for LoRA"},
    )

    # Flash attention
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"},
    )

def add_padding_to_tokenizer(tokenizer):
    """ add the padding tokens in the tokenizer """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
