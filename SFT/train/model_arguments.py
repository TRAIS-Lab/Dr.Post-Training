"""
Model arguments for SFT experiments.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

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
        default_factory=lambda: ["all-linear"],
        metadata={"help": "Target modules for LoRA. Either explicit module-name suffixes (e.g. ['q_proj','k_proj','v_proj','o_proj']), or the single-element list ['all-linear'] which is unwrapped to the string 'all-linear' so PEFT applies LoRA to every nn.Linear except lm_head."},
    )

    # Flash attention
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"},
    )

    # Stop-token initialisation (Qwen3-Base's <|im_end|> row is near-untrained)
    init_eot_from_eos: bool = field(
        default=False,
        metadata={"help": "Before training, copy the <|endoftext|> embedding/output row into the <|im_end|> row so the chat end-of-turn token is emittable from step 0."},
    )

def add_padding_to_tokenizer(tokenizer):
    """ add the padding tokens in the tokenizer """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def init_end_of_turn_from_eos(model, tokenizer, eot_token: str = "<|im_end|>", eos_token: str = "<|endoftext|>") -> dict:
    """Copy the (well-trained) end-of-text row into the (near-untrained) chat end-of-turn row.

    Qwen3-Base ships <|im_end|>/<|im_start|> with tiny, effectively random embedding rows (norm ~0.38 vs ~1.6 for
    ordinary tokens); with tied embeddings that row is also the output row, so P(<|im_end|>) has to be learned from
    scratch and is not learned at LR < 1e-5 within 4000 steps (runaway generations). Copying the <|endoftext|> row makes
    end-of-turn as emittable as end-of-text at step 0; SFT then only has to separate the two. Handles tied and untied
    output embeddings. Returns a small dict for logging. No-op with a warning if either token is missing."""
    import torch, warnings
    eot = tokenizer.convert_tokens_to_ids(eot_token); eos = tokenizer.convert_tokens_to_ids(eos_token)
    unk = tokenizer.unk_token_id
    if eot is None or eos is None or eot == unk or eos == unk or eot == eos:
        warnings.warn(f"init_end_of_turn_from_eos: tokens not found ({eot_token}->{eot}, {eos_token}->{eos}); skipping")
        return {"applied": False}
    emb = model.get_input_embeddings().weight
    out_mod = model.get_output_embeddings(); out = out_mod.weight if out_mod is not None else None
    tied = out is not None and out.data_ptr() == emb.data_ptr()
    with torch.no_grad():
        before = emb[eot].float().norm().item()
        emb[eot].copy_(emb[eos])
        if out is not None and not tied:
            out[eot].copy_(out[eos])
        after = emb[eot].float().norm().item()
    return {"applied": True, "eot_id": eot, "eos_id": eos, "tied": tied, "eot_norm_before": round(before, 4),
            "eot_norm_after": round(after, 4), "eos_norm": round(emb[eos].float().norm().item(), 4)}
