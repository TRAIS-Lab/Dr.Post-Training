"""Chat rendering shared by training encoding, target loading, and evaluation.

Two rendering regimes are supported through one interface:

* **Native chat template** (e.g. Qwen3-Base ships one): every string that
  reaches the model is produced by ``tokenizer.apply_chat_template``.
* **Tulu fallback** (Llama-3.2-1B-Base ships no template): the open-instruct
  plaintext markers used by the paper's runs. The fallback Jinja template
  reproduces ``get_train_dataset.concat_messages`` byte-for-byte, so the
  legacy Llama experiments are unaffected.

Training encoding (``encode_messages_with_chat_template``) is the single encoder
behind ``get_train_dataset.encode_with_messages_format``. It supervises assistant
turns via the fast tokenizer's offset mapping, so it stays correct on Qwen3
multi-turn data where prefix renders are not byte-prefixes of the full render.
Like the rest of the repo it tokenizes with ``add_special_tokens=False``: the
chat template already emits every special token the model should see.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Open-instruct/Tulu-style fallback chat template (same text upstream used in
# get_val_dataset.py). Used when the tokenizer doesn't ship its own (e.g. base
# models like meta-llama/Llama-3.2-1B). The role markers are plaintext, tokenized
# by BPE, the convention of the open-instruct / LESS / Tulu codebases.
TULU_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'].strip() }}\n"
    "{%- elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'].strip() }}\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'].strip() }}{{ eos_token }}\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

_NATIVE_TEMPLATE_FLAG = "_drpt_native_chat_template"


def has_native_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """True if the tokenizer shipped its own chat template (not our fallback)."""
    flag = getattr(tokenizer, _NATIVE_TEMPLATE_FLAG, None)
    if flag is not None:
        return bool(flag)
    template = getattr(tokenizer, "chat_template", None)
    return bool(template) and template != TULU_CHAT_TEMPLATE


def ensure_chat_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Install the tulu fallback template if the tokenizer has none. Idempotent."""
    if getattr(tokenizer, _NATIVE_TEMPLATE_FLAG, None) is None:
        native = bool(getattr(tokenizer, "chat_template", None))
        setattr(tokenizer, _NATIVE_TEMPLATE_FLAG, native)
        if not native:
            tokenizer.chat_template = TULU_CHAT_TEMPLATE
            logger.info("Tokenizer has no chat template; installed tulu-style fallback")
    return tokenizer


def _template_kwargs(tokenizer: PreTrainedTokenizerBase, enable_thinking: Optional[bool]) -> Dict[str, Any]:
    # ``enable_thinking`` only means something to Qwen3-style templates; other
    # templates silently ignore unknown variables.
    if enable_thinking is None or not has_native_chat_template(tokenizer):
        return {}
    return {"enable_thinking": bool(enable_thinking)}


def render_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Render a conversation to a string with the active chat template."""
    ensure_chat_template(tokenizer)
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **_template_kwargs(tokenizer, enable_thinking),
    )


def render_generation_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    *,
    enable_thinking: bool = False,
    system_content: Optional[str] = None,
) -> str:
    """Render a single-turn prompt that ends where the assistant starts writing.

    ``enable_thinking=False`` is the default because training renders every
    supervised assistant turn with the template's empty ``<think>`` scaffold in
    the prompt position (see ``encode_messages_with_chat_template``), so the
    evaluation prompt should look the same.
    """
    messages: List[Dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return render_messages(
        tokenizer, messages, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def render_prompt_and_answer(
    tokenizer: PreTrainedTokenizerBase,
    user_content: str,
    assistant_content: str,
    *,
    system_content: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(prompt, answer)`` such that ``prompt + answer`` is the full render.

    ``prompt`` is the generation prompt (assistant header, plus the empty think
    scaffold on Qwen3); ``answer`` is the assistant content followed by the
    template's end-of-turn marker. Loss is meant to be computed on ``answer``.
    """
    prompt = render_generation_prompt(
        tokenizer, user_content, enable_thinking=False, system_content=system_content
    )
    messages: List[Dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    full = render_messages(tokenizer, messages)
    if not full.startswith(prompt):
        # Fall back to the header-only prompt (templates that do not inject a
        # think scaffold into the final turn).
        prompt = render_generation_prompt(
            tokenizer, user_content, enable_thinking=True, system_content=system_content
        )
        if not full.startswith(prompt):
            raise ValueError(
                "Chat template rendering is not prefix-consistent; cannot split "
                "prompt from answer.\nPROMPT:\n" + prompt + "\nFULL:\n" + full
            )
    return prompt, full[len(prompt):]


def get_eos_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """All ids that should terminate generation.

    Qwen3-Base's ``eos_token`` is ``<|endoftext|>`` but its chat template ends
    assistant turns with ``<|im_end|>``; an SFT'd model emits the latter. Pass
    both so ``model.generate`` stops on either.
    """
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id is not None
            and token_id != getattr(tokenizer, "unk_token_id", None)
            and isinstance(token_id, int)
            and token_id >= 0
            and token_id not in ids
        ):
            # ``convert_tokens_to_ids`` returns unk for unknown strings on most
            # tokenizers, but some return the string back; the isinstance check
            # guards that.
            if tokenizer.convert_ids_to_tokens(token_id) == token:
                ids.append(token_id)
    return ids


# ---------------------------------------------------------------------------
# Assistant-only supervision through the native chat template
# ---------------------------------------------------------------------------

_PROBE_USER = "DRPT_PROBE_QUESTION"
_PROBE_ANSWER = "DRPT_PROBE_ANSWER"


def end_of_turn_marker(tokenizer: PreTrainedTokenizerBase) -> str:
    """Text the template emits right after assistant content (e.g. ``<|im_end|>\\n``)."""
    rendered = render_messages(
        tokenizer,
        [
            {"role": "user", "content": _PROBE_USER},
            {"role": "assistant", "content": _PROBE_ANSWER},
        ],
    )
    index = rendered.rfind(_PROBE_ANSWER)
    if index == -1:
        return ""
    return rendered[index + len(_PROBE_ANSWER):]


def _assistant_char_spans(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    full_text: str,
) -> List[Tuple[int, int]]:
    """Character spans of ``full_text`` that hold supervised assistant tokens.

    For each assistant turn the span starts right after the assistant header
    (including any template scaffold such as Qwen3's empty think block, which is
    treated as prompt, not target) and ends after the end-of-turn marker.

    The start is located by rendering ``messages[:i]`` with the generation
    prompt; Qwen3 renders the *final* assistant turn with a think scaffold and
    intermediate turns without, so both variants are tried. The end is located
    by matching the assistant content itself, which avoids the prefix
    inconsistency of rendering ``messages[:i+1]`` on multi-turn Qwen3 data.
    """
    eot = end_of_turn_marker(tokenizer)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        start: Optional[int] = None
        for thinking in (False, True):
            head = render_messages(
                tokenizer,
                messages[:index],
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            if full_text.startswith(head) and len(head) >= cursor:
                start = len(head)
                break
        if start is None:
            # Prefix rendering failed (unusual role ordering); search for the
            # content after the previous span instead.
            probe = content.strip()
            found = full_text.find(probe, cursor) if probe else -1
            if found == -1:
                logger.warning(
                    "Could not locate assistant turn %d in rendered text; leaving it unsupervised",
                    index,
                )
                continue
            start = found

        end: Optional[int] = None
        for candidate in (content, content.lstrip("\n"), content.strip(), content.rstrip()):
            if candidate and full_text.startswith(candidate, start):
                end = start + len(candidate)
                break
        if end is None:
            # Content was transformed by the template (e.g. embedded </think>
            # splitting); fall back to the next end-of-turn marker.
            marker_at = full_text.find(eot, start) if eot else -1
            end = marker_at if marker_at != -1 else len(full_text)
        if eot and full_text.startswith(eot, end):
            end += len(eot)
        if end > start:
            spans.append((start, end))
            cursor = end
    return spans


def encode_messages_with_chat_template(
    example: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize a ``messages`` example with the native template; supervise assistant turns only.

    Labels are ``-100`` everywhere except assistant content plus the end-of-turn
    marker. Token/character alignment uses the fast tokenizer's offset mapping,
    so it does not depend on prefix-tokenization boundaries.
    """
    messages = example["messages"]
    if not messages:
        raise ValueError("messages field is empty.")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "encode_messages_with_chat_template requires a fast tokenizer "
            "(offset mapping); load with use_fast=True."
        )

    full_text = render_messages(tokenizer, messages)
    spans = _assistant_char_spans(tokenizer, messages, full_text)

    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,  # the template already emitted BOS/EOS-equivalents
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    offsets = encoded["offset_mapping"]
    labels = torch.full_like(input_ids, -100)
    span_index = 0
    for position, (char_start, char_end) in enumerate(offsets):
        if char_end <= char_start:
            continue  # zero-width entries (e.g. injected special tokens) carry no text
        while span_index < len(spans) and spans[span_index][1] <= char_start:
            span_index += 1
        if span_index < len(spans):
            span_start, span_end = spans[span_index]
            if char_start < span_end and char_end > span_start:
                labels[position] = input_ids[position]
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
