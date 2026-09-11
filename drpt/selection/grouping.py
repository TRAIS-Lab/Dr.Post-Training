"""
Layer grouping for GroupWiseSubset curation.

GroupWiseSubset generalises the two existing curation granularities:

- LayerWiseSubset  == every hooked layer is its own group
- GlobalSubset     == a single group holding every hooked layer

Any other *partition* of the hooked layers is valid. Each group scores its
layers' gradients jointly, selects once when all of its layers have finished
backward, and assembles the selected gradient for all of them.

Presets (``granularity``)
-------------------------
layer     one group per hooked layer (== LayerWiseSubset)
sublayer  one group per (decoder block, first sub-module), e.g.
          ``model.layers.3.self_attn`` = {q,k,v,o}, ``model.layers.3.mlp`` = {gate,up,down}
block     one group per decoder block ``model.layers.N``
global    a single group (== GlobalSubset, one-pass)
custom    per-block rules given as a string (see :func:`parse_group_rules`)

Layers that do not live inside a decoder block (token embedding, ``lm_head``)
are singleton groups under every preset except ``global``.

Custom rule syntax
------------------
``"<name>=<member>[,<member>...];<name>=<member>..."`` — e.g.

    attn.qkv=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj;attn.o=self_attn.o_proj;
    mlp.gateup=mlp.gate_proj,mlp.up_proj;mlp.down=mlp.down_proj

For every hooked layer inside block ``P`` (``P = model.layers.N``) the layer's
suffix after ``P.`` is compared with each member: a member matches when its
dot-separated components appear contiguously in the suffix's components, so
``q_proj`` matches both ``self_attn.q_proj`` and the PEFT name
``self_attn.q_proj.lora_A.default``. The first matching rule wins and the
layer joins group ``P.<name>``. Unmatched layers stay singletons. The string
must not contain ``:`` or quotes so that it survives the YAML/bash config
parser used by ``train.sh``.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

GRANULARITIES = ("layer", "sublayer", "block", "global", "custom")

GLOBAL_GROUP_KEY = "__global__"

# ``<prefix>.layers.<N>.<suffix>`` — matches HF decoder stacks
# (``model.layers.3.mlp.down_proj``) and PEFT-wrapped names
# (``base_model.model.model.layers.3.self_attn.q_proj.lora_A.default``).
_BLOCK_RE = re.compile(r"^(?P<block>(?:.*?\.)?layers\.\d+)\.(?P<suffix>.+)$")


def split_block(layer_name: str) -> Optional[Tuple[str, str]]:
    """Split ``layer_name`` into ``(block_prefix, suffix)``; None if not in a block."""
    m = _BLOCK_RE.match(layer_name)
    if m is None:
        return None
    return m.group("block"), m.group("suffix")


def parse_group_rules(rules: str) -> "OrderedDict[str, List[str]]":
    """Parse the custom rule string into an ordered ``{name: [members]}`` mapping."""
    parsed: "OrderedDict[str, List[str]]" = OrderedDict()
    if rules is None:
        return parsed
    for raw in rules.split(";"):
        rule = raw.strip()
        if not rule:
            continue
        if "=" not in rule:
            raise ValueError(
                f"Invalid group rule {rule!r}: expected '<name>=<member>[,<member>...]'"
            )
        name, members_str = rule.split("=", 1)
        name = name.strip()
        members = [m.strip() for m in members_str.split(",") if m.strip()]
        if not name or not members:
            raise ValueError(f"Invalid group rule {rule!r}: empty name or member list")
        if name in parsed:
            raise ValueError(f"Duplicate group name {name!r} in rules")
        parsed[name] = members
    if not parsed:
        raise ValueError("selection_groups is set but contains no rules")
    return parsed


def _member_matches(member: str, suffix: str) -> bool:
    """True when ``member``'s components occur contiguously in ``suffix``'s components."""
    m_parts = member.split(".")
    s_parts = suffix.split(".")
    n = len(m_parts)
    if n == 0 or n > len(s_parts):
        return False
    return any(s_parts[i:i + n] == m_parts for i in range(len(s_parts) - n + 1))


def build_layer_groups(
    layer_names: Sequence[str],
    granularity: str = "block",
    rules: Optional[str] = None,
) -> List[str]:
    """
    Assign a group key to every hooked layer.

    Args:
        layer_names: Hooked layer names, in hook index order.
        granularity: One of :data:`GRANULARITIES`. ``rules`` implies ``custom``.
        rules: Custom rule string (see module docstring). Required for ``custom``.

    Returns:
        ``layer_groups[i]`` = group key (str) of ``layer_names[i]``.
    """
    if rules:
        # Custom rules take precedence over any preset name (the trainers pass
        # their default preset alongside the rules string).
        granularity = "custom"
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"Unknown selection granularity {granularity!r}; expected one of {GRANULARITIES}"
        )
    if granularity == "custom" and not rules:
        raise ValueError("selection_granularity='custom' requires selection_groups rules")

    parsed_rules = parse_group_rules(rules) if granularity == "custom" else None

    groups: List[str] = []
    for name in layer_names:
        if granularity == "layer":
            groups.append(name)
            continue
        if granularity == "global":
            groups.append(GLOBAL_GROUP_KEY)
            continue

        split = split_block(name)
        if split is None:
            # embed_tokens / lm_head / anything outside the decoder stack
            groups.append(name)
            continue
        block, suffix = split

        if granularity == "block":
            groups.append(block)
        elif granularity == "sublayer":
            groups.append(f"{block}.{suffix.split('.')[0]}")
        else:  # custom
            key = name
            for rule_name, members in parsed_rules.items():
                if any(_member_matches(m, suffix) for m in members):
                    key = f"{block}.{rule_name}"
                    break
            groups.append(key)
    return groups


def group_members(layer_names: Sequence[str], layer_groups: Sequence[str]) -> "OrderedDict[str, List[int]]":
    """``{group_key: [layer indices]}`` in first-appearance order."""
    if len(layer_names) != len(layer_groups):
        raise ValueError("layer_names and layer_groups must have the same length")
    out: "OrderedDict[str, List[int]]" = OrderedDict()
    for idx, key in enumerate(layer_groups):
        out.setdefault(key, []).append(idx)
    return out


def describe_layer_groups(
    layer_names: Sequence[str],
    layer_groups: Sequence[str],
    max_groups: int = 6,
) -> str:
    """Human-readable summary for logging."""
    members = group_members(layer_names, layer_groups)
    sizes = [len(v) for v in members.values()]
    lines = [
        f"{len(members)} selection groups over {len(layer_names)} hooked layers "
        f"(group sizes: min={min(sizes)}, max={max(sizes)}, "
        f"singletons={sum(1 for s in sizes if s == 1)})"
    ]
    for i, (key, idxs) in enumerate(members.items()):
        if i >= max_groups:
            lines.append(f"  ... and {len(members) - max_groups} more groups")
            break
        shown = ", ".join(layer_names[j] for j in idxs[:4])
        more = f", ... (+{len(idxs) - 4})" if len(idxs) > 4 else ""
        lines.append(f"  [{key}] ({len(idxs)}): {shown}{more}")
    return "\n".join(lines)
