"""Vendored copy of google-research/instruction_following_eval.

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Licensed under the Apache License, Version 2.0 (see the header of each module).

Only the imports were changed (absolute ``instruction_following_eval`` imports
became package-relative). The instruction implementations and the registry are
byte-for-byte upstream so IFEval strict/loose numbers match the official scorer.

Requires ``langdetect``, ``immutabledict``, ``nltk``, and ``absl-py``.
"""

from SFT.eval.tasks.ifeval_lib import instructions, instructions_registry, instructions_util

__all__ = ["instructions", "instructions_registry", "instructions_util"]
