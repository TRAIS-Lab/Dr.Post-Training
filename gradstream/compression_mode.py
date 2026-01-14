"""
Compression mode configuration for GradStream.

This module defines the CompressionMode enum which specifies how gradients
are compressed for scoring and model updates.
"""

from enum import Enum


class CompressionMode(Enum):
    """
    Defines how gradients are compressed for scoring and updates.

    The modes form a hierarchy with increasing compression:

    - NONE: No compression anywhere. Full gradients for both scoring and updates.
    - SCORE_ONLY: Compress for scoring only, full gradients for model updates.
    - FULL: Compress for both scoring AND model updates (requires MeSO optimizer).

    Key invariant: If model updates use compressed gradients (FULL mode),
    scoring MUST also use compressed gradients with the same compressor.
    You cannot have compressed updates with full-gradient scoring.

    Usage:
        # No compression (baseline or selection-only)
        hook.compression_mode = CompressionMode.NONE

        # Hybrid mode: compressed scores, full gradient updates
        hook.compression_mode = CompressionMode.SCORE_ONLY

        # Full compression with MeSO optimizer
        hook.compression_mode = CompressionMode.FULL
    """

    NONE = "none"
    """No compression. Full gradients for scoring and model updates."""

    SCORE_ONLY = "score"
    """Compress for scoring only. Full gradients for model updates."""

    FULL = "full"
    """Full compression. Compressed gradients for both scoring and model updates."""

    @property
    def uses_compression(self) -> bool:
        """Check if this mode uses any compression."""
        return self != CompressionMode.NONE

    @property
    def uses_compressed_updates(self) -> bool:
        """Check if this mode uses compressed gradients for model updates."""
        return self == CompressionMode.FULL

    @property
    def uses_compressed_scoring(self) -> bool:
        """Check if this mode uses compressed gradients for scoring."""
        return self in (CompressionMode.SCORE_ONLY, CompressionMode.FULL)
