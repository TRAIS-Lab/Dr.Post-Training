"""
Difficulty predictor using attention-based similarity.

Implements few-shot regression for difficulty prediction as in RLVR paper:
- Query samples attend to reference samples via scaled dot-product attention
- Difficulty predicted as weighted sum of reference difficulties
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Literal
    from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DifficultyPredictor(nn.Module):
    """
    Attention-based difficulty predictor for RLVR.

    Uses scaled dot-product attention between query activations and
    reference activations to predict difficulty scores.

    For layer-wise prediction, each layer can have its own predictor
    or share weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 256,
        num_proj_layers: int = 2,
        tau: float = 1.0,
        share_projection: bool = True,
    ) -> None:
        """
        Initialize the difficulty predictor.

        Args:
            hidden_dim: Dimension of input activations
            proj_dim: Dimension of projection space for attention
            num_proj_layers: Number of MLP layers in projection
            tau: Temperature for softmax attention (lower = sharper)
            share_projection: If True, query and reference share projection weights
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.tau = tau
        self.share_projection = share_projection

        # Build projection network
        layers = []
        in_dim = hidden_dim
        for i in range(num_proj_layers - 1):
            layers.extend([
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            ])
            in_dim = proj_dim
        layers.append(nn.Linear(in_dim, proj_dim))

        self.query_proj = nn.Sequential(*layers)

        if share_projection:
            self.ref_proj = self.query_proj
        else:
            # Separate projection for reference
            ref_layers = []
            in_dim = hidden_dim
            for i in range(num_proj_layers - 1):
                ref_layers.extend([
                    nn.Linear(in_dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                ])
                in_dim = proj_dim
            ref_layers.append(nn.Linear(in_dim, proj_dim))
            self.ref_proj = nn.Sequential(*ref_layers)

        logger.debug(
            f"Initialized DifficultyPredictor: "
            f"hidden_dim={hidden_dim}, proj_dim={proj_dim}, tau={tau}"
        )

    def forward(
        self,
        query_activations: Tensor,
        ref_activations: Tensor,
        ref_difficulties: Tensor,
        tau: Optional[float] = None,
        return_weights: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Predict difficulty scores for query samples.

        Args:
            query_activations: [batch_size, hidden_dim] - query sample activations
            ref_activations: [num_ref, hidden_dim] - reference sample activations
            ref_difficulties: [num_ref] - reference difficulty scores (0-1)
            tau: Temperature override (if None, use self.tau)
            return_weights: If True, also return attention weights

        Returns:
            predicted_difficulties: [batch_size] - predicted difficulty scores
            weights: [batch_size, num_ref] - attention weights (if return_weights=True)
        """
        if tau is None:
            tau = self.tau

        # Project to attention space
        q_proj = self.query_proj(query_activations)  # [B, proj_dim]
        r_proj = self.ref_proj(ref_activations)      # [V, proj_dim]

        # Compute scaled dot-product attention scores
        # q_proj: [B, D], r_proj: [V, D]
        # scores: [B, V]
        scores = torch.mm(q_proj, r_proj.t()) / (self.proj_dim ** 0.5)

        # Apply temperature and softmax
        weights = F.softmax(scores / tau, dim=-1)  # [B, V]

        # Predict difficulty as weighted sum
        # ref_difficulties: [V] -> [1, V] for broadcasting
        predicted = torch.mv(weights, ref_difficulties)  # [B]

        if return_weights:
            return predicted, weights
        return predicted


class LayerWiseDifficultyPredictor(nn.Module):
    """
    Layer-wise difficulty predictor for per-layer selection.

    Each layer has its own predictor (or shared), producing
    per-layer difficulty predictions that can guide per-layer selection.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dims: list[int],
        proj_dim: int = 256,
        num_proj_layers: int = 2,
        tau: float = 1.0,
        share_across_layers: bool = False,
    ) -> None:
        """
        Initialize layer-wise difficulty predictor.

        Args:
            num_layers: Number of layers to predict for
            hidden_dims: Hidden dimension for each layer
            proj_dim: Projection dimension
            num_proj_layers: Number of MLP layers in projection
            tau: Temperature for softmax
            share_across_layers: If True, share predictor weights across layers
        """
        super().__init__()

        self.num_layers = num_layers
        self.tau = tau
        self.share_across_layers = share_across_layers

        if share_across_layers:
            # All layers must have same hidden dim
            if len(set(hidden_dims)) > 1:
                raise ValueError(
                    "Cannot share across layers with different hidden dims"
                )
            self.shared_predictor = DifficultyPredictor(
                hidden_dim=hidden_dims[0],
                proj_dim=proj_dim,
                num_proj_layers=num_proj_layers,
                tau=tau,
            )
            self.predictors = None
        else:
            # Separate predictor per layer
            self.predictors = nn.ModuleList([
                DifficultyPredictor(
                    hidden_dim=hidden_dims[i],
                    proj_dim=proj_dim,
                    num_proj_layers=num_proj_layers,
                    tau=tau,
                )
                for i in range(num_layers)
            ])
            self.shared_predictor = None

    def forward(
        self,
        query_activations: list[Tensor],
        ref_activations: list[Tensor],
        ref_difficulties: Tensor,
        tau: Optional[float] = None,
    ) -> list[Tensor]:
        """
        Predict difficulty scores for each layer.

        Args:
            query_activations: List of [batch_size, hidden_dim] per layer
            ref_activations: List of [num_ref, hidden_dim] per layer
            ref_difficulties: [num_ref] - shared across layers

        Returns:
            List of [batch_size] difficulty predictions per layer
        """
        predictions = []
        for i in range(self.num_layers):
            if self.share_across_layers:
                pred = self.shared_predictor(
                    query_activations[i],
                    ref_activations[i],
                    ref_difficulties,
                    tau=tau,
                )
            else:
                pred = self.predictors[i](
                    query_activations[i],
                    ref_activations[i],
                    ref_difficulties,
                    tau=tau,
                )
            predictions.append(pred)

        return predictions

    def predict_aggregated(
        self,
        query_activations: list[Tensor],
        ref_activations: list[Tensor],
        ref_difficulties: Tensor,
        aggregation: str = 'mean',
        tau: Optional[float] = None,
    ) -> Tensor:
        """
        Predict difficulty with aggregation across layers.

        Args:
            query_activations: List of [batch_size, hidden_dim] per layer
            ref_activations: List of [num_ref, hidden_dim] per layer
            ref_difficulties: [num_ref]
            aggregation: 'mean', 'max', 'min'

        Returns:
            [batch_size] aggregated difficulty predictions
        """
        predictions = self.forward(
            query_activations, ref_activations, ref_difficulties, tau
        )

        # Stack and aggregate
        stacked = torch.stack(predictions, dim=0)  # [num_layers, batch_size]

        if aggregation == 'mean':
            return stacked.mean(dim=0)
        elif aggregation == 'max':
            return stacked.max(dim=0)[0]
        elif aggregation == 'min':
            return stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


class SimpleCosineSimilarityPredictor:
    """
    Simple difficulty predictor using cosine similarity.

    No learned parameters - uses raw cosine similarity between
    query and reference activations.
    """

    def __init__(self, tau: float = 1.0):
        """
        Initialize simple predictor.

        Args:
            tau: Temperature for softmax attention
        """
        self.tau = tau

    def __call__(
        self,
        query_activations: Tensor,
        ref_activations: Tensor,
        ref_difficulties: Tensor,
        tau: Optional[float] = None,
    ) -> Tensor:
        """
        Predict difficulty using cosine similarity.

        Args:
            query_activations: [batch_size, hidden_dim]
            ref_activations: [num_ref, hidden_dim]
            ref_difficulties: [num_ref]

        Returns:
            [batch_size] predicted difficulties
        """
        if tau is None:
            tau = self.tau

        # Normalize for cosine similarity
        q_norm = F.normalize(query_activations, dim=-1)  # [B, D]
        r_norm = F.normalize(ref_activations, dim=-1)    # [V, D]

        # Cosine similarity scores
        scores = torch.mm(q_norm, r_norm.t())  # [B, V]

        # Apply temperature and softmax
        weights = F.softmax(scores / tau, dim=-1)

        # Predict difficulty
        return torch.mv(weights, ref_difficulties)


def create_difficulty_predictor(
    method: Literal['learned', 'simple', 'layerwise'],
    hidden_dim: int | list[int],
    num_layers: int = 1,
    proj_dim: int = 256,
    tau: float = 1.0,
    **kwargs,
) -> nn.Module | SimpleCosineSimilarityPredictor:
    """
    Factory function to create difficulty predictor.

    Args:
        method: 'learned' (MLP projection), 'simple' (cosine similarity),
                'layerwise' (per-layer predictors)
        hidden_dim: Hidden dimension(s)
        num_layers: Number of layers (for layerwise)
        proj_dim: Projection dimension
        tau: Temperature

    Returns:
        Difficulty predictor instance
    """
    if method == 'learned':
        return DifficultyPredictor(
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            tau=tau,
            **kwargs,
        )
    elif method == 'simple':
        return SimpleCosineSimilarityPredictor(tau=tau)
    elif method == 'layerwise':
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        return LayerWiseDifficultyPredictor(
            num_layers=num_layers,
            hidden_dims=hidden_dim,
            proj_dim=proj_dim,
            tau=tau,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
