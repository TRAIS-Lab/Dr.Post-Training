"""
Implementation of projection methods for dimension reduction.

This file contains functions to construct all random projection methods for dimension reduction.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, List, Union, Optional

import torch
from torch import Tensor

from .utils import vectorize, get_parameter_chunk_sizes, _rademacher, _mask

class ProjectionType(str, Enum):
    """Projection type used for projectors."""
    identity: str = "identity"
    normal: str = "normal"
    rademacher: str = "rademacher"
    sjlt: str = "sjlt"
    random_mask: str = "random_mask"


def _preprocess(
    features: Union[dict, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert features to tensor on correct device and dtype.

    Args:
        features (Union[dict, Tensor]): Input features.
        device (torch.device): Target device.
        dtype (torch.dtype): Target dtype.

    Returns:
        Tensor: Preprocessed features tensor.
    """
    if isinstance(features, dict):
        features = vectorize(features, device=device)
    elif features.device != device:
        features = features.to(device)
    if features.dtype != dtype:
        features = features.to(dtype)
    return features


class AbstractProjector(ABC):
    """Base Class for projectors."""

    @abstractmethod
    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: torch.device,
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension after the projection.
            seed (int): Random seed for the generation of the sketching
                (projection) matrix.
            proj_type (ProjectionType): The random projection type used for the
                projection. Available options are "sjlt" (if cuda), "rademacher",
                "normal", "random_mask", "identity".
            device (torch.device): Device to use. Defaults to cpu.
        """
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on feature matrix.

        This function will take features and an ensemble_id, which allows us
        to generate different projection matrices, and output the projected
        matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """

    @abstractmethod
    def transpose(self, projected_features: Tensor, ensemble_id: int = 0) -> Tensor:
        """Apply transpose of projection to recover original feature space.

        This is the inverse operation of project(), mathematically P^T @ projected_features.

        Args:
            projected_features (Tensor): Features in projected space [batch, proj_dim]
            ensemble_id (int): A unique ID for this ensemble (must match project call)

        Returns:
            Tensor: Features in original space [batch, feature_dim]
        """

    @abstractmethod
    def refresh(self, new_seed: int) -> None:
        """Refresh the projector with new randomness.

        This regenerates the projection matrix/indices using a new seed,
        similar to GaLore's periodic subspace refresh mechanism.

        Args:
            new_seed (int): New random seed for regenerating randomness
        """

    @abstractmethod
    def free_memory(self) -> None:
        """Frees up memory used by the projector."""


class BasicProjector(AbstractProjector):
    """A simple block-wise implementation of the projection.

    The projection matrix is generated on-device in blocks.
    The accumulated result across blocks is returned.

    Note: This class will be significantly slower and have a larger memory
    footprint than the CudaProjector. It is recommended that you use this method
    only if the CudaProjector is not available to you -- e.g. if you don't have
    a CUDA-enabled device with compute capability >=7.0 (see
    https://developer.nvidia.com/cuda-gpus).
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: torch.device,
        block_size: int = 100,
        dtype: torch.dtype = torch.float32,
        ensemble_id: int = 0,
    ) -> None:
        """Initializes hyperparameters for BasicProjector.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension after the projection.
            seed (int): Random seed for the generation of the sketching
                (projection) matrix.
            proj_type (ProjectionType): The random projection type used for the
                projection. Available options are "rademacher", "normal",
                "random_mask", "identity".
            device (torch.device): Device to use. Defaults to cpu.
            block_size (int): Maximum number of projection dimension allowed.
                Thus, min(block_size, proj_dim) will be used as the actual
                projection dimension.
            dtype (torch.dtype): The dtype of the projected matrix.
            ensemble_id (int): A unique ID for this ensemble.
        """
        super().__init__(feature_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.ensemble_id = ensemble_id

        if proj_type in {ProjectionType.normal, ProjectionType.rademacher}:
            self.proj_matrix = torch.empty(
                self.feature_dim,
                self.block_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.proj_matrix_available = True

        self.generator = torch.Generator(device=self.device)

        self.get_generator_states()
        if proj_type == ProjectionType.random_mask:
            self._gen_randomness_mask(self.generator_states[0])
        elif proj_type == ProjectionType.identity:
            pass  # No randomness needed for identity projection
        else:
            self._gen_randomness_dense(self.generator_states[0])

    def free_memory(self) -> None:
        """Delete the projection matrix."""
        if hasattr(self, "proj_matrix"):
            del self.proj_matrix
            self.proj_matrix_available = False

    def get_generator_states(self) -> None:
        """Set generator seeds for each block."""
        self.generator_states = []
        self.seeds = []
        self.jl_size = self.feature_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.ensemble_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def _gen_randomness_mask(self, generator_state: List) -> None:
        """Generate random mask indices for random_mask projection.

        Args:
            generator_state (List): A list of generator states. Usually each
                block will be given a unique generator states.
        """
        self.generator.set_state(generator_state)
        self.active_indices = _mask(
            self.feature_dim,
            self.proj_dim,
            self.generator,
            self.device,
        )

    def _gen_randomness_dense(self, generator_state: List) -> None:
        """Set generator states and generate sketch matrices.

        Args:
            generator_state (List): A list of generator states. Usually each
                block will be given a unique generator states.

        Raises:
            KeyError: Projection type is not recognized.
        """
        if not self.proj_matrix_available:
            self.proj_matrix = torch.empty(
                self.feature_dim,
                self.block_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.proj_matrix_available = True

        self.generator.set_state(generator_state)

        if self.proj_type == ProjectionType.normal:
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher:
            self.proj_matrix = _rademacher(
                self.proj_matrix,
                self.generator,
                self.dtype,
            )
        else:
            msg = f"Projection type {self.proj_type} not recognized."
            raise KeyError(msg)

    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """
        features = _preprocess(features, self.device, self.dtype)

        if ensemble_id != self.ensemble_id:
            self.ensemble_id = ensemble_id
            self.get_generator_states()  # regenerate random seeds for new ensemble_id
            if self.proj_type == ProjectionType.random_mask:
                self._gen_randomness_mask(self.generator_states[0])
            elif self.proj_type == ProjectionType.identity:
                pass  # No randomness needed for identity projection
            elif self.num_blocks == 1:
                self._gen_randomness_dense(self.generator_states[0])

        # Handle random_mask projection separately
        if self.proj_type == ProjectionType.random_mask:
            return features[:, self.active_indices]
        elif self.proj_type == ProjectionType.identity:
            return features

        sketch = torch.zeros(
            size=(features.size(0), self.proj_dim),
            dtype=self.dtype,
            device=self.device,
        )

        if self.num_blocks == 1:
            torch.matmul(features.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self._gen_randomness_dense(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = (
                    features.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )

        sketch = sketch / (self.proj_dim ** 0.5)

        return sketch.type(features.dtype)

    def transpose(self, projected_features: Tensor, ensemble_id: int = 0) -> Tensor:
        """Apply transpose of projection to recover original feature space.

        Args:
            projected_features (Tensor): Features in projected space [batch, proj_dim]
            ensemble_id (int): A unique ID for this ensemble

        Returns:
            Tensor: Features in original space [batch, feature_dim]
        """
        if ensemble_id != self.ensemble_id:
            self.ensemble_id = ensemble_id
            self.get_generator_states()
            if self.proj_type == ProjectionType.random_mask:
                self._gen_randomness_mask(self.generator_states[0])
            elif self.num_blocks == 1:
                self._gen_randomness_dense(self.generator_states[0])

        # Handle random_mask: scatter operation
        if self.proj_type == ProjectionType.random_mask:
            original = torch.zeros(
                projected_features.size(0), self.feature_dim,
                dtype=projected_features.dtype,
                device=self.device
            )
            original[:, self.active_indices] = projected_features
            return original
        elif self.proj_type == ProjectionType.identity:
            return projected_features

        # For dense projections: P^T @ projected_features

        original = torch.zeros(
            size=(projected_features.size(0), self.feature_dim),
            dtype=self.dtype,
            device=self.device,
        )

        if self.num_blocks == 1:
            # P^T is just proj_matrix.T
            torch.matmul(projected_features, self.proj_matrix.T, out=original)
        else:
            for ind in range(self.num_blocks):
                self._gen_randomness_dense(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                original += projected_features[:, st:ed] @ self.proj_matrix[:, : (ed - st)].T

        original = original / (self.proj_dim ** 0.5)

        return original.type(projected_features.dtype)

    def refresh(self, new_seed: int) -> None:
        """Refresh the projector with new randomness.

        Args:
            new_seed (int): New random seed
        """
        self.seed = new_seed
        self.get_generator_states()

        if self.proj_type == ProjectionType.random_mask:
            self._gen_randomness_mask(self.generator_states[0])
        elif self.num_blocks == 1:
            self._gen_randomness_dense(self.generator_states[0])


class CudaProjector(AbstractProjector):
    """Projector implemented using CUDA.

    A performant implementation of the projection
    for CUDA with compute capability >= 7.0.
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initializes hyperparameters for CudaProjector.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension we project *to* during the projection step
            seed (int): Random seed.
            proj_type (ProjectionType): The random projection type used for the
                projection. Available options are "sjlt", "rademacher", "normal",
                "random_mask", "identity".
            device (torch.device): Device to use.
            dtype (torch.dtype): The dtype used in the projector.

        Raises:
            ValueError: When attempting to use this on a non-CUDA device.
            ModuleNotFoundError: When sjlt is not installed.
        """
        super().__init__(feature_dim, proj_dim, seed, proj_type, device)
        self.dtype = dtype

        if self.device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; \
            Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        # Use a generator for reproducible randomness
        self.generator = torch.Generator(device=self.device)
        # Track the current ensemble to know when to regenerate randomness
        self.current_ensemble_id = -1  # Init to -1 to force generation on first call

        # Initialize placeholders for projection components
        if self.proj_type == ProjectionType.sjlt:
            # Check for sjlt import early if needed
            try:
                from sjlt import SJLTProjection  # noqa: F401
            except ImportError:
                msg = "sjlt not found. Please run `pip install sjlt` to install."
                raise ModuleNotFoundError(msg) from None
            self.sjlt = None
        elif self.proj_type in [ProjectionType.rademacher, ProjectionType.normal]:
            self.proj_matrix = None
        elif self.proj_type == ProjectionType.random_mask:
            self.active_indices = None


    def _gen_randomness_sjlt(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None) -> None:
        """Generates randomness for 'sjlt' projection.

        Args:
            input_dim (int): Input dimension for SJLT. If None, uses self.feature_dim.
            output_dim (int): Output dimension for SJLT. If None, uses self.proj_dim.
        """
        from sjlt import SJLTProjection

        if input_dim is None:
            input_dim = self.feature_dim
        if output_dim is None:
            output_dim = self.proj_dim

        c = 1  # Hard set the column sparsity to 1
        rand_indices = torch.randint(
            output_dim,
            (input_dim, c),
            generator=self.generator,
            device=self.device,
        )
        rand_signs = (
            torch.randint(
                0,
                2,
                (input_dim, c),
                generator=self.generator,
                device=self.device,
            )
            * 2
            - 1
        )

        # Recreate SJLT object if dimensions don't match
        need_recreate = (
            self.sjlt is None
            or self.sjlt.rand_indices.shape[0] != input_dim
            or rand_indices.shape[1] != self.sjlt.rand_indices.shape[1]
        )

        if need_recreate:
            self.sjlt = SJLTProjection(
                input_dim,
                output_dim,
                c,
                device=self.device,
            )

        self.sjlt.rand_indices.copy_(rand_indices)
        self.sjlt.rand_signs.copy_(rand_signs.to(torch.int8))

    def _gen_randomness_mask(self, input_dim: Optional[int] = None, output_dim: Optional[int] = None) -> None:
        """Generate random mask indices for random_mask projection.

        Args:
            input_dim (int): Input dimension for mask. If None, uses self.feature_dim.
            output_dim (int): Output dimension for mask. If None, uses self.proj_dim.
        """
        if input_dim is None:
            input_dim = self.feature_dim
        if output_dim is None:
            output_dim = self.proj_dim

        self.active_indices = _mask(
            input_dim,
            output_dim,
            self.generator,
            self.device,
        )

    def _gen_randomness_dense(self, method: str) -> None:
        """Generates the random projection matrix for dense projections.

        Args:
            method (str): The method to use for generating the projection
                matrix. Must be either "rademacher" or "normal".
        """
        if self.proj_matrix is None:
            self.proj_matrix = torch.empty(
                self.feature_dim,
                self.proj_dim,
                dtype=self.dtype,
                device=self.device,
            )

        if method == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif method == "rademacher":
            self.proj_matrix = _rademacher(
                self.proj_matrix,
                self.generator,
                self.dtype,
            )

    def _generate_randomness(self, ensemble_id: int) -> None:
        """Generates the random projection components for a given ensemble_id.

        Args:
            ensemble_id (int): A unique ID for this ensemble to generate
                reproducible randomness.

        Raises:
            ValueError: If the projection type is unknown.
        """
        # Create a unique, reproducible seed for this specific ensemble
        current_seed = self.seed + int(1e5) * ensemble_id
        self.generator.manual_seed(current_seed)

        # Initialize based on method
        if self.proj_type == ProjectionType.sjlt:
            self._gen_randomness_sjlt()
        elif self.proj_type == ProjectionType.rademacher:
            self._gen_randomness_dense("rademacher")
        elif self.proj_type == ProjectionType.normal:
            self._gen_randomness_dense("normal")
        elif self.proj_type == ProjectionType.random_mask:
            self._gen_randomness_mask()
        elif self.proj_type == ProjectionType.identity:
            pass  # No randomness needed for identity projection
        else:
            msg = f"Unknown projection type: {self.proj_type}"
            raise ValueError(msg)

    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """
        # Regenerate randomness if the ensemble_id has changed
        if ensemble_id != self.current_ensemble_id:
            self._generate_randomness(ensemble_id)
            self.current_ensemble_id = ensemble_id

        features = _preprocess(features, self.device, self.dtype)

        if self.proj_type in [ProjectionType.rademacher, ProjectionType.normal]:
            result = features @ self.proj_matrix
            result /= (self.proj_dim ** 0.5)

        elif self.proj_type == ProjectionType.sjlt:
            with torch.no_grad():
                result = self.sjlt(features)

        elif self.proj_type == ProjectionType.random_mask:
            result = features[:, self.active_indices]

        elif self.proj_type == ProjectionType.identity:
            result = features

        return result

    def transpose(self, features: Union[dict, Tensor], ensemble_id: int = 0) -> Tensor:
        """Apply transpose of projection to recover original feature space.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The transposed-projected features
        """
        # Regenerate randomness if the ensemble_id has changed
        if ensemble_id != self.current_ensemble_id:
            self._generate_randomness(ensemble_id)
            self.current_ensemble_id = ensemble_id

        features = _preprocess(features, self.device, self.dtype)

        if self.proj_type in [ProjectionType.rademacher, ProjectionType.normal]:
            result = features @ self.proj_matrix.T
            result /= (self.proj_dim ** 0.5)

        elif self.proj_type == ProjectionType.sjlt:
            # SJLT has its own transpose method
            with torch.no_grad():
                result = self.sjlt.transpose(features)

        elif self.proj_type == ProjectionType.random_mask:
            # Scatter operation
            result = torch.zeros(
                features.size(0), self.feature_dim,
                dtype=features.dtype,
                device=self.device
            )
            result[:, self.active_indices] = features

        elif self.proj_type == ProjectionType.identity:
            result = features

        return result

    def refresh(self, new_seed: int) -> None:
        """Refresh the projector with new randomness.

        Args:
            new_seed (int): New random seed
        """
        self.seed = new_seed
        # Immediately regenerate randomness for ensemble_id=0
        # This ensures active_indices and other state are updated synchronously
        self._generate_randomness(ensemble_id=0)
        self.current_ensemble_id = 0

    def free_memory(self) -> None:
        """A no-op method."""


class ChunkedCudaProjector:
    """Chunked CudaProjector implemented using CUDA.

    This projector is used when (# dim of features)*(# batch size) is too large.
    If the features are gradients, then (# dim of features) equals to the number
    of parameters in the model.
    """

    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        dim_per_chunk: list,
        feature_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initializes hyperparameters for ChunkedCudaProjector.

        Args:
            projector_per_chunk (list): A list of projectors. Specifying
                the projector used by each chunk.
            max_chunk_size (int): The maximum size of each chunk.
            dim_per_chunk (list): The number of feature dimensions per chunk.
            feature_batch_size (int): The batch size of input feature.
            device (torch.device): Device to use.
            dtype (torch.dtype): The dtype of the projected matrix.
        """
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.feature_dim = sum(dim_per_chunk)  # Total feature dimension across all chunks
        self.dim_per_chunk = dim_per_chunk
        self.feature_batch_size = feature_batch_size
        self.max_chunk_size = max_chunk_size
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self) -> None:
        """Allocate zero tensor for input."""
        if self.input_allocated:
            return

        self.ch_input = torch.zeros(
            size=(self.feature_batch_size, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self) -> None:
        """Frees up memory used by the projector."""
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """
        # allocate zero tensor for output
        ch_output = torch.zeros(
            size=(self.feature_batch_size, self.proj_dim),
            device=self.device,
            dtype=self.dtype,
        )
        # force the input to be Tensor for now
        # TODO: support dict input
        if isinstance(features, dict):
            features = vectorize(features, device=self.device)
        if features.device.type != self.device:
            features = features.to(self.device)

        pointer = 0
        for chunk_idx, chunk_dim in enumerate(self.dim_per_chunk):
            ch_output.add_(
                self.projector_per_chunk[chunk_idx].project(
                    features[:, pointer : pointer + chunk_dim].contiguous(),
                    ensemble_id=ensemble_id,
                ),
            )

            pointer += chunk_dim

        return ch_output

    def transpose(self, projected_features: Tensor, ensemble_id: int = 0) -> Tensor:
        """Apply transpose of projection to recover original feature space.

        Args:
            projected_features (Tensor): Features in projected space [batch, proj_dim]
            ensemble_id (int): A unique ID for this ensemble

        Returns:
            Tensor: Features in original space [batch, feature_dim]
        """
        # Allocate output tensor for full feature space
        original = torch.zeros(
            size=(projected_features.size(0), self.feature_dim),
            dtype=self.dtype,
            device=self.device,
        )

        pointer = 0
        for chunk_idx, chunk_dim in enumerate(self.dim_per_chunk):
            # Each chunk projector transposes from proj_dim back to its chunk_dim
            chunk_result = self.projector_per_chunk[chunk_idx].transpose(
                projected_features,
                ensemble_id=ensemble_id,
            )
            original[:, pointer : pointer + chunk_dim] = chunk_result

            pointer += chunk_dim

        return original

    def refresh(self, new_seed: int) -> None:
        """Refresh all chunk projectors with new randomness.

        Args:
            new_seed (int): New random seed
        """
        # Refresh each chunk projector with a unique seed
        for idx, projector in enumerate(self.projector_per_chunk):
            chunk_seed = new_seed + idx * 1000
            projector.refresh(chunk_seed)


def make_random_projector(
    param_shape_list: List,
    feature_batch_size: int,
    proj_dim: int,
    proj_max_batch_size: int,
    device: torch.device,
    proj_seed: int = 0,
    proj_type: ProjectionType = ProjectionType.sjlt,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Initialize random projector by the info of feature about to be projected.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of total parameter
            size of each module in a torch.nn.Module model. Total parameter size
            of each module equals to feature_batch_size * param_size of that module.
        feature_batch_size (int): The batch size of each tensor in the feature
            about to be projected. The typical type of feature are gradients of
            torch.nn.Module model but can be restricted to this.
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The batch size used to calculate safe chunk sizes
            for ChunkedCudaProjector (to avoid int32 overflow). This determines when
            parameters need to be split into multiple chunks. Not stored in projectors.
        device (torch.device): Device to use. Defaults to cpu.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (ProjectionType): The random projection type used for the
            projection. Available options are "sjlt" (if cuda), "rademacher",
            "normal", "random_mask", "identity".
        dtype (torch.dtype): The dtype used in the projector.

    Returns:
        The initialized projector object
        (CudaProjector, ChunkedCudaProjector, or BasicProjector).
    """
    # the total feature dim
    feature_dim = sum(param_shape_list)

    projector = BasicProjector if device.type == "cpu" else CudaProjector

    if projector == CudaProjector:
        max_chunk_size, param_chunk_sizes = get_parameter_chunk_sizes(
            param_shape_list,
            proj_max_batch_size,
        )
        if len(param_chunk_sizes) == 1:
            assigned_projector = projector(
                feature_dim=feature_dim,
                proj_dim=proj_dim,
                seed=proj_seed,
                proj_type=proj_type,
                device=device,
                dtype=dtype,
            )
        else:  # we have to use the ChunkedCudaProjector
            generator = torch.Generator(device=device)
            generator.manual_seed(proj_seed)

            # Generate seeds using torch.randint
            seeds = torch.randint(
                low=0,
                high=500,
                size=(len(param_chunk_sizes),),
                generator=generator,
                dtype=torch.int64,
                device=device,
            ).tolist()  # Convert to list for indexing

            projector_per_chunk = [
                projector(
                    feature_dim=chunk_size,
                    proj_dim=proj_dim,
                    seed=seeds[i],
                    proj_type=proj_type,
                    device=device,
                    dtype=dtype,
                )
                for i, chunk_size in enumerate(param_chunk_sizes)
            ]
            assigned_projector = ChunkedCudaProjector(
                projector_per_chunk,
                max_chunk_size,
                param_chunk_sizes,
                feature_batch_size,
                device,
                dtype,
            )
    else:
        assigned_projector = projector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=proj_seed,
            proj_type=proj_type,
            dtype=dtype,
            device=device,
        )

    return assigned_projector


def random_project(
    feature: Union[Dict[str, Tensor], Tensor],
    feature_batch_size: int,
    proj_dim: int,
    proj_max_batch_size: int,
    proj_seed: int = 0,
    proj_type: str = "normal",
    *,
    device: Union[str, torch.device] = "cpu",
) -> Union[BasicProjector, CudaProjector, ChunkedCudaProjector]:
    """Randomly projects the features to a smaller dimension.

    Args:
        feature (Union[Dict[str, Tensor], Tensor]): The feature needs to be
            projected. This can simple be a tensor with size [feature_batch_size,
            feature_dim]. Or typically, if the this is gradient of some
            torch.nn.Module models, it will have the structure similar to the
            result of model.named_parameters().
        feature_batch_size (int): The batch size of each tensor in the feature
            about to be projected. The typical type of feature are gradients of
            torch.nn.Module model but can restricted to this.
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The batch size used to calculate safe chunk sizes
            for ChunkedCudaProjector (to avoid int32 overflow). This determines when
            parameters need to be split into multiple chunks.
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        proj_type (str): The random projection type used for the projection.
            Available options are "identity" "sjlt", "rademacher", "normal", "random_mask".
        device (Union[str, torch.device]): "cuda" or "cpu". Defaults to "cpu".

    Raises:
        ValueError: When an invalid proj_type or device is provided.

    Returns:
        A projector object (BasicProjector, CudaProjector, or ChunkedCudaProjector)
        that can be used to project features via the .project() method.
    """
    # check the type of feature
    if isinstance(feature, dict):
        param_shape_list = [
            feature[param_name].numel() // feature_batch_size for param_name in feature
        ]
        dtype = feature[next(iter(feature))].dtype
    else:
        param_shape_list = [feature.numel() // feature_batch_size]
        dtype = feature.dtype

    # convert device to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # convert proj_type to ProjectionType
    # Define valid projection types for each device
    proj_type_mapping = {
        "identity": ProjectionType.identity,
        "rademacher": ProjectionType.rademacher,
        "normal": ProjectionType.normal,
        "random_mask": ProjectionType.random_mask,
        "sjlt": ProjectionType.sjlt,
    }

    valid_proj_types = {
        "cpu": {"rademacher", "normal", "random_mask", "identity"},
        "cuda": {"sjlt", "rademacher", "normal", "random_mask", "identity"},
    }

    if device.type not in valid_proj_types:
        msg = f"Invalid device type {device.type}. \
            Available options are 'cpu' and 'cuda'."
        raise ValueError(msg)

    if proj_type not in valid_proj_types[device.type]:
        available = ", ".join(f"'{t}'" for t in sorted(valid_proj_types[device.type]))
        msg = f"Invalid proj_type '{proj_type}' for {device.type}. \
            Available options are {available}."
        raise ValueError(msg)

    proj_type = proj_type_mapping[proj_type]

    return make_random_projector(
        param_shape_list=param_shape_list,
        feature_batch_size=feature_batch_size,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        device=device,
        proj_seed=proj_seed,
        proj_type=proj_type,
        dtype=dtype,
    )