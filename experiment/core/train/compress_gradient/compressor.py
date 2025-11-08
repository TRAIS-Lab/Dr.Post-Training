"""
Compression container classes for two-stage tensor compression.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import logging

from torch import Tensor
from .projection import random_project, AbstractProjector, ProjectionType

# Configure logger
logger = logging.getLogger(__name__)


class ProjectionContainer(ABC):
    """
    Abstract base class for compression containers.
    """
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    @abstractmethod
    def forward(self, input: Any, scale: str = "forward") -> Any:
        pass

    @abstractmethod
    def transpose(self, input: Any, scale: str = "forward") -> Any:
        pass

    @abstractmethod
    def refresh(self, step: int) -> Any:
        pass


class Sparsifier(ProjectionContainer):
    """
    Container for sparsification (first stage of two-step compression).

    Sparsification is always factorized, operating component-wise:
        Pv = (P_1 ⊗ P_2) (v_1 ⊗ v_2) = (P_1 v_1) ⊗ (P_2 v_2),
    with P_1 in R^{d_1 * k_1'}, P_2 in R^{d_2 * k_2'}, and v in R^{d_1 * d_2}.
    """
    def __init__(self, name: str, index: int, update_freq: int = 200):
        super().__init__(name, index)
        # Sparsifier functions (always factorized)
        self.sparsifier_comp: Tuple[AbstractProjector, AbstractProjector] = (None, None)

        # Dimensions after sparsification
        self.intermediate_dims = None  # (k_1', k_2')

        self.update_freq = update_freq
        self.base_seed = None
        self.current_step = 0

    def forward(self, input: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], scale: str = "forward") -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply sparsification forward pass with automatic mode detection.

        This method supports two input types:

        Mode 1 - Factorized Sparsification (tuple input):
            Input: (v_1, v_2) - input components [batch, d_1] and [batch, d_2]
            Output: (v_1_sparse, v_2_sparse) - sparsified components [batch, k_1'] and [batch, k_2']
            Use case: During gradient computation, apply sparsifiers to each component

        Mode 2 - Sparsification (tensor input):
            Input: v - full input [1, d_1 * d_2]
            Output: v_sparse - sparsified vector [1, k_1' * k_2']
            Use case: During optimizer state transformation, apply to general state vector

        Args:
            input: Either (Mode 1) a tuple (v_1, v_2) or (Mode 2) a tensor v
            scale: Scaling mode for norm preservation
                - "forward": Scale to preserve forward projection norm
                - "backward": Scale to preserve backward projection norm

        Returns:
            Either a tuple (v_1_sparse, v_2_sparse) or a tensor v_sparse
        """
        # Mode detection based on input type
        if isinstance(input, tuple):
            # MODE 1: component-wise
            return self._forward_components(input, scale=scale)
        else:
            # MODE 2: via vec-trick
            return self._forward_full(input, scale=scale)

    def _forward_components(self, input_components: Tuple[torch.Tensor, torch.Tensor], scale: str = "forward") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparsifiers to input components (Mode 1).

        Args:
            input_components: (component_1, component_2) - tuple of input components
            scale: Scaling mode passed to projectors

        Returns:
            (component_1_sparse, component_2_sparse) - sparsified components
        """
        component_1, component_2 = input_components
        sparsifier_1, sparsifier_2 = self.sparsifier_comp

        # Apply sparsifiers component-wise, passing through scale parameter
        component_1_sparse = sparsifier_1.project(component_1, ensemble_id=0, scale=scale)
        component_2_sparse = sparsifier_2.project(component_2, ensemble_id=0, scale=scale)

        return component_1_sparse, component_2_sparse

    def _forward_full(self, input: torch.Tensor, scale: str = "forward") -> torch.Tensor:
        """
        Apply sparsification to the full input using vec-trick (Mode 2).

        This method handles arbitrary tensor inputs. Unlike forward_components(), this operates
        on a flattened vector without assuming specific outer-product structure.

        Uses the vec-trick dual identity:
            (A ⊗ B) vec(C) = vec(B C A^T)

        where A = S_2, B = S_1, C = matrix(input), S_i are sparsifiers

        Args:
            input: Full input vector [batch, d_1 * d_2]
            scale: Scaling mode passed to projectors

        Returns:
            Sparsified vector [batch, k_1' * k_2']
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)  # Handle single vector case

        batch_size = input.shape[0]  # Get batch size

        if self.intermediate_dims is None:
            raise ValueError(f"Cannot forward sparsifier {self.name}: dimensions not set")

        k_1_prime, k_2_prime = self.intermediate_dims

        sparsifier_1, sparsifier_2 = self.sparsifier_comp
        d_1 = sparsifier_1.feature_dim
        d_2 = sparsifier_2.feature_dim

        # Reshape from flattened to batched 2D matrix: [batch, d_1 * d_2] -> [batch, d_1, d_2]
        input_matrix = input.reshape(batch_size, d_1, d_2)

        # OPTIMIZED PATH: random_mask sparsification uses gather
        if sparsifier_1.proj_type == ProjectionType.random_mask and sparsifier_2.proj_type == ProjectionType.random_mask:
            indices_1 = sparsifier_1.active_indices
            indices_2 = sparsifier_2.active_indices

            # Batched advanced indexing for 2D gather
            # Select [b, indices_1, indices_2] for all b in B
            intermediate = input_matrix[:, indices_1[:, None], indices_2[None, :]]

            # Flatten batch items: [batch, k_1', k_2'] -> [batch, k_1' * k_2']
            return intermediate.reshape(batch_size, -1)

        # GENERAL PATH: dense projections use vec-trick dual: vec(P1 @ C @ P2.T)

        # 1. Apply second sparsifier forward (P2)
        #    Input: [batch, d_1, d_2] -> [batch * d_1, d_2]
        temp = sparsifier_2.project(input_matrix.reshape(batch_size * d_1, d_2), ensemble_id=0, scale=scale)
        #    Output: [batch * d_1, k_2'] -> [batch, d_1, k_2']
        temp = temp.reshape(batch_size, d_1, k_2_prime)

        # 2. Transpose for next operation: [batch, d_1, k_2'] -> [batch, k_2', d_1]
        temp_T = temp.transpose(1, 2)

        # 3. Apply first sparsifier forward (P1)
        #    Input: [batch, k_2', d_1] -> [batch * k_2', d_1]
        intermediate_T = sparsifier_1.project(temp_T.reshape(batch_size * k_2_prime, d_1), ensemble_id=0, scale=scale)
        #    Output: [batch * k_2', k_1'] -> [batch, k_2', k_1']
        intermediate_T = intermediate_T.reshape(batch_size, k_2_prime, k_1_prime)

        # 4. Transpose back to standard form: [batch, k_2', k_1'] -> [batch, k_1', k_2']
        intermediate = intermediate_T.transpose(1, 2)

        # 5. Flatten batch items: [batch, k_1', k_2'] -> [batch, k_1' * k_2']
        return intermediate.reshape(batch_size, -1)

    def transpose(self, input: torch.Tensor, scale: str = "forward") -> torch.Tensor:
        """
        This implements the transpose of the sparsification operation, mapping from
        the intermediate space back to the full space.

        Supports batching (input shape [batch, k']).

        Args:
            input: tensor [batch, k_1' * k_2']
            scale: Scaling mode passed to projectors

        Returns:
            Full tensor after sparsification transpose [batch, d_1 * d_2]
        """
        if self.intermediate_dims is None:
            raise ValueError(f"Cannot transpose sparsifier {self.name}: dimensions not set")

        k_1_prime, k_2_prime = self.intermediate_dims
        expected_size = k_1_prime * k_2_prime

        if input.dim() == 1:
            input = input.unsqueeze(0)  # Handle single vector case [k'] -> [1, k']

        batch_size = input.shape[0] # Get batch size
        actual_size = input.shape[1]

        if actual_size != expected_size:
            raise ValueError(
                f"Sparsifier.transpose dimension mismatch for {self.name}: "
                f"expected [batch, {expected_size}] but got {input.shape}. "
                f"intermediate_dims=({k_1_prime}, {k_2_prime})"
            )

        sparsifier_1 = self.sparsifier_comp[0]
        sparsifier_2 = self.sparsifier_comp[1]

        d_1 = sparsifier_1.feature_dim
        d_2 = sparsifier_2.feature_dim

        # Reshape to batched 2D matrix: [batch, k_1' * k_2'] -> [batch, k_1', k_2']
        intermediate = input.reshape(batch_size, k_1_prime, k_2_prime)

        # OPTIMIZED PATH: random_mask sparsification uses scatter
        if sparsifier_1.proj_type == ProjectionType.random_mask and sparsifier_2.proj_type == ProjectionType.random_mask:
            indices_1 = sparsifier_1.active_indices
            indices_2 = sparsifier_2.active_indices

            full_matrix = torch.zeros(batch_size, d_1, d_2, device=input.device, dtype=input.dtype)

            # Batched scatter using advanced indexing:
            # full_matrix[b, indices_1, indices_2] = intermediate[b]
            full_matrix[:, indices_1[:, None], indices_2[None, :]] = intermediate

            # Flatten and return: [batch, d_1 * d_2]
            return full_matrix.reshape(batch_size, -1)

        # GENERAL PATH: dense projections use vec-trick with batched operations
        # 1. Reshape intermediate for component-wise transpose: [batch, k_1', k_2'] -> [batch, k_2', k_1']
        intermediate_T = intermediate.transpose(1, 2)

        # 2. Apply sparsifier_1 transpose
        #    Input: [batch * k_2', k_1']
        temp_T = sparsifier_1.transpose(intermediate_T.reshape(batch_size * k_2_prime, k_1_prime), ensemble_id=0, scale=scale)
        #    Output: [batch * k_2', d_1] -> [batch, k_2', d_1]
        temp_T = temp_T.reshape(batch_size, k_2_prime, d_1)

        # 3. Transpose back: [batch, k_2', d_1] -> [batch, d_1, k_2']
        temp = temp_T.transpose(1, 2)

        # 4. Apply sparsifier_2 transpose
        #    Input: [batch * d_1, k_2']
        full_matrix = sparsifier_2.transpose(temp.reshape(batch_size * d_1, k_2_prime), ensemble_id=0, scale=scale)
        #    Output: [batch * d_1, d_2] -> [batch, d_1, d_2]
        full_matrix = full_matrix.reshape(batch_size, d_1, d_2)

        # 5. Flatten and return: [batch, d_1 * d_2]
        return full_matrix.reshape(batch_size, -1)


    def refresh(self, step: int) -> Sparsifier:
        """
        Check if sparsifier should be refreshed and refresh if needed.

        Similar to GaLore's check: `if iter % self.update_freq == 0`

        Args:
            step: Current training step

        Returns:
            Old Sparsifier if refresh occurred (for state transformation), None otherwise
        """
        self.current_step = step

        # Check if refresh is needed
        if step <= 0 or step % self.update_freq != 0:
            return None

        if self.sparsifier_comp == (None, None):
            logger.warning(f"No sparsifiers for layer {self.name}, skipping refresh")
            return None

        # Create temporary container with old sparsifier state (shallow copy for minimal memory)
        old_container = Sparsifier(self.name, self.index, self.update_freq)
        old_container.sparsifier_comp = self.sparsifier_comp  # Share reference temporarily
        old_container.intermediate_dims = self.intermediate_dims

        # Calculate refresh seed
        refresh_epoch = self.current_step // self.update_freq
        base_refresh_seed = self.base_seed + int(1e6) * refresh_epoch

        # Refresh first sparsifier
        if self.sparsifier_comp[0] is not None:
            sparsifier_1 = self.sparsifier_comp[0]
            sparsifier_1.refresh(base_refresh_seed)

        # Refresh second sparsifier (use different seed)
        if self.sparsifier_comp[1] is not None:
            sparsifier_2 = self.sparsifier_comp[1]
            sparsifier_2.refresh(base_refresh_seed + 1)

        return old_container

class Projector(ProjectionContainer):
    """
    Container for projection functions (second stage of two-step compression).

    Projection is a simple, one-step projection:
        Pv = P v.
    with P in R^{k * k'} and v in R^{k'}.
    """
    def __init__(self, name: str, index: int, update_freq: int = 200):
        super().__init__(name, index)
        self.projector: AbstractProjector = None

        self.update_freq = update_freq
        self.base_seed = None
        self.current_step = 0

    def forward(self, input: Tensor, scale: str = "forward") -> Tensor:
        """
        This applies the final projection to the intermediate tensor (after sparsification).

        Args:
            input: batch vector [batch, k'] after sparsification
            scale: Scaling mode passed to projector

        Returns:
            Compressed batch vector [batch, k] after projection
        """
        return self.projector.project(input, ensemble_id=0, scale=scale)

    def transpose(self, input: Tensor, scale: str = "forward") -> Tensor:
        """
        This applies the transpose of the projection to map back to the intermediate space.

        Args:
            input: Compressed batch vector [batch, k]
            scale: Scaling mode passed to projector

        Returns:
            Intermediate tensor after projection transpose [batch, k']
        """
        return self.projector.transpose(input, ensemble_id=0, scale=scale)

    def refresh(self, step: int) -> Projector:
        """
        Check if projector should be refreshed and refresh if needed.

        Similar to GaLore's check: `if iter % self.update_freq == 0`

        Args:
            step: Current training step

        Returns:
            Old Projector if refresh occurred (for state transformation), None otherwise
        """
        self.current_step = step

        # Check if refresh is needed
        if step <= 0 or step % self.update_freq != 0:
            return None

        if not hasattr(self, 'projector') or self.projector is None:
            logger.warning(f"No projector for layer {self.name}, skipping refresh")
            return None

        # Create temporary container with old projector (shallow copy for minimal memory)
        # This will be used for state transformation and then discarded
        old_container = Projector(self.name, self.index, self.update_freq)
        old_container.projector = self.projector  # Share reference temporarily

        # Calculate refresh seed
        refresh_epoch = self.current_step // self.update_freq
        refresh_seed = self.base_seed + int(1e6) * refresh_epoch

        # Get projector object and call its refresh method
        # This modifies self.projector in place
        proj_obj = self.projector
        proj_obj.refresh(refresh_seed)
        return old_container


class Compressor(ProjectionContainer):
    """
    Unified container that encapsulates the full two-stage compression pipeline.
        Forward:   input → Sparsifier → Projector → compressed
        Transpose: compressed → Projector^T → Sparsifier^T → full
    """

    def __init__(self, name: str, index: int, update_freq: int = 200):
        super().__init__(name, index)
        self.sparsifier: Sparsifier = None
        self.projector: Projector = None

        self.update_freq = update_freq
        self.base_seed = None
        self.current_step = 0

    def forward(self, input: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], scale: str = "forward") -> torch.Tensor:
        """
        Apply full compression pipeline: input → sparsifier → Kronecker product → projector → output

        It supports two input modes:
        1. Component mode (tuple input): (component1, component2)
           - Components can be 2D [batch, features] or 3D [batch, seq, features]
           - Applies sparsification to each component, then computes Kronecker product
           - Applies projection
           - Example use: Gradient compression where components are (grad_output, input)

        2. Direct mode (tensor input): [batch, d]
           - Input is a batch of full vectors
           - Applies sparsification via vec-trick
           - Applies projection
           - Example use: Optimizer state transformation during refresh

        Args:
            input: Either a tuple of batch components or batch full vectors
            scale: Scaling mode. Forward preserves forward projection's norm, backward preserves backward projection's norm

        Returns:
            Compressed output tensor
        """
        if isinstance(input, tuple):
            # MODE 1: Component-based compression (Kronecker product)
            component1, component2 = input

            # Detect if components are 3D or 2D
            is_3d = component2.dim() == 3

            if is_3d:
                # Extract dimensions for 3D case
                batch_size, seq_length, _ = component2.shape

                # Reshape to 2D for sparsification
                component1_2d = component1.reshape(-1, component1.shape[-1])
                component2_2d = component2.reshape(-1, component2.shape[-1])
            else:
                # Already 2D
                batch_size = component2.shape[0]
                component1_2d = component1
                component2_2d = component2

            # Stage 1: Apply sparsification to each component
            component1_sparse, component2_sparse = self.sparsifier.forward(
                (component1_2d, component2_2d), scale=scale
            )

            # Stage 1.5: Compute Kronecker product (outer product)
            if is_3d:
                # Reshape back to 3D for proper outer product computation
                component1_sparse_3d = component1_sparse.reshape(batch_size, seq_length, -1)
                component2_sparse_3d = component2_sparse.reshape(batch_size, seq_length, -1)

                # Compute outer product with batch scaling
                outer_product = torch.einsum(
                    'bsi,bsj->bij',
                    component1_sparse_3d * batch_size,
                    component2_sparse_3d
                )
            else:
                # Compute outer product for 2D case with batch scaling
                outer_product = torch.einsum(
                    'bi,bj->bij',
                    component1_sparse * batch_size,
                    component2_sparse
                )

            # Flatten the outer product result
            intermediate = outer_product.reshape(batch_size, -1)

            # Stage 2: Apply projection
            output = self.projector.forward(intermediate, scale=scale)

        else:
            # MODE 2: Full tensor compression (via vec-trick)
            # Stage 1: Apply sparsifier
            intermediate = self.sparsifier.forward(input, scale=scale)

            # Stage 2: Apply projector
            output = self.projector.forward(intermediate, scale=scale)
        return output

    def transpose(self, input: torch.Tensor, scale: str = "forward") -> torch.Tensor:
        """
        Apply full decompression pipeline: input → projector^T → sparsifier^T → output

        Note: Operations are applied in REVERSE order compared to forward.

        Args:
            input: Compressed vector [1, k]
            scale: Scaling mode passed to projectors

        Returns:
            Full (decompressed) tensor [1, d]
        """
        # Stage 1 (reverse): Apply projector transpose
        intermediate = self.projector.transpose(input, scale=scale)

        # Stage 2 (reverse): Apply sparsifier transpose
        output = self.sparsifier.transpose(intermediate, scale=scale)

        return output

    def refresh(self, step: int) -> Compressor:
        """
        Refresh both sparsifier and projector if needed.

        Args:
            step: Current training step

        Returns:
            Old Compressor with old components if refresh occurred, None otherwise
        """
        self.current_step = step

        # Check if refresh is needed
        if step <= 0 or step % self.update_freq != 0:
            return None

        # Refresh both components
        old_sparsifier = self.sparsifier.refresh(step)
        old_projector = self.projector.refresh(step)

        # If either component was refreshed, return old container
        old_container = Compressor(self.name, self.index, self.update_freq)

        old_container.sparsifier = old_sparsifier
        old_container.projector = old_projector

        # Copy other attributes
        old_container.base_seed = self.base_seed
        old_container.current_step = self.current_step

        return old_container


def setup_model_compressors(
    model: nn.Module,
    layer_names: List[str],
    sparsifier_kwargs: Optional[Dict[str, Any]] = None,
    projector_kwargs: Optional[Dict[str, Any]] = None,
    sample_inputs: Optional[Dict[str, Tensor]] = None,
    device: str = 'cpu',
    update_freq: int = 200
) -> List[Compressor]:
    """
    Sets up unified Compressors for each layer in the model.

    Each Compressor encapsulates both sparsification and projection stages.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set compressors for
        sparsifier_kwargs: Keyword arguments for sparsifier configuration (optional)
        projector_kwargs: Keyword arguments for projector configuration (optional)
        sample_inputs: Input batch to run a forward pass (optional)
        device: Device to run the model on
        update_freq: Number of steps between compressor refreshes (default: 200)

    Returns:
        List of Compressors, ordered by layer_names
    """
    if sample_inputs is None:
        return []

    # Initialize compressors list
    compressors = [None] * len(layer_names)
    if not (sparsifier_kwargs or projector_kwargs):
        return compressors

    # Create name to index mapping for faster lookup
    name_to_index = {name: idx for idx, name in enumerate(layer_names)}

    # Extract configuration parameters
    sparsifier_seed = sparsifier_kwargs.get('proj_seed', 0) if sparsifier_kwargs else 0
    projector_seed = projector_kwargs.get('proj_seed', 0) if projector_kwargs else 0

    # Remove parameters that are handled separately
    if sparsifier_kwargs:
        sparsifier_kwargs_copy = sparsifier_kwargs.copy()
        if 'proj_seed' in sparsifier_kwargs_copy:
            sparsifier_kwargs_copy.pop("proj_seed")
    else:
        sparsifier_kwargs_copy = {}

    if projector_kwargs:
        projector_kwargs_copy = projector_kwargs.copy()
        if 'proj_seed' in projector_kwargs_copy:
            projector_kwargs_copy.pop("proj_seed")
    else:
        projector_kwargs_copy = {}

    # Ensure model is on the correct device before running forward pass
    original_device = next(model.parameters()).device
    if str(original_device) != device:
        model.to(device)

    # Use no_grad to avoid autograd issues during setup
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            inputs = {k: v.to(device) for k, v in sample_inputs.items()}
            model(**inputs)
        else:
            inputs = sample_inputs[0].to(device)
            model(inputs)

    # First, capture inputs and outputs for each layer
    layer_inputs = {}
    layer_outputs = {}
    hooks = []

    def capture_hook(name, mod, inp, out):
        layer_inputs[name] = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
        layer_outputs[name] = out

    # Register temporary hooks to capture layer I/O
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(lambda mod, inp, out, n=name: capture_hook(n, mod, inp, out))
            hooks.append(hook)

    # Run another forward pass to capture inputs/outputs
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            model(**inputs)
        else:
            model(inputs)

    # Remove temporary hooks
    for hook in hooks:
        hook.remove()

    # Create sparsifiers and projectors for each layer
    module_list = list(model.named_modules())

    for module_id, (module_name, module) in enumerate(module_list):
        if module_name in layer_names:
            idx = name_to_index[module_name]

            # Create unified compressor container
            compressor = Compressor(module_name, idx, update_freq=update_freq)

            # Create sparsifier (ALWAYS factorized) - Stage 1
            sparsifier = Sparsifier(module_name, idx, update_freq=update_freq)
            base_seed = sparsifier_seed + int(1e4) * module_id
            sparse_kwargs = sparsifier_kwargs_copy.copy()

            # Create appropriate sparsifiers based on layer type
            # Sparsifiers are ALWAYS factorized
            if isinstance(module, nn.Linear):
                _setup_linear_sparsifier(
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    sparse_kwargs
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_sparsifier(
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    sparse_kwargs
                )
            else:
                raise ValueError(f"Unsupported layer type: {type(module)}")

            # Check if sparsifier was successfully initialized
            if sparsifier.sparsifier_comp == (None, None):
                logger.warning(f"Skipping layer {module_name}: sparsifier setup failed (likely missing inputs/outputs)")
                continue

            # Store base seed for refresh
            sparsifier.base_seed = base_seed

            # Create projector (ALWAYS non-factorized, operates after sparsification) - Stage 2
            projector = Projector(module_name, idx, update_freq=update_freq)
            base_seed = projector_seed + int(1e4) * module_id + 1
            proj_kwargs = projector_kwargs_copy.copy()

            # Set up projectors for sparsified dimensions
            # Projectors are ALWAYS non-factorized and operate AFTER sparsification
            if isinstance(module, nn.Linear):
                _setup_linear_projector(
                    projector,
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs
                )
            elif isinstance(module, nn.LayerNorm):
                _setup_layernorm_projector(
                    projector,
                    sparsifier,
                    module,
                    layer_inputs.get(module_name),
                    layer_outputs.get(module_name),
                    base_seed,
                    proj_kwargs
                )
            else:
                raise ValueError(f"Unsupported layer type: {type(module)}")

            # Check if projector was successfully initialized
            if projector.projector is None:
                logger.warning(f"Skipping layer {module_name}: projector setup failed")
                continue

            # Store base seed for refresh
            projector.base_seed = base_seed

            # Assemble compressor container with both stages
            compressor.sparsifier = sparsifier
            compressor.projector = projector
            compressor.base_seed = sparsifier.base_seed  # Use sparsifier's seed as base
            compressor.current_step = 0

            compressors[idx] = compressor

    return compressors


def _setup_linear_sparsifier(
    sparsifier: Sparsifier,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up sparsifier for a Linear layer.

    STRICT CONVENTION:
    - Sparsifier MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        container: Sparsifier to store the sparsifier functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3

    input_features = layer_input
    if layer.bias is not None:
        if is_3d:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features = input_features.reshape(-1, hidden_size)
        else:
            batch_size = input_features.shape[0]

        ones = torch.ones(input_features.size(0), 1,
                         device=input_features.device,
                         dtype=input_features.dtype)
        input_features = torch.cat([input_features, ones], dim=1)

        if is_3d:
            input_features = input_features.reshape(batch_size, seq_length, -1)

    # Sparsifiers are ALWAYS factorized
    sample_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_1 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_1 == -1:
        proj_dim_comp_1 = sample_comp_1.shape[-1]

    sparsifier_comp_1 = random_project(
        sample_comp_1,
        sample_comp_1.shape[0],
        proj_dim=proj_dim_comp_1,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    sample_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_2 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_2 == -1:
        proj_dim_comp_2 = sample_comp_2.shape[-1]

    sparsifier_comp_2 = random_project(
        sample_comp_2,
        sample_comp_2.shape[0],
        proj_dim=proj_dim_comp_2,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    test_comp_1 = sparsifier_comp_1.project(sample_comp_1[:1], ensemble_id=0)
    test_comp_2 = sparsifier_comp_2.project(sample_comp_2[:1], ensemble_id=0)
    sparsifier.intermediate_dims = (test_comp_1.shape[-1], test_comp_2.shape[-1])

    # Store factorized sparsifiers in sparsifier_comp
    sparsifier.sparsifier_comp = (sparsifier_comp_1, sparsifier_comp_2)


def _setup_linear_projector(
    projector: Projector,
    sparsifier: Sparsifier,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up projector for a Linear layer after sparsification.

    STRICT CONVENTION:
    - Projector MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in Sparsifier) are ALWAYS factorized

    Args:
        projector: Projector to store the projector
        sparsifier: Sparsifier with sparsification functions
        layer: Linear layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if pre_activation is None or layer_input is None:
        return

    batch_size = pre_activation.shape[0]
    is_3d = layer_input.dim() == 3

    # Extract sparsifier components
    sparsifier_comp_1, sparsifier_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    if is_3d:
        sample_pre_activation = pre_activation[:1, :1].reshape(-1, pre_activation.shape[-1])
        sparse_sample_pre_activation = sparsifier_comp_1.project(sample_pre_activation, ensemble_id=0)
        sparsified_dim_1 = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            batch_size, seq_length, hidden_size = input_features.shape
            input_features_with_bias = torch.cat([
                input_features,
                torch.ones(batch_size, seq_length, 1, device=input_features.device, dtype=input_features.dtype)
            ], dim=2)
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1, :1].reshape(-1, input_features_with_bias.shape[-1])
        sparse_sample_input_features = sparsifier_comp_2.project(sample_input_features, ensemble_id=0)
        sparsified_dim_2 = sparse_sample_input_features.shape[-1]
    else:
        sample_pre_activation = pre_activation[:1]
        sparse_sample_pre_activation = sparsifier_comp_1.project(sample_pre_activation, ensemble_id=0)
        sparsified_dim_1 = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            input_features_with_bias = torch.cat([
                input_features,
                torch.ones(input_features.size(0), 1, device=input_features.device, dtype=input_features.dtype)
            ], dim=1)
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1]
        sparse_sample_input_features = sparsifier_comp_2.project(sample_input_features, ensemble_id=0)
        sparsified_dim_2 = sparse_sample_input_features.shape[-1]

    # Create non-factorized projector (operates on flattened intermediate after sparsification)
    # Calculate dimension after sparsification
    intermediate_dim = sparsified_dim_1 * sparsified_dim_2

    sample_intermediate = torch.zeros(
        (batch_size, intermediate_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype
    )

    # For identity projection, use feature dimension instead of -1
    proj_dim = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim == -1:
        proj_dim = intermediate_dim

    projector_func = random_project(
        sample_intermediate,
        sample_intermediate.shape[0],
        proj_dim=proj_dim,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store in projector attribute
    projector.projector = projector_func


def _setup_layernorm_sparsifier(
    sparsifier: Sparsifier,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up sparsifier for a LayerNorm layer.

    STRICT CONVENTION:
    - Sparsifier MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        container: Sparsifier to store the sparsifier functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or layer_input is None:
        return

    # Sparsifiers are ALWAYS factorized
    sample_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_1 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_1 == -1:
        proj_dim_comp_1 = sample_comp_1.shape[-1]

    sparsifier_comp_1 = random_project(
        sample_comp_1,
        sample_comp_1.shape[0],
        proj_dim=proj_dim_comp_1,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    sample_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))

    # For identity projection, use feature dimension instead of -1
    proj_dim_comp_2 = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim_comp_2 == -1:
        proj_dim_comp_2 = sample_comp_2.shape[-1]

    sparsifier_comp_2 = random_project(
        sample_comp_2,
        sample_comp_2.shape[0],
        proj_dim=proj_dim_comp_2,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    test_comp_1 = sparsifier_comp_1.project(sample_comp_1[:1], ensemble_id=0)
    test_comp_2 = sparsifier_comp_2.project(sample_comp_2[:1], ensemble_id=0)
    sparsifier.intermediate_dims = (test_comp_1.shape[-1], test_comp_2.shape[-1])

    # Store factorized sparsifiers in sparsifier_comp
    sparsifier.sparsifier_comp = (sparsifier_comp_1, sparsifier_comp_2)


def _setup_layernorm_projector(
    projector: Projector,
    sparsifier: Sparsifier,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up projector for a LayerNorm layer after sparsification.

    STRICT CONVENTION:
    - Projector MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in Sparsifier) are ALWAYS factorized

    Args:
        projector: Projector to store the projector
        sparsifier: Sparsifier with sparsification functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or pre_activation is None:
        return

    # Apply sparsification to get correct dimensions for projector setup
    sparsifier_comp_1, sparsifier_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    sample_pre_activation = pre_activation[:1]
    sparse_sample_pre_activation = sparsifier_comp_1.project(sample_pre_activation, ensemble_id=0)

    # Create non-factorized projector (operates on concatenated sparsified components)
    # For LayerNorm: intermediate is concatenation of weight and bias components
    intermediate_dim = sparse_sample_pre_activation.shape[-1] * 2  # weight + bias

    sample_intermediate = torch.zeros(
        (pre_activation.shape[0], intermediate_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype
    )

    # For identity projection, use feature dimension instead of -1
    proj_dim = kwargs.get("proj_dim")
    if kwargs.get("proj_type") == "identity" and proj_dim == -1:
        proj_dim = intermediate_dim

    projector = random_project(
        sample_intermediate,
        sample_intermediate.shape[0],
        proj_dim=proj_dim,
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store in projector attribute
    projector.projector = projector