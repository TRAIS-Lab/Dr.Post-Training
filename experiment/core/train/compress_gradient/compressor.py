"""
Projector container classes for gradient compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
if TYPE_CHECKING:
    from typing import Union

import torch
import torch.nn as nn
import logging

from torch import Tensor
from .projection import random_project
from .utils import create_sample_inputs

# Configure logger
logger = logging.getLogger(__name__)

class BaseContainer:
    """
    Base container for compression functions associated with a layer.

    Note: This is an abstract base. Use SparsifierContainer or ProjectorContainer.
    """
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def refresh(self, step: int) -> bool:
        """
        Refresh the compressor if needed.

        Args:
            step: Current training step
        Returns:
            True if refreshed, False otherwise
        """
        raise NotImplementedError("BaseContainer does not implement refresh()")

class SparsifierContainer(BaseContainer):
    """
    Container for sparsification (first stage of two-step compression).

    Sparsification is ALWAYS factorized, operating component-wise:
    - sparsifier_output: Acts on gradient w.r.t. pre-activation (d_out → k_out')
    - sparsifier_input: Acts on input features (d_in → k_in')

    Forward:  (g_out, g_in) → (g_out', g_in')  [via gather for random_mask or matrix mult for dense]
    Transpose: (g_out', g_in') → (ḡ_out, ḡ_in)  [via scatter for random_mask or vec-trick for dense]

    Attributes:
        sparsifier_comp: Tuple of (sparsifier_output, sparsifier_input) functions
        mask_indices: Tuple of (output_indices, input_indices) for random_mask transpose
        intermediate_dims: Tuple of (k_out', k_in') after sparsification
        original_dims: Tuple of (d_out, d_in) before sparsification
        update_compressor_freq: Number of steps between refreshes
        base_seed: Base random seed
        current_step: Current training step
    """
    def __init__(self, name: str, index: int, update_compressor_freq: int = 200):
        super().__init__(name, index)
        # Sparsifier functions (always factorized)
        self.sparsifier_comp = (None, None)  # (output_sparsifier, input_sparsifier)

        # Transpose-related attributes
        self.mask_indices = (None, None)  # (output_indices, input_indices) for mask
        self.intermediate_dims = None  # (k_out', k_in')

        # Refresh mechanism
        self.update_compressor_freq = update_compressor_freq
        self.base_seed = None
        self.current_step = 0

    def transpose(self, g_intermediate: torch.Tensor) -> torch.Tensor:
        """
        Apply sparsification transpose: g' → ḡ

        This implements the transpose of the sparsification operation, which is the
        first stage of the two-step compression (applied in REVERSE order during decompression).

        For random_mask: Uses optimized scatter operation based on mask indices.
        For dense projections: Uses vec-trick: vec((P_out)ᵀ @ G' @ P_in) where G' is [k_out', k_in']

        Args:
            g_intermediate: Intermediate gradient after projection transpose [1, k_out' * k_in']
                           (already aggregated, single sample)

        Returns:
            Full gradient after sparsification transpose [1, d_out * d_in]
        """
        if self.sparsifier_comp == (None, None):
            # No sparsification, return as-is
            return g_intermediate

        if g_intermediate is None:
            return None

        # Get dimensions
        if self.intermediate_dims is None:
            raise ValueError(f"Cannot transpose sparsifier {self.name}: dimensions not set")

        k_out_prime, k_in_prime = self.intermediate_dims

        # Ensure we have [1, k'] format
        if g_intermediate.dim() == 1:
            g_intermediate = g_intermediate.unsqueeze(0)

        # OPTIMIZED PATH: random_mask sparsification uses scatter
        if self.mask_indices is not None and self.mask_indices != (None, None):
            output_indices, input_indices = self.mask_indices

            # Get projector objects to access original dimensions
            output_sparsifier = self.sparsifier_comp[0]
            input_sparsifier = self.sparsifier_comp[1]

            output_proj = output_sparsifier.projector
            input_proj = input_sparsifier.projector

            # Get original dimensions from projector
            d_out = output_proj.feature_dim  # Original output dimension
            d_in = input_proj.feature_dim    # Original input dimension

            # Reshape from flattened to 2D matrix: [1, k_out' * k_in'] → [k_out', k_in']
            G_intermediate = g_intermediate.reshape(k_out_prime, k_in_prime)

            # Initialize full gradient matrix with zeros
            G_full = torch.zeros(d_out, d_in, device=g_intermediate.device, dtype=g_intermediate.dtype)

            # Scatter operation: G_full[output_indices[:, None], input_indices[None, :]] = G_intermediate
            # Use advanced indexing with meshgrid for 2D scatter
            output_idx_expanded = output_indices.unsqueeze(1).expand(k_out_prime, k_in_prime)
            input_idx_expanded = input_indices.unsqueeze(0).expand(k_out_prime, k_in_prime)

            # Flatten indices and values for 1D scatter
            flat_indices = output_idx_expanded * d_in + input_idx_expanded
            G_full.view(-1).scatter_(0, flat_indices.reshape(-1), G_intermediate.reshape(-1))

            # Flatten and add batch dimension: [1, d_out * d_in]
            result = G_full.reshape(1, -1)
            return result

        # GENERAL PATH: dense projections use vec-trick
        # Reshape from flattened to 2D matrix: [1, k_out' * k_in'] → [k_out', k_in']
        G_intermediate = g_intermediate.reshape(k_out_prime, k_in_prime)

        # Get projector objects from the sparsifier closures
        output_sparsifier = self.sparsifier_comp[0]
        input_sparsifier = self.sparsifier_comp[1]

        if not (hasattr(output_sparsifier, 'projector') and hasattr(input_sparsifier, 'projector')):
            raise ValueError(f"Sparsifier {self.name} does not have projector objects")

        output_proj = output_sparsifier.projector
        input_proj = input_sparsifier.projector

        d_out = output_proj.feature_dim
        d_in = input_proj.feature_dim

        # Apply transpose using vec-trick: vec((P_out)ᵀ @ G' @ P_in)
        # Step 1: Apply output projector transpose column-wise
        # Treat columns as batch: transpose [k_in', k_out'] where each column is a sample
        G_transposed = G_intermediate.t()  # [k_in', k_out']

        G_output_transposed = output_proj.transpose(G_transposed, ensemble_id=0)  # [k_in', d_out]

        G_output_transposed = G_output_transposed.t()  # [d_out, k_in']

        # Step 2: Apply input projector transpose row-wise
        # Treat rows as batch: [d_out, k_in'] where each row is a sample
        G_full = input_proj.transpose(G_output_transposed, ensemble_id=0)  # [d_out, d_in]

        # Flatten and add batch dimension: [1, d_out * d_in]
        result = G_full.reshape(1, -1)

        return result

    def refresh(self, step: int) -> bool:
        """
        Check if sparsifier should be refreshed and refresh if needed.

        Similar to GaLore's check: `if iter % self.update_compressor_freq == 0`

        Args:
            step: Current training step
            device: Device to place tensors on

        Returns:
            True if sparsifier was refreshed, False otherwise
        """
        self.current_step = step

        # Check if refresh is needed
        if step <= 0 or step % self.update_compressor_freq != 0:
            return False

        if self.sparsifier_comp == (None, None):
            logger.warning(f"No sparsifiers for layer {self.name}, skipping refresh")
            return False

        # Calculate refresh seed
        refresh_epoch = self.current_step // self.update_compressor_freq
        base_refresh_seed = self.base_seed + int(1e6) * refresh_epoch

        try:
            # Refresh output sparsifier
            if self.sparsifier_comp[0] is not None:
                output_sparsifier = self.sparsifier_comp[0]
                output_proj = output_sparsifier.projector
                output_proj.refresh(base_refresh_seed)

            # Refresh input sparsifier (use different seed)
            if self.sparsifier_comp[1] is not None:
                input_sparsifier = self.sparsifier_comp[1]
                input_proj = input_sparsifier.projector
                input_proj.refresh(base_refresh_seed + 1)

            return True

        except Exception as e:
            logger.error(f"Error refreshing sparsifier for {self.name}: {e}")
            return False

class ProjectorContainer(BaseContainer):
    """
    Container for projector functions associated with a layer.
    Used to store projectors without modifying the original layer.

    STRICT CONVENTION:
    - ProjectorContainer ALWAYS contains non-factorized projector (stored in self.projector)
    - Projectors operate AFTER outer product on flattened gradient
    - Projectors are NEVER factorized (use SparsifierContainer for factorized operations)

    Attributes:
        projector: Projection function (ALWAYS non-factorized)
        mask_indices: Tuple of (output_indices, input_indices) from sparsifier (for reference)
        intermediate_dims: Tuple of (k_out', k_in') after sparsification
        original_dims: Tuple of (d_out, d_in) original gradient dimensions
        update_compressor_freq: Number of steps between projector refreshes (similar to GaLore)
        base_seed: Base random seed for projector initialization
        current_step: Current training step (for tracking refresh)
    """
    def __init__(self, name: str, index: int, update_compressor_freq: int = 200):
        super().__init__(name, index)
        self.projector = None  # Non-factorized projector function

        # Refresh mechanism (similar to GaLore)
        self.update_compressor_freq = update_compressor_freq
        self.base_seed = None
        self.current_step = 0

    def transpose(self, compressed_grad: Tensor) -> Tensor:
        """
        Apply ONLY projection transpose: ĝ → g'

        This should NOT handle sparsification transpose. The sparsification transpose
        is handled by SparsifierContainer.transpose().

        According to the strict two-step architecture:
        - ProjectorContainer handles stage 2 transpose (projection)
        - SparsifierContainer handles stage 1 transpose (sparsification)

        Args:
            compressed_grad: Compressed gradient [1, k_l] (already aggregated)

        Returns:
            Intermediate gradient after projection transpose [1, k'_l]
        """
        if compressed_grad is None:
            return None

        # Delegate to projector's transpose method
        if self.projector is not None:
            proj_obj = self.projector.projector

            # Get dimensions for verification
            proj_dim = proj_obj.proj_dim
            feature_dim = proj_obj.feature_dim

            result = proj_obj.transpose(compressed_grad, ensemble_id=0)
            return result

        # No projection configured, return as-is (identity projection)
        return compressed_grad

    def refresh(self, step: int) -> bool:
        """
        Check if projector should be refreshed and refresh if needed.

        Similar to GaLore's check: `if iter % self.update_compressor_freq == 0`

        Args:
            step: Current training step
            device: Device to place tensors on

        Returns:
            True if projector was refreshed, False otherwise
        """
        self.current_step = step

        # Check if refresh is needed
        if step <= 0 or step % self.update_compressor_freq != 0:
            return False

        if not hasattr(self, 'projector') or self.projector is None:
            logger.warning(f"No projector for layer {self.name}, skipping refresh")
            return False

        # Calculate refresh seed
        refresh_epoch = self.current_step // self.update_compressor_freq
        refresh_seed = self.base_seed + int(1e6) * refresh_epoch

        try:
            # Get projector object and call its refresh method
            proj_obj = self.projector.projector
            proj_obj.refresh(refresh_seed)
            return True
        except Exception as e:
            logger.error(f"Error refreshing projector for {self.name}: {e}")
            return False


def setup_model_compressors(
    model: nn.Module,
    layer_names: List[str],
    sparsifier_kwargs: Optional[Dict[str, Any]] = None,
    projector_kwargs: Optional[Dict[str, Any]] = None,
    sample_inputs: Optional[Dict[str, Tensor]] = None,
    device: str = 'cpu',
    update_compressor_freq: int = 200
) -> Tuple[List[SparsifierContainer], List[ProjectorContainer]]:
    """
    Sets up sparsifiers and projectors for each layer in the model.

    Args:
        model: The PyTorch model
        layer_names: Names of layers to set projectors for
        sparsifier_kwargs: Keyword arguments for sparsifier configuration (optional)
        projector_kwargs: Keyword arguments for projector configuration (optional)
        sample_inputs: Input batch to run a forward pass (optional)
        device: Device to run the model on
        update_compressor_freq: Number of steps between projector refreshes (default: 200)

    Returns:
        Tuple of (sparsifiers, projectors) lists, ordered by layer_names
    """
    if sample_inputs is None:
        return [], []

    # Initialize containers lists
    sparsifiers = [None] * len(layer_names) if sparsifier_kwargs else []
    projectors = [None] * len(layer_names) if projector_kwargs else []
    if not (sparsifier_kwargs or projector_kwargs):
        return sparsifiers, projectors

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

            # Create sparsifier (ALWAYS factorized)
            sparsifier = SparsifierContainer(module_name, idx, update_compressor_freq=update_compressor_freq)
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
            sparsifiers[idx] = sparsifier

            # Create projector (ALWAYS non-factorized, operates after sparsification)
            projector = ProjectorContainer(module_name, idx, update_compressor_freq=update_compressor_freq)
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
                # Remove the sparsifier we just added since we need both
                sparsifiers[idx] = None
                continue

            # Store base seed for refresh
            projector.base_seed = base_seed
            projectors[idx] = projector

    return sparsifiers, projectors


def _setup_linear_sparsifier(
    container: SparsifierContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up sparsifier for a Linear layer.

    STRICT CONVENTION:
    - SparsifierContainer MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        container: SparsifierContainer to store the sparsifier functions
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
    dumb_grad_comp_1 = torch.zeros_like(pre_activation.view(-1, pre_activation.shape[-1]))

    projector_grad_comp_1 = random_project(
        dumb_grad_comp_1,
        dumb_grad_comp_1.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    dumb_grad_comp_2 = torch.zeros_like(input_features.view(-1, input_features.shape[-1]))
    projector_grad_comp_2 = random_project(
        dumb_grad_comp_2,
        dumb_grad_comp_2.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    # IMPORTANT: This also initializes active_indices for random_mask projectors
    test_out = projector_grad_comp_1(dumb_grad_comp_1[:1], ensemble_id=0)
    test_in = projector_grad_comp_2(dumb_grad_comp_2[:1], ensemble_id=0)
    container.intermediate_dims = (test_out.shape[-1], test_in.shape[-1])

    # Extract mask indices AFTER dry run (if using random_mask)
    # The dry run above initialized active_indices for ensemble_id=0
    if kwargs.get("proj_type") == "random_mask":
        # Extract from the created projectors BEFORE torch.compile
        # The projector function has a .projector attribute that contains the CudaProjector
        output_indices = projector_grad_comp_1.projector.active_indices
        input_indices = projector_grad_comp_2.projector.active_indices

        if output_indices is None or input_indices is None:
            logger.warning(f"Failed to initialize mask indices for {container.name} - will use dense path")
            container.mask_indices = (None, None)
        else:
            container.mask_indices = (output_indices, input_indices)
    else:
        # For dense projections, explicitly set to (None, None) to use vec-trick path
        container.mask_indices = (None, None)

    # Store factorized sparsifiers in sparsifier_comp
    container.sparsifier_comp = (
        torch.compile(projector_grad_comp_1),
        torch.compile(projector_grad_comp_2)
    )


def _setup_linear_projector(
    projector: ProjectorContainer,
    sparsifier: SparsifierContainer,
    layer: nn.Linear,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up projector for a Linear layer after sparsification.

    STRICT CONVENTION:
    - ProjectorContainer MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in SparsifierContainer) are ALWAYS factorized

    Args:
        projector: ProjectorContainer to store the projector
        sparsifier: SparsifierContainer with sparsification functions
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

    # Extract sparsifier components (using new naming convention)
    sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    if is_3d:
        sample_pre_activation = pre_activation[:1, :1].reshape(-1, pre_activation.shape[-1])
        sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)
        sparsified_output_dim = sparse_sample_pre_activation.shape[-1]

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
        sparse_sample_input_features = sparsifier_grad_comp_2(sample_input_features)
        sparsified_input_dim = sparse_sample_input_features.shape[-1]
    else:
        sample_pre_activation = pre_activation[:1]
        sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)
        sparsified_output_dim = sparse_sample_pre_activation.shape[-1]

        input_features = layer_input
        if layer.bias is not None:
            input_features_with_bias = torch.cat([
                input_features,
                torch.ones(input_features.size(0), 1, device=input_features.device, dtype=input_features.dtype)
            ], dim=1)
        else:
            input_features_with_bias = input_features

        sample_input_features = input_features_with_bias[:1]
        sparse_sample_input_features = sparsifier_grad_comp_2(sample_input_features)
        sparsified_input_dim = sparse_sample_input_features.shape[-1]

    # Create non-factorized projector (operates on flattened gradient after sparsification)
    # Calculate outer product dimension after sparsification
    gradient_dim = sparsified_output_dim * sparsified_input_dim

    dumb_grad_full = torch.zeros(
        (batch_size, gradient_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype
    )

    projector_func = random_project(
        dumb_grad_full,
        dumb_grad_full.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store in projector attribute
    projector.projector = torch.compile(projector_func)


def _setup_layernorm_sparsifier(
    container: SparsifierContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up sparsifier for a LayerNorm layer.

    STRICT CONVENTION:
    - SparsifierContainer MUST contain factorized sparsifiers
    - Sparsifiers operate BEFORE outer product on components

    Args:
        container: SparsifierContainer to store the sparsifier functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or layer_input is None:
        return

    # Sparsifiers are ALWAYS factorized
    dumb_grad_comp_1 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
    projector_grad_comp_1 = random_project(
        dumb_grad_comp_1,
        dumb_grad_comp_1.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    dumb_grad_comp_2 = torch.zeros((pre_activation.shape[0], pre_activation.shape[-1]))
    projector_grad_comp_2 = random_project(
        dumb_grad_comp_2,
        dumb_grad_comp_2.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed + 1,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store dimensions needed for transpose operation
    # Test projection to get actual intermediate dimensions
    # IMPORTANT: This also initializes active_indices for random_mask projectors
    test_out = projector_grad_comp_1(dumb_grad_comp_1[:1], ensemble_id=0)
    test_in = projector_grad_comp_2(dumb_grad_comp_2[:1], ensemble_id=0)
    container.intermediate_dims = (test_out.shape[-1], test_in.shape[-1])

    # Extract mask indices AFTER dry run (if using random_mask)
    # The dry run above initialized active_indices for ensemble_id=0
    if kwargs.get("proj_type") == "random_mask":
        # Extract from the created projectors BEFORE torch.compile
        # The projector function has a .projector attribute that contains the CudaProjector
        output_indices = projector_grad_comp_1.projector.active_indices
        input_indices = projector_grad_comp_2.projector.active_indices

        if output_indices is None or input_indices is None:
            logger.warning(f"Failed to initialize mask indices for {container.name} - will use dense path")
            container.mask_indices = (None, None)
        else:
            container.mask_indices = (output_indices, input_indices)
    else:
        # For dense projections, explicitly set to (None, None) to use vec-trick path
        container.mask_indices = (None, None)

    # Store factorized sparsifiers in sparsifier_comp
    container.sparsifier_comp = (
        torch.compile(projector_grad_comp_1),
        torch.compile(projector_grad_comp_2)
    )


def _setup_layernorm_projector(
    projector: ProjectorContainer,
    sparsifier: SparsifierContainer,
    layer: nn.LayerNorm,
    layer_input: Tensor,
    pre_activation: Tensor,
    base_seed: int,
    kwargs: Dict[str, Any]
) -> None:
    """
    Set up projector for a LayerNorm layer after sparsification.

    STRICT CONVENTION:
    - ProjectorContainer MUST contain non-factorized projectors
    - Projectors operate AFTER outer product on flattened gradient
    - Sparsifiers (in SparsifierContainer) are ALWAYS factorized

    Args:
        projector: ProjectorContainer to store the projector
        sparsifier: SparsifierContainer with sparsification functions
        layer: LayerNorm layer
        layer_input: Input tensor to the layer
        pre_activation: Output tensor from the layer
        base_seed: Base seed for random projection
        kwargs: Keyword arguments for the projection
    """
    if not layer.elementwise_affine or pre_activation is None:
        return

    # Apply sparsification to get correct dimensions for projector setup
    sparsifier_grad_comp_1, sparsifier_grad_comp_2 = sparsifier.sparsifier_comp

    # Get sample tensors to determine output dimensions of sparsifiers
    sample_pre_activation = pre_activation[:1]
    sparse_sample_pre_activation = sparsifier_grad_comp_1(sample_pre_activation)

    # Create non-factorized projector (operates on concatenated sparsified components)
    # For LayerNorm: gradient is concatenation of weight and bias gradients
    gradient_dim = sparse_sample_pre_activation.shape[-1] * 2  # weight + bias

    dumb_grad_full = torch.zeros(
        (pre_activation.shape[0], gradient_dim),
        device=pre_activation.device,
        dtype=pre_activation.dtype
    )

    projector_func = random_project(
        dumb_grad_full,
        dumb_grad_full.shape[0],
        proj_dim=kwargs.get("proj_dim"),
        proj_max_batch_size=kwargs.get("proj_max_batch_size"),
        proj_seed=base_seed,
        proj_type=kwargs.get("proj_type", "normal"),
        device=kwargs.get("device", "cpu")
    )

    # Store in projector attribute
    projector.projector = torch.compile(projector_func)