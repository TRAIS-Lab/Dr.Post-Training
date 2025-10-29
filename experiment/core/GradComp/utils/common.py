"""
Common utility functions for gradient computation and influence attribution.
"""

import torch
import logging
from typing import Dict, List, Optional

# Configure logger
logger = logging.getLogger(__name__)

def stable_inverse(matrix: torch.Tensor, damping: float = None) -> torch.Tensor:
    """
    Compute a numerically stable inverse of a matrix using eigendecomposition.

    Args:
        matrix: Input matrix to invert
        damping: (Adaptive) Damping factor for numerical stability

    Returns:
        Stable inverse of the input matrix with the same dtype as input
    """
    orig_dtype = matrix.dtype
    matrix = matrix.to(dtype=torch.float32)

    assert matrix.dim() == 2, "Input must be a 2D matrix"

    # Add damping to the diagonal
    if damping is None:
        damping = 1e-5 * torch.trace(matrix) / matrix.size(0)
    else:
        damping = damping * torch.trace(matrix) / matrix.size(0)

    damped_matrix = matrix + damping * torch.eye(matrix.size(0), device=matrix.device)

    try:
        L = torch.linalg.cholesky(damped_matrix)
        inverse = torch.cholesky_inverse(L)
    except RuntimeError:
        logger.warning(f"Falling back to direct inverse due to Cholesky failure")
        inverse = torch.inverse(damped_matrix)

    return inverse.to(dtype=orig_dtype)


def vectorize(
    g: Dict[str, torch.Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[torch.Tensor] = None,
    device: Optional[str] = "cuda",
) -> torch.Tensor:
    """
    Vectorize gradients into a flattened tensor.

    This function takes a dictionary of gradients and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g: A dictionary containing gradient tensors to be vectorized
        batch_dim: Whether to include the batch dimension in the returned tensor
        arr: An optional pre-allocated tensor to store the vectorized gradients
        device: The device to store the tensor on

    Returns:
        A flattened tensor of gradients
    """
    if arr is None:
        if batch_dim:
            g_elt = g[next(iter(g.keys()))]
            batch_size = g_elt.shape[0]
            num_params = 0
            for param in g.values():
                if param.shape[0] != batch_size:
                    msg = "Parameter row num doesn't match batch size."
                    raise ValueError(msg)
                num_params += int(param.numel() / batch_size)
            arr = torch.empty(
                size=(batch_size, num_params),
                dtype=g_elt.dtype,
                device=device,
            )
        else:
            num_params = 0
            for param in g.values():
                num_params += int(param.numel())
            arr = torch.empty(size=(num_params,), dtype=param.dtype, device=device)

    pointer = 0
    vector_dim = 1
    for param in g.values():
        if batch_dim:
            if len(param.shape) <= vector_dim:
                num_param = 1
                p = param.data.reshape(-1, 1)
            else:
                num_param = param[0].numel()
                p = param.flatten(start_dim=1).data
            arr[:, pointer : pointer + num_param] = p.to(device)
            pointer += num_param
        else:
            num_param = param.numel()
            arr[pointer : pointer + num_param] = param.reshape(-1).to(device)
            pointer += num_param
    return arr

def get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, List[int]]:
    """Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size (int): The batch size. Each term (or module) in feature
            will have the same batch size.

    Returns:
        tuple[int, List[int]]: A tuple containing:
            - Maximum number of parameters per chunk
            - A list of the number of parameters in each chunk
    """
    param_shapes = torch.tensor(param_shape_list, dtype=torch.int64)

    max_chunk_size = torch.iinfo(torch.int32).max // (batch_size * 8)

    params_per_chunk = []
    chunk_sum = 0

    for ps in param_shapes:
        # If adding the current param exceeds the max size,
        # finalize the current chunk (if not empty) and start a new one.

        current_ps = ps

        if chunk_sum + current_ps >= max_chunk_size:
            if chunk_sum > 0:
                params_per_chunk.append(chunk_sum)
            chunk_sum = 0  # Reset for new chunk

        # Handle the case where a single param layer is
        # larger than the max_chunk_size by splitting it.
        while current_ps >= max_chunk_size:
            params_per_chunk.append(max_chunk_size)
            current_ps -= max_chunk_size

        # Add the (remainder of) the current param to the chunk
        chunk_sum += current_ps

    # Add the final chunk if it has any params
    if chunk_sum > 0:
        params_per_chunk.append(chunk_sum)

    # Handle edge case of no params
    if not params_per_chunk:
        params_per_chunk = [0]

    return max_chunk_size, params_per_chunk


def find_layers(model, layer_type="Linear", return_type="instance"):
    """
    Find layers of specified type in a model.

    Args:
        model: PyTorch model to search
        layer_type: Type of layer to find ('Linear', 'LayerNorm', or 'Linear_LayerNorm')
        return_type: What to return ('instance', 'name', or 'name_instance')

    Returns:
        List of layers, layer names, or (name, layer) tuples
    """
    layers = []
    return_module_name = not (return_type == "instance")

    if return_module_name:
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.Embedding):
                layers.append((module_name, module))
    else:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.Embedding):
                layers.append(module)

    if return_module_name:
        if layer_type == "Linear":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, torch.nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm))]
        elif layer_type == "LayerNorm":
            layers = [(name, layer) for name, layer in layers if isinstance(layer, torch.nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")
    else:
        if layer_type == "Linear":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
        elif layer_type == "Linear_LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.LayerNorm)]
        elif layer_type == "LayerNorm":
            layers = [layer for layer in layers if isinstance(layer, torch.nn.LayerNorm)]
        else:
            raise ValueError("Invalid setting now. Choose from 'Linear', 'LayerNorm', and 'Linear_LayerNorm'.")

    if return_type == "instance":
        return layers
    elif return_type == "name":
        return [name for name, layer in layers]
    elif return_type == "name_instance":
        return [(name, layer) for name, layer in layers]
    else:
        raise ValueError("Invalid return_type. Choose from 'instance', 'name', and 'name_instance'.")