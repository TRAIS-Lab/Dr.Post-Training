import torch
import logging
from typing import Dict, List, Optional
from torch import Tensor

# Configure logger
logger = logging.getLogger(__name__)


def vectorize(
    g: Dict[str, Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[Tensor] = None,
    device: Optional[str] = "cuda",
) -> Tensor:
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

def create_sample_inputs(
    tokenizer,
    max_seq_length: int = 512,
    device: str = 'cpu',
    sample_text: Optional[str] = None
) -> Dict[str, Tensor]:
    """
    Create sample inputs for initializing gradient compressors.

    This helper function creates a minimal batch of tokenized inputs that can be used
    to run a forward pass through the model to determine layer dimensions.

    Args:
        tokenizer: HuggingFace tokenizer instance
        max_seq_length: Maximum sequence length for tokenization
        device: Device to place the tensors on
        sample_text: Optional sample text to tokenize. If None, uses a default message.

    Returns:
        Dictionary containing tokenized inputs ready for model forward pass

    Example:
        >>> sample_inputs = create_sample_inputs(tokenizer, max_seq_length=512, device='cuda')
        >>> sparsifiers, projectors = setup_model_compressors(
        ...     model=model,
        ...     layer_names=layer_names,
        ...     sample_inputs=sample_inputs,
        ...     device='cuda'
        ... )
    """
    if sample_text is None:
        sample_text = "This is a sample text for compression initialization."

    sample_inputs = tokenizer(
        [sample_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    )

    # Move to device and add labels for language modeling
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    sample_inputs['labels'] = sample_inputs['input_ids'].clone()

    return sample_inputs


def greedy_selection(scores, interaction_matrix, k: int):
    """
    Select k data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - k: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    selected_indices = []

    for _ in range(k):
        idx_max = torch.argmax(scores).item()
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -float('inf')

    return selected_indices