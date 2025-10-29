from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers.pytorch_utils import Conv1D

def replace_conv1d_modules(model):
    """
    GPT-2 is defined in terms of Conv1D. However, this does not work for EK-FAC.
    Here, we convert these Conv1D modules to linear modules recursively.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
    return model


class SubsetSampler(Sampler):
    """Samples elements from a predefined list of indices.

    Note that for training, the built-in PyTorch
    SubsetRandomSampler should be used. This class is for
    attributing process.
    """

    def __init__(self, indices: List[int]) -> None:
        """Initialize the sampler.

        Args:
            indices (list): A list of indices to sample from.
        """
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        """Get an iterator for the sampler.

        Returns:
            An iterator for the sampler.
        """
        return iter(self.indices)

    def __len__(self) -> int:
        """Get the number of indices in the sampler.

        Returns:
            The number of indices in the sampler.
        """
        return len(self.indices)

def setup_compression_kwargs(args, device):
    """Setup compression arguments for gradient compression methods."""
    if args.sparsification is None:
        sparsifier_kwargs = None
    else:
        sparsification_method, sparsification_dim = args.sparsification.split("-")
        if "*" in sparsification_dim:
            sparsification_factorize = True
            sparsification_dim = sparsification_dim.split("*")
            assert sparsification_dim[0] == sparsification_dim[1], "Sparsification dimension must be the same for factorized projection."

            sparsification_dim = int(sparsification_dim[0])
        else:
            sparsification_factorize = False
            sparsification_dim = int(sparsification_dim)

        sparsifier_kwargs = {
            "proj_dim": sparsification_dim,
            "proj_max_batch_size": 64,
            "proj_seed": args.seed,
            "proj_factorize": sparsification_factorize,
            "device": device,
            "method": sparsification_method,
            "use_half_precision": False,
        }

    if args.projection is None:
        projector_kwargs = {
            "proj_dim": -1,
            "proj_max_batch_size": -1,
            "proj_seed": args.seed,
            "proj_factorize": False,
            "device": device,
            "method": "Identity",
            "use_half_precision": False,
        }
    else:
        proj_method, proj_dim = args.projection.split("-")
        if "*" in proj_dim:
            proj_factorize = True
            proj_dim = proj_dim.split("*")
            assert proj_dim[0] == proj_dim[1], "Projection dimension must be the same for factorized projection."

            proj_dim = int(proj_dim[0])
        else:
            proj_factorize = False
            proj_dim = int(proj_dim)

        projector_kwargs = {
            "proj_dim": proj_dim,
            "proj_max_batch_size": 64,
            "proj_seed": args.seed,
            "proj_factorize": proj_factorize,
            "device": device,
            "method": proj_method,
            "use_half_precision": False,
        }

    return sparsifier_kwargs, projector_kwargs

def result_filename(args):
    if args.sparsification is not None:
        sparsification_name = args.sparsification
    else:
        sparsification_name = "NA"

    if args.projection is not None:
        projection_name = args.projection
    else:
        projection_name = "NA"

    training_setting = args.output_dir.split("/")[-1]
    result_filename = f"./results/{training_setting}/{args.baseline}/{args.tda}/{args.layer_type}/{sparsification_name}->{projection_name}.pt"

    return result_filename