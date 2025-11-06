"""
General memory-efficient optimizer for compressed gradient training.

This optimizer works with any compression method that provides a transpose operation,
including GraSS, LoGra, GaLore, or custom projectors.

Key features:
- Optimizer states stored in compressed space
- Works with hook-based per-sample gradient computation
- Supports data selection (e.g., GREATS)
- Generic design: works with any projector with transpose method
"""

import torch
import math
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Callable
import logging

from .hook import GradientHook
logger = logging.getLogger(__name__)


class MeSOAdamW(Optimizer):
    """
    AdamW optimizer that maintains states in compressed gradient space.

    This is a general implementation that works with any compression method
    that provides a transpose operation (GraSS, LoGra, etc.).

    The optimizer:
    1. Receives compressed gradients from hooks
    2. Maintains first and second moments in compressed space
    3. Applies optimizer updates in compressed space
    4. Uses transpose to project updates back to parameter space

    Args:
        params: Model parameters to optimize (only Linear layer params will be optimized via compression)
        grad_hook: GradientHook instance for gradient compression and transpose
        lr: Learning rate
        betas: Coefficients for computing running averages of gradient and its square
        eps: Term added to denominator to improve numerical stability
        weight_decay: Weight decay coefficient (L2 penalty)
        compressed_layer_names: Optional list of layer names to apply compression.
                                If None, will attempt compression on all layers.
    """

    def __init__(
        self,
        params,
        grad_hook: GradientHook,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        compressed_layer_names: Optional[List[str]] = None
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.grad_hook = grad_hook
        self.compressed_layer_names = compressed_layer_names or grad_hook.layer_names

        # Create mapping from parameter to layer name
        self._param_to_layer_name = {}
        self._setup_param_mapping()

        logger.info(f"Initialized MeSOAdamW optimizer")
        logger.info(f"  Compressed layers: {len(self.compressed_layer_names)}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Betas: {betas}")
        logger.info(f"  Weight decay: {weight_decay}")

    def _setup_param_mapping(self):
        """
        Create mapping from parameters to layer names.

        This is crucial for knowing which compressed gradient corresponds to which parameter.
        """
        model = self.grad_hook.model

        for layer_name in self.compressed_layer_names:
            # Navigate to the module using layer_name
            module = model
            for attr in layer_name.split('.'):
                module = getattr(module, attr)

            # Map weight and bias parameters
            if hasattr(module, 'weight') and module.weight is not None:
                self._param_to_layer_name[id(module.weight)] = (layer_name, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                self._param_to_layer_name[id(module.bias)] = (layer_name, 'bias')

    def _aggregate_compressed_grads(self, selected_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate compressed gradients for selected samples.

        This extracts compressed gradients directly from the hook without decompressing them.

        Args:
            selected_indices: Optional list of sample indices to aggregate

        Returns:
            Dictionary mapping layer names to aggregated compressed gradients [1, k_l]
        """
        compressed_grads_dict = {}

        for idx, layer_name in enumerate(self.grad_hook.layer_names):
            compressed_grad = self.grad_hook.compressed_grads[idx]

            if compressed_grad is None:
                continue

            # Aggregate selected samples
            if selected_indices is not None and len(selected_indices) > 0:
                selected_compressed = compressed_grad[selected_indices]  # [|S|, k_l]
                aggregated_compressed = selected_compressed.mean(dim=0, keepdim=True)  # [1, k_l]
            else:
                # If no selection, use mean of all samples
                aggregated_compressed = compressed_grad.mean(dim=0, keepdim=True)  # [1, k_l]

            compressed_grads_dict[layer_name] = aggregated_compressed

        return compressed_grads_dict

    def _get_layer_info(self, param):
        """
        Get layer name and parameter type for a given parameter.

        Returns:
            tuple: (layer_name, param_type) or (None, None) if not found
        """
        param_id = id(param)
        return self._param_to_layer_name.get(param_id, (None, None))

    def get_current_step(self) -> int:
        """
        Get the current training step from optimizer state.

        Returns:
            Current step number (0 if no state exists yet)
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'step' in state:
                    return state['step']
        return 0

    def refresh_compressors_if_needed(self) -> int:
        """
        Refresh compressors if needed based on current step.

        This should be called BEFORE forward/backward passes to ensure
        gradients are computed with the correct (refreshed) projectors.

        Returns:
            Number of compressors refreshed

        Note:
            This method is preferred over automatic refresh in step() because
            it allows refresh to happen before gradient computation, matching
            GaLore's behavior.
        """
        current_step = self.get_current_step()

        # Add 1 because we want to refresh for the NEXT step
        next_step = current_step + 1

        num_refreshed = self.grad_hook.refresh_compressors(next_step)
        if num_refreshed > 0:
            logger.info(f"Refreshed {num_refreshed} compressors at step {current_step} (for step {next_step})")

        return num_refreshed

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, selected_indices: Optional[List[int]] = None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure to reevaluate the model and return the loss
            selected_indices: Optional list of sample indices to use (for data selection).
                            If None, uses all samples.

        Returns:
            Optional loss value if closure is provided

        Note:
            For correct refresh timing (matching GaLore), call refresh_compressors_if_needed()
            BEFORE your forward/backward passes in the training loop. This ensures gradients
            are computed with refreshed projectors.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Aggregate compressed gradients (don't decompress yet - we'll use them directly)
        compressed_grads_dict = self._aggregate_compressed_grads(selected_indices)

        # Update each parameter group
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get layer information
                layer_name, param_type = self._get_layer_info(p)

                # Check if this parameter should use compressed optimization
                if layer_name is not None and layer_name in compressed_grads_dict:
                    # Use compressed gradient pathway
                    self._step_compressed(p, compressed_grads_dict[layer_name], group, layer_name)
                else:
                    # Use standard gradient pathway (for non-compressed layers)
                    self._step_standard(p, group)

        return loss

    def _step_compressed(self, param, compressed_grad, group, layer_name):
        """
        Update parameter using compressed gradient optimization.

        This maintains optimizer states in compressed space and uses transpose
        to project updates back to parameter space.

        Args:
            param: Parameter tensor to update
            compressed_grad: Compressed gradient [1, k_l] - already aggregated
            group: Optimizer parameter group
            layer_name: Name of the layer
        """
        state = self.state[param]

        # Get projector and sparsifier for this layer
        layer_idx = self.grad_hook.layer_name_to_idx[layer_name]
        projector = self.grad_hook.projectors[layer_idx] if layer_idx < len(self.grad_hook.projectors) else None
        sparsifier = self.grad_hook.sparsifiers[layer_idx] if layer_idx < len(self.grad_hook.sparsifiers) else None

        # Check if we can use compressed states
        can_compress = (
            projector is not None and
            hasattr(projector, 'transpose')
        )

        if not can_compress:
            # Fallback to standard update if compression not available
            self._step_standard(param, group)
            return

        # Initialize state if needed
        if len(state) == 0:
            state['step'] = 0

            # Initialize compressed states (squeeze to remove batch dim)
            state['exp_avg'] = torch.zeros_like(compressed_grad.squeeze(0))  # First moment in compressed space [k_l]
            state['exp_avg_sq'] = torch.zeros_like(compressed_grad.squeeze(0))  # Second moment in compressed space [k_l]

        # Get hyperparameters
        beta1, beta2 = group['betas']
        state['step'] += 1

        # Update biased first and second moment estimates in compressed space
        # compressed_grad is already in compressed space [1, k_l], squeeze to [k_l]
        compressed_grad_vec = compressed_grad.squeeze(0)

        # Dimension assertion: ensure gradient matches state dimensions
        assert compressed_grad_vec.shape == state['exp_avg'].shape, \
            f"Compressed gradient dimension mismatch for {layer_name}: " \
            f"grad {compressed_grad_vec.shape} vs state {state['exp_avg'].shape}"

        state['exp_avg'].mul_(beta1).add_(compressed_grad_vec, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(
            compressed_grad_vec, compressed_grad_vec, value=1 - beta2
        )

        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Compute step in compressed space
        step_size = group['lr'] / bias_correction1
        denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        compressed_update = state['exp_avg'] / denom

        # Decompress update back to parameter space using transpose composition
        # Apply transpose in REVERSE order of compression:
        # Forward compression: g → (sparsify) → g' → (project) → ĝ
        # Backward transpose:  ĝ → (project^T) → g' → (sparsify^T) → ḡ

        # compressed_update is [k_l], need to add batch dim to make [1, k_l]
        if compressed_update.dim() == 1:
            compressed_update_batch = compressed_update.unsqueeze(0)
        else:
            compressed_update_batch = compressed_update

        # Dimension assertion: ensure batch dimension is correct
        assert compressed_update_batch.shape[0] == 1, \
            f"Compressed update batch dimension should be 1, got {compressed_update_batch.shape[0]}"
        assert compressed_update_batch.shape[1] == compressed_update.shape[-1], \
            f"Compressed update dimension mismatch after unsqueeze"

        # Step 1: Apply projection transpose (stage 2)
        g_intermediate = projector.transpose(compressed_update_batch)  # ĝ → g' [1, k']

        # Dimension check: g_intermediate should have batch dim = 1
        assert g_intermediate.shape[0] == 1, \
            f"Intermediate gradient batch dimension should be 1, got {g_intermediate.shape[0]}"

        # Step 2: Apply sparsification transpose (stage 1) if available
        if sparsifier and hasattr(sparsifier, 'transpose'):
            full_update = sparsifier.transpose(g_intermediate)  # g' → ḡ [1, p_l]

            # Dimension check: full_update should match parameter size
            param_numel = param.numel()
            assert full_update.numel() == param_numel, \
                f"Full update size mismatch for {layer_name}: " \
                f"expected {param_numel}, got {full_update.numel()}"
        else:
            full_update = g_intermediate

        # Apply weight decay (in parameter space)
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.add_(full_update.view_as(param), alpha=-step_size)

    def _step_standard(self, param, group):
        """
        Standard AdamW update for parameters without compression.

        Args:
            param: Parameter tensor to update
            group: Optimizer parameter group
        """
        grad = param.grad
        if grad is None:
            return

        state = self.state[param]

        # Initialize state if needed
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)

        # Get hyperparameters
        beta1, beta2 = group['betas']
        state['step'] += 1

        # Update biased first and second moment estimates
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        step_size = group['lr'] / bias_correction1

        # Compute update
        denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        # Apply weight decay
        if group['weight_decay'] != 0:
            param.mul_(1 - group['lr'] * group['weight_decay'])

        # Apply update
        param.addcdiv_(state['exp_avg'], denom, value=-step_size)

    def get_memory_stats(self):
        """
        Get memory statistics for optimizer states.

        Returns:
            dict: Memory statistics including compressed and standard state sizes
        """
        compressed_size = 0
        standard_size = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'exp_avg' in state:
                    size = state['exp_avg'].numel() * state['exp_avg'].element_size()
                    size += state['exp_avg_sq'].numel() * state['exp_avg_sq'].element_size()

                    # Check if this is compressed
                    layer_name, _ = self._get_layer_info(p)
                    if layer_name in self.compressed_layer_names:
                        compressed_size += size
                    else:
                        standard_size += size

        total_size = compressed_size + standard_size

        return {
            'compressed_size_mb': compressed_size / (1024 ** 2),
            'standard_size_mb': standard_size / (1024 ** 2),
            'total_size_mb': total_size / (1024 ** 2),
            'compression_ratio': (compressed_size + standard_size) / compressed_size if compressed_size > 0 else 1.0
        }


class MeSOSGD(Optimizer):
    """
    SGD optimizer with momentum that maintains states in compressed gradient space.

    Simpler variant of MeSOAdamW for comparison and debugging.
    """

    def __init__(
        self,
        params,
        grad_hook,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        compressed_layer_names: Optional[List[str]] = None
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.grad_hook = grad_hook
        self.compressed_layer_names = compressed_layer_names or grad_hook.layer_names

        # Create mapping from parameter to layer name
        self._param_to_layer_name = {}
        self._setup_param_mapping()

    def _setup_param_mapping(self):
        """Create mapping from parameters to layer names."""
        model = self.grad_hook.model

        for layer_name in self.compressed_layer_names:
            module = model
            for attr in layer_name.split('.'):
                module = getattr(module, attr)

            if hasattr(module, 'weight') and module.weight is not None:
                self._param_to_layer_name[id(module.weight)] = (layer_name, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                self._param_to_layer_name[id(module.bias)] = (layer_name, 'bias')

    def _get_layer_info(self, param):
        """Get layer name and parameter type for a given parameter."""
        param_id = id(param)
        return self._param_to_layer_name.get(param_id, (None, None))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None, selected_indices: Optional[List[int]] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Get decompressed (full) gradients from transpose
        # Note: SGD doesn't maintain states in compressed space, so we need full gradients
        full_grads_dict = self.grad_hook.get_decompressed_grads(selected_indices)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                layer_name, param_type = self._get_layer_info(p)

                if layer_name is not None and layer_name in full_grads_dict:
                    full_grad = full_grads_dict[layer_name]

                    # Apply weight decay
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Simple SGD update (no momentum for now, can add later)
                    p.add_(full_grad.view_as(p), alpha=-group['lr'])
                else:
                    # Standard update
                    grad = p.grad

                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])

                    p.add_(grad, alpha=-group['lr'])

        return loss
