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
from .compressor import Compressor
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

    def _transform_optimizer_state(
        self,
        state_old: torch.Tensor,
        layer_idx: int,
        old_compressor: Compressor,
        is_second_moment: bool = False
    ) -> torch.Tensor:
        """
        Transform optimizer state from old subspace to new subspace during compressor refresh.

        **For first moments (is_second_moment=False)**:
            m_new = M @ m_old where M = P_new @ P_old^T

        **For second moments (is_second_moment=True)**:
            Simply zero out - variance transformation is too complex and expensive.
            The optimizer will rebuild second moments from scratch.

        Args:
            state_old: Old optimizer state in compressed space [k_old]
            layer_idx: Index of the layer
            old_compressor: Old Compressor (before refresh) for transformation
            is_second_moment: If True, zero out; if False, use linear transformation

        Returns:
            Transformed state in new compressed space [k_new]
        """
        new_compressor = self.grad_hook.compressors[layer_idx]
        layer_name = new_compressor.name if hasattr(new_compressor, 'name') else f"layer_{layer_idx}"

        k_old = state_old.shape[0]
        norm_old = state_old.norm().item()

        moment_type = "2nd_moment" if is_second_moment else "1st_moment"
        print(f"[STATE TRANSFORM] {layer_name}: {moment_type}, k={k_old}, norm_old={norm_old:.4e}")

        # For second moments, simply zero out (too complex to transform correctly)
        if is_second_moment:
            # Determine k_new
            dummy = torch.zeros(1, k_old, device=state_old.device, dtype=state_old.dtype)
            dummy_full = old_compressor.transpose(dummy, scale="forward")
            dummy_new = new_compressor.forward(dummy_full, scale="forward")
            k_new = dummy_new.shape[1]

            state_new = torch.zeros(k_new, device=state_old.device, dtype=state_old.dtype)
            norm_new = 0.0

            print(f"  → ZEROED OUT: k={k_new}, norm_new={norm_new:.4e}")
            return state_new

        # For first moments, use simple linear transformation
        # Add batch dimension if needed [k] -> [1, k]
        if state_old.dim() == 1:
            state_old_batch = state_old.unsqueeze(0)
        else:
            state_old_batch = state_old

        # Transform: m_new = M @ m_old where M = P_new @ P_old^T
        full = old_compressor.transpose(state_old_batch, scale="backward")
        state_new_batch = new_compressor.forward(full, scale="backward")
        state_new = state_new_batch.squeeze(0)

        k_new = state_new.shape[0]
        norm_new = state_new.norm().item()

        print(f"  → TRANSFORMED: k={k_new}, norm_new={norm_new:.4e}, ratio={norm_new/norm_old if norm_old > 0 else 0:.4f}")
        return state_new

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

        num_refreshed, old_compressors = self.grad_hook.refresh_compressors(next_step)

        if num_refreshed > 0:
            print(f"Refreshed {num_refreshed} compressors at step {current_step} (for step {next_step})")

            # CRITICAL: Transform optimizer states after refresh
            # This stabilizes training by mapping states from old to new subspace
            # Pass old compressor containers temporarily for transformation, then discard
            self._transform_all_optimizer_states(old_compressors)

            # Old containers are no longer needed, they'll be garbage collected

            # Debug: Check which layers have states after transformation
            print(f"Checking optimizer states after transformation...")
            for group in self.param_groups:
                for p in group['params']:
                    layer_name, _ = self._get_layer_info(p)
                    if layer_name is not None:
                        state = self.state.get(p, {})
                        if 'exp_avg' in state and 'exp_avg_sq' in state:
                            is_compressed = layer_name in self.compressed_layer_names
                            has_nan = torch.isnan(state['exp_avg_sq']).any()
                            if has_nan:
                                logger.error(f"Layer {layer_name} (compressed={is_compressed}): exp_avg_sq has NaN!")

        return num_refreshed

    def _transform_all_optimizer_states(self, old_compressors: Compressor) -> int:
        """
        Transform all optimizer states after compressor refresh.

        This is called automatically after refresh to stabilize training by
        mapping first and second moments from the old subspace to the new one.

        Args:
            old_compressors: List of old CompressorContainers (before refresh)

        Returns:
            Number of layers with transformed states
        """
        num_transformed = 0
        num_no_state = 0
        num_no_old_compressor = 0
        num_compressed_layers = len(self.compressed_layer_names)

        for group in self.param_groups:
            for p in group['params']:
                # Get layer information
                layer_name, _ = self._get_layer_info(p)

                # Only transform compressed layers with existing states
                if layer_name is None or layer_name not in self.compressed_layer_names:
                    continue

                state = self.state.get(p, {})
                if len(state) == 0 or 'exp_avg' not in state:
                    num_no_state += 1
                    continue

                # Get layer index
                layer_idx = self.grad_hook.layer_name_to_idx[layer_name]

                # Check if old compressor is available
                old_compressor = old_compressors[layer_idx] if layer_idx < len(old_compressors) else None

                if old_compressor is None:
                    num_no_old_compressor += 1
                    continue

                try:
                    # Transform first moment
                    state['exp_avg'] = self._transform_optimizer_state(
                        state['exp_avg'],
                        layer_idx,
                        old_compressor,
                        is_second_moment=False
                    )

                    # Transform second moment
                    state['exp_avg_sq'] = self._transform_optimizer_state(
                        state['exp_avg_sq'],
                        layer_idx,
                        old_compressor,
                        is_second_moment=True
                    )

                    num_transformed += 1

                except Exception as e:
                    logger.error(f"Failed to transform state for layer {layer_name}: {e}")
                    # Continue with other layers even if one fails
                    continue

        if num_transformed > 0:
            print(f"Successfully transformed optimizer states for {num_transformed} layers")

        return num_transformed

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
                # Get layer information
                layer_name, param_type = self._get_layer_info(p)

                # Check if this parameter should use compressed optimization
                if layer_name is not None and layer_name in compressed_grads_dict:
                    # Use compressed gradient pathway
                    # Note: p.grad will be None for hooked layers (intentional)
                    self._step_compressed(p, compressed_grads_dict[layer_name], group, layer_name)
                else:
                    # Use standard gradient pathway (for non-compressed layers)
                    # Skip if gradient is None
                    if p.grad is None:
                        continue
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

        # Get compressor for this layer (unified API)
        layer_idx = self.grad_hook.layer_name_to_idx[layer_name]
        compressor = self.grad_hook.compressors[layer_idx] if layer_idx < len(self.grad_hook.compressors) else None

        # Check if we can use compressed states
        can_compress = (
            compressor is not None and
            hasattr(compressor, 'transpose')
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

        # Apply decompression using unified compressor API
        # compressor.transpose() handles both stages: projector^T then sparsifier^T
        full_update = compressor.transpose(compressed_update_batch, scale="backward")  # ĝ → ḡ [1, p_l]

        # Dimension check: full_update should match parameter size
        param_numel = param.numel()
        assert full_update.numel() == param_numel, \
            f"Full update size mismatch for {layer_name}: " \
            f"expected {param_numel}, got {full_update.numel()}"

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

        # Check gradient for NaN/Inf
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            logger.error(f"Standard AdamW: NaN/Inf in gradient! Skipping update.")
            return

        # Update biased first and second moment estimates
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Check states for NaN/Inf
        if torch.isnan(state['exp_avg_sq']).any() or torch.isinf(state['exp_avg_sq']).any():
            logger.error(f"Standard AdamW: NaN/Inf in exp_avg_sq after update! grad_norm={grad.norm():.2e}")
            logger.error(f"  exp_avg_sq: min={state['exp_avg_sq'].min():.2e}, max={state['exp_avg_sq'].max():.2e}")
            logger.error(f"  Resetting exp_avg_sq to avoid corruption")
            state['exp_avg_sq'].zero_()
            state['exp_avg_sq'].add_(grad.square())
            return

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
        grad_hook: GradientHook,
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
