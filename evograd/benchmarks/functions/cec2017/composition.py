"""
CEC 2017 Composition Functions (F21-F30).

Composition functions are weighted combinations of multiple basic or hybrid
functions, each with different shifts and rotations.

Reference: CEC 2017 Competition on Real-Parameter Single Objective Optimization

F21: Composition Function 1 (N=3)
F22: Composition Function 2 (N=3)
F23: Composition Function 3 (N=4)
F24: Composition Function 4 (N=4)
F25: Composition Function 5 (N=5)
F26: Composition Function 6 (N=5)
F27: Composition Function 7 (N=6)
F28: Composition Function 8 (N=6)
F29: Composition Function 9 (N=3) - Hybrid composition
F30: Composition Function 10 (N=3) - Hybrid composition
"""

from typing import List, Optional, Tuple, Callable

import torch
from torch import Tensor

from ..base import BenchmarkFunction
from . import basic
from . import data as cec_data
from . import hybrid as hybrid_module


class CEC2017CompositionFunction(BenchmarkFunction):
    """Base class for CEC 2017 composition functions."""
    
    def __init__(
        self,
        func_num: int,
        n_var: int = 10,
        n_components: int = 3,
        rotations: Optional[List[Tensor]] = None,
        shifts: Optional[List[Tensor]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize CEC 2017 composition function.
        
        Args:
            func_num: Function number (21-30).
            n_var: Number of variables.
            n_components: Number of component functions.
            rotations: Optional list of rotation matrices.
            shifts: Optional list of shift vectors.
            seed: Random seed for generating transforms if not provided.
        """
        super().__init__(n_var=n_var, xl=-100.0, xu=100.0)
        
        self.func_num = func_num
        self.bias = func_num * 100.0
        self.n_components = n_components
        
        # Load or generate transforms for each component
        if rotations is not None:
            self.rotations = rotations
        else:
            self.rotations = [
                cec_data.get_rotation_cf(func_num, i, n_var, seed=seed)
                for i in range(n_components)
            ]
        
        if shifts is not None:
            self.shifts = shifts
        else:
            self.shifts = [
                cec_data.get_shift_cf(func_num, i, n_var, seed=seed)
                for i in range(n_components)
            ]
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def _calc_weights(self, x: Tensor, sigmas: Tensor) -> Tensor:
        """
        Calculate composition weights based on distance from each shift.
        
        Args:
            x: Input tensor of shape [..., n_var].
            sigmas: Sigma values for each component of shape [n_components].
        
        Returns:
            Weights of shape [..., n_components].
        """
        batch_shape = x.shape[:-1]
        nx = x.shape[-1]
        n = self.n_components
        
        distances = []
        for i in range(n):
            shift = self.shifts[i].to(x.device, x.dtype)
            x_shifted = x - shift
            distances.append((x_shifted ** 2).sum(dim=-1))

        dist2 = torch.stack(distances, dim=-1)
        zero_mask = dist2 == 0
        zero_count = zero_mask.sum(dim=-1, keepdim=True)

        sigma_shape = (1,) * len(batch_shape) + (n,)
        sigma_values = sigmas.to(x.device, x.dtype).reshape(sigma_shape)
        safe_dist2 = dist2.clamp_min(torch.finfo(x.dtype).tiny)
        regular = safe_dist2.rsqrt() * torch.exp(
            -dist2 / (2.0 * nx * sigma_values.square())
        )
        regular_sum = regular.sum(dim=-1, keepdim=True)
        regular_weights = torch.where(
            regular_sum > 0,
            regular / regular_sum.clamp_min(torch.finfo(x.dtype).tiny),
            torch.full_like(regular, 1.0 / n),
        )

        # At a component centre the official finite-INF convention assigns all
        # mass to that component. Handle it directly to avoid inf / inf NaNs.
        centre_weights = zero_mask.to(x.dtype) / zero_count.clamp_min(1).to(x.dtype)
        weights = torch.where(zero_count > 0, centre_weights, regular_weights)
        
        return weights
    
    def _evaluate_composition(
        self,
        x: Tensor,
        funcs: List[Callable],
        sigmas: Tensor,
        lambdas: Tensor,
        biases: Tensor,
    ) -> Tensor:
        """
        Evaluate composition function.
        
        Args:
            x: Input tensor of shape [..., n_var].
            funcs: List of basic functions.
            sigmas: Sigma values for weight calculation.
            lambdas: Scaling factors for each component.
            biases: Bias values for each component.
        
        Returns:
            Function values.
        """
        batch_shape = x.shape[:-1]
        nx = x.shape[-1]
        n = len(funcs)
        
        # Move tensors to correct device
        sigmas = sigmas.to(x.device, x.dtype)
        lambdas = lambdas.to(x.device, x.dtype)
        biases = biases.to(x.device, x.dtype)
        
        # Calculate weights
        weights = self._calc_weights(x, sigmas)
        
        # Evaluate each component
        vals = torch.zeros(*batch_shape, n, device=x.device, dtype=x.dtype)
        
        for i in range(n):
            shift = self.shifts[i].to(x.device, x.dtype)
            rotation = self.rotations[i].to(x.device, x.dtype)
            
            # Transform: R @ (x - shift)
            z = cec_data.shift_rotate(x, shift, rotation)
            
            # Evaluate function
            vals[..., i] = funcs[i](z)
        
        # Weighted sum: sum(w * (lambda * f + bias))
        result = (weights * (lambdas * vals + biases)).sum(dim=-1)
        
        return result + self.bias


class CEC2017_F21(CEC2017CompositionFunction):
    """
    F21: Composition Function 1 (N=3)
    
    Components: Rosenbrock, High Conditioned Elliptic, Rastrigin
    
    Optimal value: F21* = 2100
    """
    name = "cec2017_f21"
    optimal_value = 2100.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=21, n_var=n_var, n_components=3, **kwargs)
        
        self.funcs = [
            basic.rosenbrock,
            basic.high_conditioned_elliptic,
            basic.rastrigin,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0])
        self.lambdas = torch.tensor([1.0, 1e-6, 1.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F22(CEC2017CompositionFunction):
    """
    F22: Composition Function 2 (N=3)
    
    Components: Rastrigin, Griewank, Modified Schwefel
    
    Optimal value: F22* = 2200
    """
    name = "cec2017_f22"
    optimal_value = 2200.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=22, n_var=n_var, n_components=3, **kwargs)
        
        self.funcs = [
            basic.rastrigin,
            basic.griewank,
            basic.modified_schwefel,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0])
        self.lambdas = torch.tensor([1.0, 10.0, 1.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F23(CEC2017CompositionFunction):
    """
    F23: Composition Function 3 (N=4)
    
    Components: Rosenbrock, Ackley, Modified Schwefel, Rastrigin
    
    Optimal value: F23* = 2300
    """
    name = "cec2017_f23"
    optimal_value = 2300.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=23, n_var=n_var, n_components=4, **kwargs)
        
        self.funcs = [
            basic.rosenbrock,
            basic.ackley,
            basic.modified_schwefel,
            basic.rastrigin,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self.lambdas = torch.tensor([1.0, 10.0, 1.0, 1.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F24(CEC2017CompositionFunction):
    """
    F24: Composition Function 4 (N=4)
    
    Components: Ackley, High Conditioned Elliptic, Griewank, Rastrigin
    
    Optimal value: F24* = 2400
    """
    name = "cec2017_f24"
    optimal_value = 2400.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=24, n_var=n_var, n_components=4, **kwargs)
        
        self.funcs = [
            basic.ackley,
            basic.high_conditioned_elliptic,
            basic.griewank,
            basic.rastrigin,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0, 40.0])
        self.lambdas = torch.tensor([1.0, 1e-6, 10.0, 1.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F25(CEC2017CompositionFunction):
    """
    F25: Composition Function 5 (N=5)
    
    Components: Rastrigin, Happy Cat, Ackley, Discus, Rosenbrock
    
    Optimal value: F25* = 2500
    """
    name = "cec2017_f25"
    optimal_value = 2500.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=25, n_var=n_var, n_components=5, **kwargs)
        
        self.funcs = [
            basic.rastrigin,
            basic.happy_cat,
            basic.ackley,
            basic.discus,
            basic.rosenbrock,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        self.lambdas = torch.tensor([10.0, 1.0, 10.0, 1e-6, 1.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0, 400.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F26(CEC2017CompositionFunction):
    """
    F26: Composition Function 6 (N=5)
    
    Components: Expanded Schaffer's F6, Modified Schwefel, Griewank, 
                Rosenbrock, Rastrigin
    
    Optimal value: F26* = 2600
    """
    name = "cec2017_f26"
    optimal_value = 2600.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=26, n_var=n_var, n_components=5, **kwargs)
        
        self.funcs = [
            basic.expanded_schaffers_f6,
            basic.modified_schwefel,
            basic.griewank,
            basic.rosenbrock,
            basic.rastrigin,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 20.0, 30.0, 40.0])
        # Note: Using lambdas from the actual code, not the problem definitions
        self.lambdas = torch.tensor([5e-4, 1.0, 10.0, 1.0, 10.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0, 400.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F27(CEC2017CompositionFunction):
    """
    F27: Composition Function 7 (N=6)
    
    Components: HGBat, Rastrigin, Modified Schwefel, Bent Cigar,
                High Conditioned Elliptic, Expanded Schaffer's F6
    
    Optimal value: F27* = 2700
    """
    name = "cec2017_f27"
    optimal_value = 2700.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=27, n_var=n_var, n_components=6, **kwargs)
        
        self.funcs = [
            basic.h_g_bat,
            basic.rastrigin,
            basic.modified_schwefel,
            basic.bent_cigar,
            basic.high_conditioned_elliptic,
            basic.expanded_schaffers_f6,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.lambdas = torch.tensor([10.0, 10.0, 2.5, 1e-26, 1e-6, 5e-4])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F28(CEC2017CompositionFunction):
    """
    F28: Composition Function 8 (N=6)
    
    Components: Ackley, Griewank, Discus, Rosenbrock, Happy Cat,
                Expanded Schaffer's F6
    
    Optimal value: F28* = 2800
    """
    name = "cec2017_f28"
    optimal_value = 2800.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=28, n_var=n_var, n_components=6, **kwargs)
        
        self.funcs = [
            basic.ackley,
            basic.griewank,
            basic.discus,
            basic.rosenbrock,
            basic.happy_cat,
            basic.expanded_schaffers_f6,
        ]
        self.sigmas = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.lambdas = torch.tensor([10.0, 10.0, 1e-6, 1.0, 1.0, 5e-4])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_composition(
            x, self.funcs, self.sigmas, self.lambdas, self.biases_comp
        )


class CEC2017_F29(CEC2017CompositionFunction):
    """
    F29: Composition Function 9 (N=3) - Hybrid Composition
    
    Components: Hybrid F15, Hybrid F16, Hybrid F17
    
    Optimal value: F29* = 2900
    """
    name = "cec2017_f29"
    optimal_value = 2900.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        seed = kwargs.get('seed', None)
        super().__init__(func_num=29, n_var=n_var, n_components=3, **kwargs)
        
        # Load shuffles for each component
        self.shuffles = [
            cec_data.get_shuffle_cf(29, i, n_var, seed=seed)
            for i in range(3)
        ]
        
        self.sigmas = torch.tensor([10.0, 30.0, 50.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0])
        # Offsets to subtract F* added at the end of hybrid functions
        self.offsets = torch.tensor([1500.0, 1600.0, 1700.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        nx = x.shape[-1]
        n = 3
        
        sigmas = self.sigmas.to(x.device, x.dtype)
        biases = self.biases_comp.to(x.device, x.dtype)
        offsets = self.offsets.to(x.device, x.dtype)
        
        # Calculate weights
        weights = self._calc_weights(x, sigmas)
        
        # Evaluate each hybrid function
        vals = torch.zeros(*batch_shape, n, device=x.device, dtype=x.dtype)
        
        hybrid_classes = [
            hybrid_module.CEC2017_F15,
            hybrid_module.CEC2017_F16,
            hybrid_module.CEC2017_F17,
        ]
        
        for i in range(n):
            # Create hybrid function with composition's rotation, shift, shuffle
            hybrid_func = hybrid_classes[i](
                n_var=nx,
                rotation=self.rotations[i],
                shift=self.shifts[i],
                shuffle=self.shuffles[i],
            )
            vals[..., i] = hybrid_func(x) - offsets[i]
        
        # Weighted sum
        result = (weights * (vals + biases)).sum(dim=-1)
        
        return result + self.bias


class CEC2017_F30(CEC2017CompositionFunction):
    """
    F30: Composition Function 10 (N=3) - Hybrid Composition
    
    Components: Hybrid F15, Hybrid F18, Hybrid F19
    
    Optimal value: F30* = 3000
    """
    name = "cec2017_f30"
    optimal_value = 3000.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        seed = kwargs.get('seed', None)
        super().__init__(func_num=30, n_var=n_var, n_components=3, **kwargs)
        
        # Load shuffles for each component
        self.shuffles = [
            cec_data.get_shuffle_cf(30, i, n_var, seed=seed)
            for i in range(3)
        ]
        
        self.sigmas = torch.tensor([10.0, 30.0, 50.0])
        self.biases_comp = torch.tensor([0.0, 100.0, 200.0])
        self.offsets = torch.tensor([1500.0, 1800.0, 1900.0])
    
    def __call__(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        nx = x.shape[-1]
        n = 3
        
        sigmas = self.sigmas.to(x.device, x.dtype)
        biases = self.biases_comp.to(x.device, x.dtype)
        offsets = self.offsets.to(x.device, x.dtype)
        
        # Calculate weights
        weights = self._calc_weights(x, sigmas)
        
        # Evaluate each hybrid function
        vals = torch.zeros(*batch_shape, n, device=x.device, dtype=x.dtype)
        
        hybrid_classes = [
            hybrid_module.CEC2017_F15,
            hybrid_module.CEC2017_F18,
            hybrid_module.CEC2017_F19,
        ]
        
        for i in range(n):
            hybrid_func = hybrid_classes[i](
                n_var=nx,
                rotation=self.rotations[i],
                shift=self.shifts[i],
                shuffle=self.shuffles[i],
            )
            vals[..., i] = hybrid_func(x) - offsets[i]
        
        # Weighted sum
        result = (weights * (vals + biases)).sum(dim=-1)
        
        return result + self.bias


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

COMPOSITION_FUNCTIONS = {
    "cec2017_f21": CEC2017_F21,
    "cec2017_f22": CEC2017_F22,
    "cec2017_f23": CEC2017_F23,
    "cec2017_f24": CEC2017_F24,
    "cec2017_f25": CEC2017_F25,
    "cec2017_f26": CEC2017_F26,
    "cec2017_f27": CEC2017_F27,
    "cec2017_f28": CEC2017_F28,
    "cec2017_f29": CEC2017_F29,
    "cec2017_f30": CEC2017_F30,
}

# List for iteration
all_functions = [
    CEC2017_F21,
    CEC2017_F22,
    CEC2017_F23,
    CEC2017_F24,
    CEC2017_F25,
    CEC2017_F26,
    CEC2017_F27,
    CEC2017_F28,
    CEC2017_F29,
    CEC2017_F30,
]
