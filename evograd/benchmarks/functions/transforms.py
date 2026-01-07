"""
Transformations for benchmark functions.

This module provides transformations that wrap base functions to create
more challenging optimization problems, following CEC competition conventions.

Transformations:
- ShiftedFunction: f(x - o) where o is the shift vector
- RotatedFunction: f(R @ x) where R is a rotation matrix
- ShiftedRotatedFunction: f(R @ (x - o))
- ScaledFunction: f(lambda * x)
- AsymmetricFunction: Applies asymmetric transformation
- OscillatedFunction: Applies oscillation transformation
"""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from .base import BenchmarkFunction


class ShiftedFunction(BenchmarkFunction):
    """
    Shifted benchmark function.
    
    f_shift(x) = f(x - shift)
    
    Moves the optimal solution away from the origin, preventing
    algorithms from exploiting symmetry.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        shift: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize shifted function.
        
        Args:
            base_function: Base function to shift.
            shift: Shift vector of shape [n_var]. If None, generated randomly.
            seed: Random seed for generating shift vector.
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"shifted_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        
        # Generate or use provided shift
        if shift is not None:
            self.shift = shift
        else:
            if seed is not None:
                torch.manual_seed(seed)
            # Generate shift within 80% of the search space
            xl, xu = base_function.xl, base_function.xu
            range_val = xu - xl
            self.shift = xl + 0.1 * range_val + 0.8 * range_val * torch.rand(self.n_var)
        
        # Update optimal location
        self._optimal_x = self.shift + base_function.optimal_x
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def __call__(self, x: Tensor) -> Tensor:
        shift = self.shift.to(x.device, x.dtype)
        return self.base_function(x - shift)


class RotatedFunction(BenchmarkFunction):
    """
    Rotated benchmark function.
    
    f_rot(x) = f(R @ x)
    
    Applies rotation to make separable functions non-separable,
    testing an algorithm's ability to handle variable interactions.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        rotation_matrix: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize rotated function.
        
        Args:
            base_function: Base function to rotate.
            rotation_matrix: Orthogonal rotation matrix of shape [n_var, n_var].
                            If None, generated randomly.
            seed: Random seed for generating rotation matrix.
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"rotated_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        
        # Generate or use provided rotation matrix
        if rotation_matrix is not None:
            self.R = rotation_matrix
        else:
            if seed is not None:
                torch.manual_seed(seed)
            # Generate orthogonal matrix using QR decomposition
            A = torch.randn(self.n_var, self.n_var)
            Q, _ = torch.linalg.qr(A)
            self.R = Q
        
        # For rotated functions, optimal is still at the same point
        self._optimal_x = base_function.optimal_x.clone()
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def __call__(self, x: Tensor) -> Tensor:
        R = self.R.to(x.device, x.dtype)
        # x @ R.T is equivalent to R @ x for each row
        x_rotated = x @ R.T
        return self.base_function(x_rotated)


class ShiftedRotatedFunction(BenchmarkFunction):
    """
    Shifted and rotated benchmark function.
    
    f_sr(x) = f(R @ (x - shift))
    
    Combines both transformations for maximum difficulty.
    This is the standard transformation used in CEC benchmarks.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        shift: Optional[Tensor] = None,
        rotation_matrix: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize shifted and rotated function.
        
        Args:
            base_function: Base function to transform.
            shift: Shift vector. If None, generated randomly.
            rotation_matrix: Rotation matrix. If None, generated randomly.
            seed: Random seed for random generation.
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"shifted_rotated_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate or use provided shift
        if shift is not None:
            self.shift = shift
        else:
            xl, xu = base_function.xl, base_function.xu
            range_val = xu - xl
            self.shift = xl + 0.1 * range_val + 0.8 * range_val * torch.rand(self.n_var)
        
        # Generate or use provided rotation matrix
        if rotation_matrix is not None:
            self.R = rotation_matrix
        else:
            A = torch.randn(self.n_var, self.n_var)
            Q, _ = torch.linalg.qr(A)
            self.R = Q
        
        # Optimal is at the shifted location
        self._optimal_x = self.shift.clone()
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def __call__(self, x: Tensor) -> Tensor:
        shift = self.shift.to(x.device, x.dtype)
        R = self.R.to(x.device, x.dtype)
        
        x_shifted = x - shift
        x_rotated = x_shifted @ R.T
        
        return self.base_function(x_rotated)


class ScaledFunction(BenchmarkFunction):
    """
    Scaled benchmark function.
    
    f_scaled(x) = f(scale * x)
    
    Changes the scale of the search space.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        scale: Union[float, Tensor] = 1.0,
    ):
        """
        Initialize scaled function.
        
        Args:
            base_function: Base function to scale.
            scale: Scale factor (scalar or per-dimension tensor).
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"scaled_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        
        if isinstance(scale, (int, float)):
            self.scale = torch.full((self.n_var,), float(scale))
        else:
            self.scale = scale
        
        self._optimal_x = base_function.optimal_x / self.scale
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def __call__(self, x: Tensor) -> Tensor:
        scale = self.scale.to(x.device, x.dtype)
        return self.base_function(x * scale)


class AsymmetricFunction(BenchmarkFunction):
    """
    Asymmetric transformation wrapper.
    
    Applies T_asy transformation from CEC benchmarks:
    x_i = x_i^(1 + beta * i/(n-1) * sqrt(x_i)) for x_i > 0
    
    Breaks symmetry around the origin.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        beta: float = 0.5,
    ):
        """
        Initialize asymmetric function.
        
        Args:
            base_function: Base function to transform.
            beta: Asymmetry parameter (typically 0.5).
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"asymmetric_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        self.beta = beta
        
        self._optimal_x = base_function.optimal_x.clone()
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def _transform(self, x: Tensor) -> Tensor:
        """Apply asymmetric transformation."""
        n = x.shape[-1]
        i = torch.arange(n, device=x.device, dtype=x.dtype)
        exponent = 1 + self.beta * i / max(n - 1, 1) * torch.sqrt(torch.abs(x) + 1e-10)
        
        x_transformed = x.clone()
        positive_mask = x > 0
        x_transformed = torch.where(
            positive_mask,
            torch.pow(x + 1e-10, exponent),
            x
        )
        return x_transformed
    
    def __call__(self, x: Tensor) -> Tensor:
        x_transformed = self._transform(x)
        return self.base_function(x_transformed)


class OscillatedFunction(BenchmarkFunction):
    """
    Oscillation transformation wrapper.
    
    Applies T_osz transformation from CEC benchmarks:
    Creates local irregularities while preserving global structure.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
    ):
        """
        Initialize oscillated function.
        
        Args:
            base_function: Base function to transform.
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"oscillated_{base_function.name}"
        self.optimal_value = base_function.optimal_value
        
        self._optimal_x = base_function.optimal_x.clone()
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def _transform(self, x: Tensor) -> Tensor:
        """Apply oscillation transformation T_osz."""
        x_hat = torch.where(
            x != 0,
            torch.log(torch.abs(x) + 1e-10),
            torch.zeros_like(x)
        )
        
        c1 = torch.where(x > 0, torch.tensor(10.0), torch.tensor(5.5))
        c2 = torch.where(x > 0, torch.tensor(7.9), torch.tensor(3.1))
        
        sign_x = torch.sign(x)
        
        return sign_x * torch.exp(
            x_hat + 0.049 * (
                torch.sin(c1 * x_hat) + torch.sin(c2 * x_hat)
            )
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        x_transformed = self._transform(x)
        return self.base_function(x_transformed)


class BiasedFunction(BenchmarkFunction):
    """
    Biased benchmark function.
    
    f_biased(x) = f(x) + bias
    
    Shifts the optimal function value.
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        bias: float = 0.0,
    ):
        """
        Initialize biased function.
        
        Args:
            base_function: Base function.
            bias: Value added to function output.
        """
        super().__init__(
            n_var=base_function.n_var,
            xl=base_function.xl,
            xu=base_function.xu,
        )
        
        self.base_function = base_function
        self.name = f"biased_{base_function.name}"
        self.optimal_value = base_function.optimal_value + bias
        self.bias = bias
        
        self._optimal_x = base_function.optimal_x.clone()
    
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.base_function(x) + self.bias


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_rotation_matrix(n: int, seed: Optional[int] = None) -> Tensor:
    """
    Generate a random orthogonal rotation matrix.
    
    Args:
        n: Dimension of the matrix.
        seed: Random seed.
    
    Returns:
        Orthogonal matrix of shape [n, n].
    """
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(n, n)
    Q, _ = torch.linalg.qr(A)
    return Q


def generate_shift_vector(
    n: int,
    xl: Union[float, Tensor],
    xu: Union[float, Tensor],
    margin: float = 0.1,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate a random shift vector within bounds.
    
    Args:
        n: Dimension of the vector.
        xl: Lower bounds.
        xu: Upper bounds.
        margin: Margin from bounds (fraction of range).
        seed: Random seed.
    
    Returns:
        Shift vector of shape [n].
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if isinstance(xl, (int, float)):
        xl = torch.full((n,), float(xl))
    if isinstance(xu, (int, float)):
        xu = torch.full((n,), float(xu))
    
    range_val = xu - xl
    return xl + margin * range_val + (1 - 2 * margin) * range_val * torch.rand(n)
