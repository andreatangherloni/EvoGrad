"""
Base class for benchmark functions.

All benchmark functions inherit from BenchmarkFunction and implement
the __call__ method for evaluation. Bounds can be specified per-function
or overridden at instantiation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class BenchmarkFunction(ABC):
    """
    Abstract base class for benchmark optimization functions.
    
    All benchmark functions support:
    - Configurable dimensionality (n_var)
    - Configurable bounds (xl, xu) - can be scalar or per-dimension
    - Known optimal value and location
    - Batch evaluation for populations
    
    Args:
        n_var: Number of decision variables (dimensions).
        xl: Lower bound(s). Scalar or tensor of shape [n_var].
        xu: Upper bound(s). Scalar or tensor of shape [n_var].
    
    Example:
        >>> sphere = Sphere(n_var=10)
        >>> x = torch.randn(100, 10)  # Population of 100 individuals
        >>> f = sphere(x)  # Shape: [100]
        
        >>> # Custom bounds
        >>> sphere = Sphere(n_var=10, xl=-5.0, xu=5.0)
    """
    
    # Class attributes (can be overridden in subclasses)
    name: str = "base_function"
    optimal_value: float = 0.0
    
    def __init__(
        self,
        n_var: int = 30,
        xl: Optional[Union[float, Tensor]] = None,
        xu: Optional[Union[float, Tensor]] = None,
    ):
        """
        Initialize benchmark function.
        
        Args:
            n_var: Number of decision variables.
            xl: Lower bound(s). If None, uses default_bounds().
            xu: Upper bound(s). If None, uses default_bounds().
        """
        self.n_var = n_var
        
        # Get default bounds
        default_lb, default_ub = self.default_bounds()
        
        # Set lower bounds
        if xl is None:
            xl = default_lb
        if isinstance(xl, (int, float)):
            self._xl = torch.full((n_var,), float(xl))
        else:
            self._xl = xl.clone() if isinstance(xl, Tensor) else torch.tensor(xl)
            if self._xl.numel() == 1:
                self._xl = self._xl.expand(n_var).clone()
        
        # Set upper bounds
        if xu is None:
            xu = default_ub
        if isinstance(xu, (int, float)):
            self._xu = torch.full((n_var,), float(xu))
        else:
            self._xu = xu.clone() if isinstance(xu, Tensor) else torch.tensor(xu)
            if self._xu.numel() == 1:
                self._xu = self._xu.expand(n_var).clone()
        
        # Compute optimal location (default: origin)
        self._optimal_x = self._compute_optimal_x()
    
    @property
    def xl(self) -> Tensor:
        """Lower bounds as tensor of shape [n_var]."""
        return self._xl
    
    @property
    def xu(self) -> Tensor:
        """Upper bounds as tensor of shape [n_var]."""
        return self._xu
    
    @property
    def bounds(self) -> Tuple[Tensor, Tensor]:
        """Return (lower_bounds, upper_bounds) tuple."""
        return (self._xl, self._xu)
    
    @property
    def optimal_x(self) -> Tensor:
        """Optimal solution location."""
        return self._optimal_x
    
    def _compute_optimal_x(self) -> Tensor:
        """
        Compute the optimal solution location.
        
        Override in subclasses if optimal is not at origin.
        """
        return torch.zeros(self.n_var)
    
    @abstractmethod
    def default_bounds(self) -> Tuple[float, float]:
        """
        Return default (lower, upper) bounds for this function.
        
        Returns:
            Tuple of (lower_bound, upper_bound) scalars.
        """
        pass
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the function at point(s) x.
        
        Args:
            x: Decision variables of shape [..., n_var].
               Can be single point [n_var] or batch [N, n_var].
        
        Returns:
            Function values of shape [...] (same batch dimensions as input).
        """
        pass
    
    def to(self, device: torch.device) -> "BenchmarkFunction":
        """Move bounds to specified device."""
        self._xl = self._xl.to(device)
        self._xu = self._xu.to(device)
        self._optimal_x = self._optimal_x.to(device)
        return self
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_var={self.n_var}, "
            f"xl={self._xl[0].item():.2f}, "
            f"xu={self._xu[0].item():.2f})"
        )
    
    def info(self) -> str:
        """Return detailed information about the function."""
        return (
            f"Function: {self.name}\n"
            f"  Dimensions: {self.n_var}\n"
            f"  Bounds: [{self._xl[0].item():.2f}, {self._xu[0].item():.2f}]\n"
            f"  Optimal value: {self.optimal_value}\n"
            f"  Optimal location: {self._optimal_x[:3].tolist()}..."
        )


class CompositeFunction(BenchmarkFunction):
    """
    Composite function combining multiple base functions.
    
    Used for creating CEC-style hybrid and composition functions.
    """
    
    def __init__(
        self,
        functions: list,
        weights: Optional[Tensor] = None,
        n_var: int = 30,
        xl: Optional[Union[float, Tensor]] = None,
        xu: Optional[Union[float, Tensor]] = None,
    ):
        """
        Initialize composite function.
        
        Args:
            functions: List of BenchmarkFunction instances.
            weights: Weights for combining functions. If None, equal weights.
            n_var: Number of decision variables.
            xl: Lower bounds.
            xu: Upper bounds.
        """
        self.functions = functions
        self.n_functions = len(functions)
        
        if weights is None:
            self.weights = torch.ones(self.n_functions) / self.n_functions
        else:
            self.weights = weights / weights.sum()
        
        # Use first function's bounds as default
        if xl is None and functions:
            xl = functions[0]._xl[0].item()
        if xu is None and functions:
            xu = functions[0]._xu[0].item()
        
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        self.name = "composite"
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Weighted sum of component functions."""
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        weights = self.weights.to(x.device, x.dtype)
        
        for i, func in enumerate(self.functions):
            result = result + weights[i] * func(x)
        
        return result
