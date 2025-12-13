"""
Repair operators for bounds handling.

This module provides repair operators that handle constraint
violations, primarily keeping solutions within variable bounds.
All methods are differentiable-friendly.

Available repair methods:
    - ClipRepair: Clamp values to bounds (simplest)
    - ReflectRepair: Bounce off boundaries (preserves momentum)
    - WrapRepair: Periodic/toroidal wrapping
    - RandomRepair: Reset violating genes randomly
    - BoundsRepair: Configurable repair with method selection

Differentiable Considerations:
    - ClipRepair: Gradient is zero at boundaries (can cause issues)
    - ReflectRepair: Gradient flows through reflection
    - WrapRepair: Gradient flows through modulo (discontinuous)
    - For differentiable mode, ReflectRepair is generally recommended

Example:
    >>> from evograd.operators import BoundsRepair
    >>> 
    >>> # Using method string
    >>> repair = BoundsRepair(method='reflect')
    >>> repaired = repair(population, xl, xu)
    >>> 
    >>> # Or use specific class
    >>> from evograd.operators import ReflectRepair
    >>> repair = ReflectRepair()
    >>> repaired = repair(population, xl, xu)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Repair",
    "ClipRepair",
    "ReflectRepair",
    "WrapRepair",
    "RandomRepair",
    "BoundsRepair",
    "RepairMethod",
    "NoRepair",
]


class RepairMethod(Enum):
    """Available repair methods."""
    CLIP = "clip"
    REFLECT = "reflect"
    WRAP = "wrap"
    RANDOM = "random"
    NONE = "none"


# =============================================================================
# Base Repair Class
# =============================================================================

class Repair(nn.Module, ABC):
    """
    Abstract base class for repair operators.
    
    Subclasses must implement:
        - _repair(): Apply repair to bring solutions within bounds
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        """
        Repair solutions to satisfy bounds.
        
        Args:
            x: Solutions to repair [n_pop, n_var].
            xl: Lower bounds [n_var] or scalar.
            xu: Upper bounds [n_var] or scalar.
        
        Returns:
            Repaired solutions [n_pop, n_var].
        """
        pass
    
    def forward(
        self,
        x: Tensor,
        xl: Optional[Tensor] = None,
        xu: Optional[Tensor] = None,
        problem: Optional["Problem"] = None,
    ) -> Tensor:
        """
        Apply repair.
        
        Args:
            x: Solutions to repair [n_pop, n_var].
            xl: Lower bounds (or provide problem).
            xu: Upper bounds (or provide problem).
            problem: Problem instance with bounds.
        
        Returns:
            Repaired solutions [n_pop, n_var].
        """
        # Get bounds from problem if provided
        if problem is not None:
            xl = problem.xl
            xu = problem.xu
        
        # Ensure bounds are provided
        if xl is None or xu is None:
            raise ValueError("Bounds must be provided via xl/xu or problem")
        
        # Ensure bounds have correct shape
        n_var = x.shape[-1]
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        return self._repair(x, xl, xu)
    
    def __call__(
        self,
        x: Tensor,
        xl: Optional[Tensor] = None,
        xu: Optional[Tensor] = None,
        problem: Optional["Problem"] = None,
    ) -> Tensor:
        """Apply repair (alias for forward)."""
        return self.forward(x, xl, xu, problem)
    
    def is_within_bounds(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
        tol: float = 1e-8,
    ) -> Tensor:
        """
        Check if solutions are within bounds.
        
        Args:
            x: Solutions to check [n_pop, n_var].
            xl: Lower bounds.
            xu: Upper bounds.
            tol: Tolerance for boundary check.
        
        Returns:
            Boolean tensor [n_pop] indicating feasibility.
        """
        within_lower = (x >= xl - tol).all(dim=-1)
        within_upper = (x <= xu + tol).all(dim=-1)
        return within_lower & within_upper


# =============================================================================
# Clip (Clamp) Repair
# =============================================================================

class ClipRepair(Repair):
    """
    Clip repair (clamping to bounds).
    
    Simply clamps values to [xl, xu]. This is the simplest and
    most common repair method.
    
    Note:
        Gradient is zero when values are clipped, which can cause
        issues in differentiable mode. Consider ReflectRepair for
        better gradient flow.
    
    Example:
        >>> repair = ClipRepair()
        >>> repaired = repair(population, xl, xu)
    """
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        return torch.clamp(x, min=xl, max=xu)
    
    def __repr__(self) -> str:
        return "ClipRepair()"


# =============================================================================
# Reflect Repair
# =============================================================================

class ReflectRepair(Repair):
    """
    Reflection repair (bounce off boundaries).
    
    When a value exceeds a bound, it bounces back into the
    feasible region. This preserves the "momentum" of the
    search and provides better gradient flow than clipping.
    
    The reflection is computed as:
        x' = xl + |x - xl| mod (2 * range)
        if x' > xu: x' = 2*xu - x'
    
    Args:
        max_iterations: Maximum reflection iterations (prevents
            infinite loops for extreme violations).
    
    Example:
        >>> repair = ReflectRepair()
        >>> repaired = repair(population, xl, xu)
    
    Note:
        This is the recommended repair method for differentiable
        mode as gradients flow through the reflection operation.
    """
    
    def __init__(self, max_iterations: int = 100) -> None:
        super().__init__()
        self.max_iterations = max_iterations
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        # Compute range
        span = xu - xl
        
        # Handle zero span (fixed variables)
        span = torch.where(span > 0, span, torch.ones_like(span))
        
        # Normalise to [0, 2*span] then fold
        x_shifted = x - xl
        x_mod = torch.remainder(x_shifted, 2 * span)
        
        # Fold back: if > span, reflect from upper bound
        x_folded = torch.where(x_mod > span, 2 * span - x_mod, x_mod)
        
        # Shift back to original space
        x_repaired = xl + x_folded
        
        return x_repaired
    
    def __repr__(self) -> str:
        return f"ReflectRepair(max_iterations={self.max_iterations})"


# =============================================================================
# Wrap (Periodic) Repair
# =============================================================================

class WrapRepair(Repair):
    """
    Wrap repair (periodic/toroidal boundaries).
    
    Values that exceed bounds wrap around to the other side,
    treating the search space as a torus. Useful for periodic
    domains like angles.
    
    The wrapping is computed as:
        x' = xl + (x - xl) mod (xu - xl)
    
    Example:
        >>> # For angular variables [0, 2*pi]
        >>> repair = WrapRepair()
        >>> repaired = repair(angles, 0, 2*np.pi)
    
    Note:
        Gradient is discontinuous at boundaries but flows
        through the modulo operation.
    """
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        span = xu - xl
        
        # Handle zero span
        span = torch.where(span > 0, span, torch.ones_like(span))
        
        # Periodic wrapping
        x_wrapped = xl + torch.remainder(x - xl, span)
        
        return x_wrapped
    
    def __repr__(self) -> str:
        return "WrapRepair()"


# =============================================================================
# Random Repair
# =============================================================================

class RandomRepair(Repair):
    """
    Random repair (reset violating genes).
    
    Genes that violate bounds are reset to random values within
    the feasible region. This is more disruptive than other
    methods but can help escape from boundary regions.
    
    Example:
        >>> repair = RandomRepair()
        >>> repaired = repair(population, xl, xu)
    
    Note:
        Not differentiable through the random reset operation.
        Gradient is zero for repaired genes.
    """
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        # Find violations
        below = x < xl
        above = x > xu
        violates = below | above
        
        # Generate random replacements
        random_vals = xl + (xu - xl) * torch.rand_like(x)
        
        # Replace only violating genes
        x_repaired = torch.where(violates, random_vals, x)
        
        return x_repaired
    
    def __repr__(self) -> str:
        return "RandomRepair()"


# =============================================================================
# No Repair (Identity)
# =============================================================================

class NoRepair(Repair):
    """
    No repair (identity operator).
    
    Returns input unchanged. Useful as a placeholder when
    repair should be disabled or handled elsewhere.
    
    Example:
        >>> repair = NoRepair()
        >>> repaired = repair(population, xl, xu)  # Returns unchanged
    """
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        return x
    
    def __repr__(self) -> str:
        return "NoRepair()"


# =============================================================================
# Configurable Bounds Repair
# =============================================================================

class BoundsRepair(Repair):
    """
    Configurable bounds repair with method selection.
    
    Convenience class that allows selecting the repair method
    via a string or enum parameter.
    
    Args:
        method: Repair method to use. Options:
            - 'clip': Clamp to bounds (default)
            - 'reflect': Bounce off boundaries
            - 'wrap': Periodic wrapping
            - 'random': Reset violating genes
            - 'none': No repair
    
    Example:
        >>> repair = BoundsRepair(method='reflect')
        >>> repaired = repair(population, xl, xu)
        >>> 
        >>> # Or use enum
        >>> from evograd.operators import RepairMethod
        >>> repair = BoundsRepair(method=RepairMethod.WRAP)
    """
    
    _METHOD_MAP = {
        'clip': ClipRepair,
        'clamp': ClipRepair,
        'reflect': ReflectRepair,
        'bounce': ReflectRepair,
        'wrap': WrapRepair,
        'periodic': WrapRepair,
        'toroidal': WrapRepair,
        'random': RandomRepair,
        'none': NoRepair,
    }
    
    def __init__(
        self,
        method: Union[str, RepairMethod] = "clip",
    ) -> None:
        super().__init__()
        
        # Convert enum to string
        if isinstance(method, RepairMethod):
            method = method.value
        
        method = method.lower()
        
        if method not in self._METHOD_MAP:
            valid = list(self._METHOD_MAP.keys())
            raise ValueError(
                f"Unknown repair method '{method}'. Valid options: {valid}"
            )
        
        self.method = method
        self._repair_impl = self._METHOD_MAP[method]()
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        return self._repair_impl._repair(x, xl, xu)
    
    def __repr__(self) -> str:
        return f"BoundsRepair(method='{self.method}')"


# =============================================================================
# Soft Clip Repair (Differentiable-friendly)
# =============================================================================

class SoftClipRepair(Repair):
    """
    Soft clip repair using smooth approximation.
    
    Uses a smooth approximation of the clip function that
    provides non-zero gradients near boundaries. This is
    useful when gradients are important but values should
    still be approximately within bounds.
    
    The soft clip is computed using:
        softplus(x - xl) - softplus(x - xu) + xl
    
    Args:
        beta: Smoothness parameter (higher = sharper, closer to hard clip).
        margin: How far outside bounds the soft clip extends.
    
    Example:
        >>> repair = SoftClipRepair(beta=10.0)
        >>> repaired = repair(population, xl, xu)
    
    Note:
        Values may slightly exceed bounds. Use hard clip after
        if strict feasibility is required.
    """
    
    def __init__(
        self,
        beta: float = 10.0,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.margin = margin
    
    def _soft_clip(
        self,
        x: Tensor,
        lower: Tensor,
        upper: Tensor,
    ) -> Tensor:
        """Smooth clip using softplus."""
        # Soft lower bound: max(x, lower) ≈ lower + softplus(x - lower)
        x_lower = lower + torch.nn.functional.softplus(
            (x - lower) * self.beta
        ) / self.beta
        
        # Soft upper bound: min(x, upper) ≈ upper - softplus(upper - x)
        x_clipped = upper - torch.nn.functional.softplus(
            (upper - x_lower) * self.beta
        ) / self.beta
        
        return x_clipped
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        return self._soft_clip(x, xl, xu)
    
    def __repr__(self) -> str:
        return f"SoftClipRepair(beta={self.beta}, margin={self.margin})"


# =============================================================================
# Penalty-based Repair (returns penalty instead of repairing)
# =============================================================================

class PenaltyRepair(Repair):
    """
    Penalty-based "repair" that computes constraint violation.
    
    Instead of modifying solutions, this computes a penalty
    term that can be added to the fitness. Useful for
    constrained optimisation with penalty methods.
    
    The penalty is computed as:
        penalty = sum(max(0, xl - x)^2 + max(0, x - xu)^2)
    
    Args:
        penalty_weight: Multiplier for the penalty term.
        power: Exponent for violation (1=linear, 2=quadratic).
    
    Example:
        >>> repair = PenaltyRepair(penalty_weight=1000)
        >>> penalty = repair.compute_penalty(population, xl, xu)
        >>> fitness_penalised = fitness + penalty
    
    Note:
        forward() still returns the input unchanged. Use
        compute_penalty() to get the penalty values.
    """
    
    def __init__(
        self,
        penalty_weight: float = 1.0,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight
        self.power = power
    
    def compute_penalty(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        """
        Compute penalty for constraint violations.
        
        Args:
            x: Solutions [n_pop, n_var].
            xl: Lower bounds.
            xu: Upper bounds.
        
        Returns:
            Penalty values [n_pop].
        """
        # Lower bound violations
        lower_violation = torch.clamp(xl - x, min=0.0)
        
        # Upper bound violations
        upper_violation = torch.clamp(x - xu, min=0.0)
        
        # Total penalty per individual
        penalty = (
            lower_violation.pow(self.power).sum(dim=-1) +
            upper_violation.pow(self.power).sum(dim=-1)
        )
        
        return self.penalty_weight * penalty
    
    def _repair(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        # No modification - penalty is computed separately
        return x
    
    def __repr__(self) -> str:
        return (
            f"PenaltyRepair("
            f"penalty_weight={self.penalty_weight}, "
            f"power={self.power})"
        )
