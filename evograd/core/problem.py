"""
Problem definition for EvoGrad optimisation.

This module provides the Problem class that encapsulates:
    - Objective function(s)
    - Variable bounds (xl, xu)
    - Constraints (equality and inequality)
    - Problem metadata

The Problem class is intentionally minimal. Bounds handling (repair),
population initialisation (sampling), and other operations belong
to their respective operator classes in the operators subpackage.

Problems can be defined in two ways:
    1. Functional: Pass objective as a callable
    2. Subclassing: Override _evaluate() method

Example (functional):
    >>> from evograd.core import Problem
    >>> 
    >>> problem = Problem(
    ...     objective=ackley,
    ...     n_var=30,
    ...     xl=-32.768,
    ...     xu=32.768,
    ... )

Example (subclassing):
    >>> class Rastrigin(Problem):
    ...     def __init__(self, n_var=30):
    ...         super().__init__(n_var=n_var, xl=-5.12, xu=5.12)
    ...     
    ...     def _evaluate(self, x):
    ...         A = 10.0
    ...         return A * x.shape[-1] + (x**2 - A * torch.cos(2 * torch.pi * x)).sum(dim=-1)
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

from evograd.utils.device import get_device, ensure_tensor

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "Problem",
]


# =============================================================================
# Problem Class
# =============================================================================

class Problem(nn.Module):
    """
    Optimisation problem definition.
    
    Encapsulates the objective function, variable bounds, and optional
    constraints. The Problem class is intentionally minimal - it only
    stores problem definition, not operations on solutions.
    
    For bounds handling, use operators/repair.py.
    For population initialisation, use operators/sampling.py.
    
    The problem can be defined either by passing an objective callable
    to the constructor, or by subclassing and overriding _evaluate().
    
    Args:
        objective: Callable that takes (N, n_var) tensor and returns (N,) fitness.
            If None, subclass must override _evaluate().
        n_var: Number of decision variables.
        xl: Lower bounds. Can be:
            - Scalar (applied to all variables)
            - List of length n_var
            - Tensor of shape (n_var,)
        xu: Upper bounds (same format as xl).
        constraints: List of constraint tuples: (func, type).
            - func: Callable (N, n_var) -> (N,) or (N, n_constraints)
            - type: 'ineq' for g(x) <= 0, 'eq' for h(x) = 0
        n_obj: Number of objectives (default: 1, multi-objective planned).
        name: Optional problem name for identification.
        device: Computation device (default: auto-detect).
        dtype: Tensor dtype (default: float32). Use ``torch.float64`` for
            problems that require higher numerical precision, such as parameter
            estimation with stiff ODE solvers. All operators respect the dtype
            propagated from the Problem; ensure the Algorithm is created with a
            matching dtype to avoid silent precision loss.

    Attributes:
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        n_ieq_constr: Number of inequality constraints.
        n_eq_constr: Number of equality constraints.
        n_constr: Total number of constraints.
        xl: Lower bounds tensor of shape (n_var,).
        xu: Upper bounds tensor of shape (n_var,).
        name: Problem name.
    
    Example:
        >>> # Simple unconstrained problem
        >>> problem = Problem(
        ...     objective=lambda x: (x ** 2).sum(dim=-1),
        ...     n_var=10,
        ...     xl=-5.0,
        ...     xu=5.0,
        ... )
        >>> 
        >>> # Evaluate a batch of solutions
        >>> x = torch.rand(100, 10) * 10 - 5
        >>> fitness = problem.evaluate(x)
        >>> print(fitness.shape)  # torch.Size([100])
        
        >>> # Problem with constraints
        >>> problem = Problem(
        ...     objective=lambda x: x[:, 0] + x[:, 1],
        ...     n_var=2,
        ...     xl=0.0,
        ...     xu=10.0,
        ...     constraints=[
        ...         (lambda x: x[:, 0] + x[:, 1] - 5, 'ineq'),  # x0 + x1 <= 5
        ...         (lambda x: x[:, 0] - 2 * x[:, 1], 'eq'),    # x0 = 2 * x1
        ...     ],
        ... )
    """
    
    # Constraint type constants
    INEQ = "ineq"  # Inequality: g(x) <= 0
    EQ = "eq"      # Equality: h(x) = 0
    
    def __init__(
        self,
        objective: Optional[Callable[[Tensor], Tensor]] = None,
        n_var: Optional[int] = None,
        xl: Union[float, List[float], Tensor] = -100.0,
        xu: Union[float, List[float], Tensor] = 100.0,
        constraints: Optional[List[Tuple[Callable[[Tensor], Tensor], str]]] = None,
        n_obj: int = 1,
        name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        
        # Validate inputs
        if objective is None and type(self)._evaluate is Problem._evaluate:
            raise ValueError(
                "Either provide 'objective' callable or subclass Problem "
                "and override _evaluate()"
            )
        
        if n_var is None:
            raise ValueError("n_var (number of variables) must be specified")
        
        if n_var < 1:
            raise ValueError(f"n_var must be >= 1, got {n_var}")
        
        if n_obj < 1:
            raise ValueError(f"n_obj must be >= 1, got {n_obj}")
        
        # Store configuration
        self.n_var = n_var
        self.n_obj = n_obj
        self._objective = objective
        self.name = name or self.__class__.__name__
        self.device = get_device(device)
        self.dtype = dtype
        
        # Process and register bounds
        xl_tensor, xu_tensor = self._process_bounds(xl, xu, n_var)
        self.register_buffer("xl", xl_tensor)
        self.register_buffer("xu", xu_tensor)
        
        # Process constraints
        self._constraints: List[Tuple[Callable, str]] = []
        self.n_ieq_constr = 0
        self.n_eq_constr = 0
        
        if constraints is not None:
            for func, ctype in constraints:
                ctype = ctype.lower()
                if ctype not in (self.INEQ, self.EQ):
                    raise ValueError(
                        f"Constraint type must be 'ineq' or 'eq', got '{ctype}'"
                    )
                self._constraints.append((func, ctype))
                if ctype == self.INEQ:
                    self.n_ieq_constr += 1
                else:
                    self.n_eq_constr += 1
        
        self.n_constr = self.n_ieq_constr + self.n_eq_constr
    
    def _process_bounds(
        self,
        xl: Union[float, List[float], Tensor],
        xu: Union[float, List[float], Tensor],
        n_var: int,
    ) -> Tuple[Tensor, Tensor]:
        """Process and validate bounds."""
        # Convert to tensors
        xl_tensor = ensure_tensor(xl, device=self.device, dtype=self.dtype)
        xu_tensor = ensure_tensor(xu, device=self.device, dtype=self.dtype)
        
        # Expand scalars to full dimension
        if xl_tensor.dim() == 0 or xl_tensor.numel() == 1:
            xl_tensor = xl_tensor.expand(n_var).clone()
        if xu_tensor.dim() == 0 or xu_tensor.numel() == 1:
            xu_tensor = xu_tensor.expand(n_var).clone()
        
        # Validate shapes
        if xl_tensor.shape[0] != n_var:
            raise ValueError(
                f"xl has {xl_tensor.shape[0]} elements but n_var={n_var}"
            )
        if xu_tensor.shape[0] != n_var:
            raise ValueError(
                f"xu has {xu_tensor.shape[0]} elements but n_var={n_var}"
            )
        
        # Validate bounds ordering
        if (xl_tensor > xu_tensor).any():
            raise ValueError("Lower bounds must be <= upper bounds")
        
        return xl_tensor, xu_tensor
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def _evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate objective function.
        
        Override this method in subclasses to define custom objectives.
        
        Args:
            x: Decision variables of shape (N, n_var).
        
        Returns:
            Fitness values of shape (N,) for single-objective,
            or (N, n_obj) for multi-objective.
        """
        if self._objective is not None:
            return self._objective(x)
        raise NotImplementedError(
            "Subclass must implement _evaluate() if objective not provided"
        )
    
    def evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate fitness of solutions.
        
        Args:
            x: Decision variables of shape (N, n_var) or (n_var,).
        
        Returns:
            Fitness values of shape (N,) or scalar.
        """
        # Handle single solution
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        # Ensure correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Validate shape
        if x.shape[-1] != self.n_var:
            raise ValueError(
                f"Expected {self.n_var} variables, got {x.shape[-1]}"
            )
        
        # Evaluate objective
        fitness = self._evaluate(x)
        
        # Ensure correct output shape
        if fitness.dim() == 0:
            fitness = fitness.unsqueeze(0)
        
        if squeeze_output and fitness.shape[0] == 1:
            fitness = fitness.squeeze(0)
        
        return fitness
    
    def evaluate_constraints(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Evaluate all constraints.
        
        Args:
            x: Decision variables of shape (N, n_var).
        
        Returns:
            Dictionary with keys:
                - 'ineq': Inequality constraint values g(x), shape (N, n_ieq_constr)
                - 'eq': Equality constraint values h(x), shape (N, n_eq_constr)
                - 'cv': Total constraint violation per solution, shape (N,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(device=self.device, dtype=self.dtype)
        n_solutions = x.shape[0]
        
        ineq_values = []
        eq_values = []
        
        for func, ctype in self._constraints:
            val = func(x)
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            
            if ctype == self.INEQ:
                ineq_values.append(val)
            else:
                eq_values.append(val)
        
        # Stack constraint values
        if ineq_values:
            ineq = torch.cat(ineq_values, dim=-1)
        else:
            ineq = torch.zeros(n_solutions, 0, device=self.device, dtype=self.dtype)
        
        if eq_values:
            eq = torch.cat(eq_values, dim=-1)
        else:
            eq = torch.zeros(n_solutions, 0, device=self.device, dtype=self.dtype)
        
        # Compute constraint violation
        # For ineq: max(0, g(x))  (violation if positive)
        # For eq: |h(x)|          (violation if non-zero)
        cv = torch.zeros(n_solutions, device=self.device, dtype=self.dtype)
        if ineq.shape[-1] > 0:
            cv = cv + torch.clamp(ineq, min=0).sum(dim=-1)
        if eq.shape[-1] > 0:
            cv = cv + torch.abs(eq).sum(dim=-1)
        
        return {
            "ineq": ineq,
            "eq": eq,
            "cv": cv,
        }
    
    def is_feasible(self, x: Tensor, tol: float = 1e-6) -> Tensor:
        """
        Check if solutions satisfy all constraints.
        
        Args:
            x: Decision variables of shape (N, n_var) or (n_var,).
            tol: Tolerance for constraint satisfaction.
        
        Returns:
            Boolean tensor of shape (N,) or scalar.
        """
        squeeze = x.dim() == 1
        
        if self.n_constr == 0:
            if squeeze:
                return torch.tensor(True, device=self.device)
            return torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
        
        cv = self.evaluate_constraints(x)["cv"]
        result = cv <= tol
        
        if squeeze:
            result = result.squeeze(0)
        
        return result
    
    def has_constraints(self) -> bool:
        """Check if problem has any constraints."""
        return self.n_constr > 0
    
    # =========================================================================
    # PyTorch Forward
    # =========================================================================
    
    def forward(self, x: Tensor) -> Tensor:
        """
        PyTorch forward pass (alias for evaluate).
        
        Enables using Problem as an nn.Module in computation graphs.
        """
        return self.evaluate(x)
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        parts = [
            f"name='{self.name}'",
            f"n_var={self.n_var}",
            f"n_obj={self.n_obj}",
        ]
        
        if self.n_constr > 0:
            parts.append(f"n_constr={self.n_ieq_constr}ineq+{self.n_eq_constr}eq")
        
        return f"{self.__class__.__name__}({', '.join(parts)})"
    
    def summary(self) -> str:
        """Return detailed problem summary."""
        lines = [
            f"{'=' * 50}",
            f"Problem: {self.name}",
            f"{'=' * 50}",
            f"  Variables: {self.n_var}",
            f"  Objectives: {self.n_obj}",
        ]
        
        if self.n_constr > 0:
            lines.append(
                f"  Constraints: {self.n_constr} "
                f"({self.n_ieq_constr} inequality, {self.n_eq_constr} equality)"
            )
        else:
            lines.append("  Constraints: None")
        
        lines.extend([
            f"",
            f"Bounds:",
            f"  xl: [{float(self.xl.min()):.4g}, {float(self.xl.max()):.4g}]",
            f"  xu: [{float(self.xu.min()):.4g}, {float(self.xu.max()):.4g}]",
            f"",
            f"Device: {self.device}",
            f"{'=' * 50}",
        ])
        
        return "\n".join(lines)
