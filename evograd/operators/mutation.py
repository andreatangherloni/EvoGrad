"""
Mutation operators for introducing variation.

This module provides mutation operators that introduce random
perturbations to individuals, promoting exploration of the search
space. All operators support both classical and differentiable modes.

Available mutations:
    - PolynomialMutation: Bounded polynomial mutation (GA)
    - GaussianMutation: Gaussian/normal perturbation
    - UniformMutation: Uniform random perturbation
    - NonUniformMutation: Decreasing perturbation over time
    - BoundaryMutation: Reset genes to boundary values

Differentiable Mode:
    When `differentiable=True`, mutation masks use Binary-Concrete
    (Gumbel-Sigmoid) relaxation, and perturbations use the
    reparameterisation trick for gradient flow.

Example:
    >>> from evograd.operators import PolynomialMutation
    >>> 
    >>> # Classical mode
    >>> mutation = PolynomialMutation(eta=20, prob=None)  # prob=1/n_var
    >>> offspring = mutation(population, xl, xu)
    >>> 
    >>> # Differentiable mode
    >>> mutation = PolynomialMutation(
    ...     eta=20,
    ...     prob=0.1,
    ...     differentiable=True,
    ...     learn_eta=True,
    ... )
    >>> offspring = mutation(population, xl, xu)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Mutation",
    "PolynomialMutation",
    "GaussianMutation",
    "UniformMutation",
    "NonUniformMutation",
    "BoundaryMutation",
    "NoMutation",
]


# =============================================================================
# Base Mutation Class
# =============================================================================

class Mutation(nn.Module, ABC):
    """
    Abstract base class for mutation operators.
    
    Subclasses must implement:
        - _mutate(): Apply mutation to individuals
    
    Args:
        prob: Mutation probability per gene. If None, defaults to 1/n_var.
        differentiable: If True, use Binary-Concrete for soft masks.
        temperature: Temperature for Binary-Concrete.
        learn_temperature: If True, temperature is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables (for per-gene probability).
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.differentiable = differentiable
        self.n_var = n_var
        self._default_prob = prob is None
        
        # Temperature parameter (log for positivity)
        if learn_temperature and differentiable:
            self._log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "_log_temperature",
                torch.tensor(temperature).log()
            )
        
        # Mutation probability as logits
        # If prob is None, we'll compute 1/n_var at runtime
        if prob is not None:
            prob_logit = self._prob_to_logit(prob)
            if learn_prob and differentiable:
                if n_var is not None:
                    self.prob_logits = nn.Parameter(
                        torch.full((n_var,), prob_logit)
                    )
                else:
                    self.prob_logits = nn.Parameter(torch.tensor(prob_logit))
            else:
                if n_var is not None:
                    self.register_buffer(
                        "prob_logits",
                        torch.full((n_var,), prob_logit)
                    )
                else:
                    self.register_buffer(
                        "prob_logits",
                        torch.tensor(prob_logit)
                    )
        else:
            # Will be set dynamically based on n_var
            self.prob_logits = None
    
    @staticmethod
    def _prob_to_logit(p: float, eps: float = 1e-7) -> float:
        """Convert probability to logit."""
        p = max(min(p, 1 - eps), eps)
        return torch.logit(torch.tensor(p)).item()
    
    @property
    def temperature(self) -> Tensor:
        """Current temperature value."""
        return self._log_temperature.exp()
    
    @property
    def prob(self) -> Optional[Tensor]:
        """Current mutation probability."""
        if self.prob_logits is not None:
            return torch.sigmoid(self.prob_logits)
        return None
    
    def _get_prob(self, n_var: int, device: torch.device) -> Tensor:
        """Get mutation probability, computing default if needed."""
        if self.prob_logits is not None:
            logits = self.prob_logits.to(device)
            if logits.dim() == 0:
                return torch.sigmoid(logits).expand(n_var)
            return torch.sigmoid(logits)
        else:
            # Default: 1/n_var
            return torch.full((n_var,), 1.0 / n_var, device=device)
    
    def _get_prob_logits(self, n_var: int, device: torch.device) -> Tensor:
        """Get probability logits, computing default if needed."""
        if self.prob_logits is not None:
            logits = self.prob_logits.to(device)
            if logits.dim() == 0:
                return logits.expand(n_var)
            return logits
        else:
            # Default: 1/n_var
            default_prob = 1.0 / n_var
            return torch.full(
                (n_var,),
                self._prob_to_logit(default_prob),
                device=device
            )
    
    def _binary_concrete(
        self,
        logits: Tensor,
        hard: bool = True,
        eps: float = 1e-10,
    ) -> Tensor:
        """
        Binary-Concrete (Gumbel-Sigmoid) with straight-through.
        
        Args:
            logits: Unnormalised log-odds.
            hard: If True, use straight-through estimator.
            eps: Small constant for numerical stability.
        
        Returns:
            Soft or hard binary mask.
        """
        u = torch.rand_like(logits)
        u = torch.clamp(u, eps, 1.0 - eps)
        
        noise = torch.log(u) - torch.log(1 - u)
        y_soft = torch.sigmoid((logits + noise) / self.temperature)
        
        if hard:
            y_hard = (y_soft > 0.5).float()
            return (y_hard - y_soft).detach() + y_soft
        
        return y_soft
    
    @abstractmethod
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        """
        Apply mutation to individuals.
        
        Args:
            x: Individuals to mutate [n_pop, n_var].
            xl: Lower bounds [n_var] or scalar.
            xu: Upper bounds [n_var] or scalar.
        
        Returns:
            Mutated individuals [n_pop, n_var].
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
        Apply mutation.
        
        Args:
            x: Individuals to mutate [n_pop, n_var].
            xl: Lower bounds (or provide problem).
            xu: Upper bounds (or provide problem).
            problem: Problem instance with bounds.
        
        Returns:
            Mutated individuals [n_pop, n_var].
        """
        # Get bounds from problem if provided
        if problem is not None:
            xl = problem.xl
            xu = problem.xu
        
        # Default bounds if not provided
        if xl is None:
            xl = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
        if xu is None:
            xu = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        
        return self._mutate(x, xl, xu)
    
    def __call__(
        self,
        x: Tensor,
        xl: Optional[Tensor] = None,
        xu: Optional[Tensor] = None,
        problem: Optional["Problem"] = None,
    ) -> Tensor:
        """Apply mutation (alias for forward)."""
        return self.forward(x, xl, xu, problem)


# =============================================================================
# Polynomial Mutation
# =============================================================================

class PolynomialMutation(Mutation):
    """
    Polynomial mutation for real-coded GAs.
    
    Applies a polynomial perturbation to selected genes, with the
    perturbation bounded by the variable bounds. The distribution
    index eta controls the spread of mutations.
    
    Higher eta values produce mutations closer to the original
    value (more exploitation), while lower values produce more
    spread (more exploration).
    
    Args:
        eta: Distribution index (higher = smaller perturbations).
        prob: Mutation probability per gene. If None, defaults to 1/n_var.
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_eta: If True, eta is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Example:
        >>> mutation = PolynomialMutation(eta=20)
        >>> mutated = mutation(population, xl, xu)
    
    Reference:
        Deb & Deb (2014). Analysing Mutation Schemes for
        Real-Parameter Genetic Algorithms.
    """
    
    def __init__(
        self,
        eta: float = 20.0,
        prob: Optional[float] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_eta: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        # Eta parameter (log for positivity)
        if learn_eta and differentiable:
            self._log_eta = nn.Parameter(torch.tensor(eta).log())
        else:
            self.register_buffer("_log_eta", torch.tensor(eta).log())
    
    @property
    def eta(self) -> Tensor:
        """Current eta value."""
        return self._log_eta.exp()
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors with correct shape
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Get mutation mask
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            logits = prob_logits.unsqueeze(0).expand(n_pop, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            prob = self._get_prob(n_var, device)
            mask = (torch.rand(n_pop, n_var, device=device) < prob).float()
        
        # Compute polynomial perturbation
        eta = self.eta
        u = torch.rand(n_pop, n_var, device=device, dtype=dtype)
        
        # Polynomial distribution
        mut_pow = 1.0 / (eta + 1.0)
        
        delta = torch.where(
            u < 0.5,
            (2.0 * u).pow(mut_pow) - 1.0,
            1.0 - (2.0 * (1.0 - u)).pow(mut_pow)
        )
        
        # Scale by bounds range
        range_val = xu - xl
        perturbation = delta * range_val
        
        # Apply mutation with mask
        y = x + mask * perturbation
        
        return y
    
    def __repr__(self) -> str:
        prob_str = f"{self.prob.mean().item():.3f}" if self.prob is not None else "1/n_var"
        return (
            f"PolynomialMutation("
            f"eta={self.eta.item():.2f}, "
            f"prob={prob_str}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Gaussian Mutation
# =============================================================================

class GaussianMutation(Mutation):
    """
    Gaussian (normal) mutation.
    
    Adds Gaussian noise to selected genes. The standard deviation
    can be specified as a fixed value or as a fraction of the
    variable range.
    
    Args:
        sigma: Standard deviation of Gaussian noise.
        sigma_frac: Sigma as fraction of range (alternative to sigma).
            If both provided, sigma takes precedence.
        prob: Mutation probability per gene. If None, defaults to 1/n_var.
        differentiable: If True, use reparameterisation trick.
        temperature: Temperature for Binary-Concrete mask.
        learn_sigma: If True, sigma is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Example:
        >>> # Fixed sigma
        >>> mutation = GaussianMutation(sigma=0.1)
        >>> 
        >>> # Sigma as fraction of range
        >>> mutation = GaussianMutation(sigma_frac=0.1)  # sigma = 0.1 * (xu - xl)
    """
    
    def __init__(
        self,
        sigma: Optional[float] = None,
        sigma_frac: float = 0.1,
        prob: Optional[float] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_sigma: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        self._use_frac = sigma is None
        
        # Sigma parameter (log for positivity)
        sigma_val = sigma if sigma is not None else sigma_frac
        if learn_sigma and differentiable:
            self._log_sigma = nn.Parameter(torch.tensor(sigma_val).log())
        else:
            self.register_buffer("_log_sigma", torch.tensor(sigma_val).log())
    
    @property
    def sigma(self) -> Tensor:
        """Current sigma value."""
        return self._log_sigma.exp()
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Get mutation mask
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            logits = prob_logits.unsqueeze(0).expand(n_pop, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            prob = self._get_prob(n_var, device)
            mask = (torch.rand(n_pop, n_var, device=device) < prob).float()
        
        # Compute sigma (possibly scaled by range)
        if self._use_frac:
            sigma = self.sigma * (xu - xl)
        else:
            sigma = self.sigma
        
        # Gaussian noise (reparameterised)
        noise = torch.randn(n_pop, n_var, device=device, dtype=dtype) * sigma
        
        # Apply mutation with mask
        y = x + mask * noise
        
        return y
    
    def __repr__(self) -> str:
        prob_str = f"{self.prob.mean().item():.3f}" if self.prob is not None else "1/n_var"
        sigma_type = "frac" if self._use_frac else "fixed"
        return (
            f"GaussianMutation("
            f"sigma={self.sigma.item():.4f} ({sigma_type}), "
            f"prob={prob_str}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Uniform Mutation
# =============================================================================

class UniformMutation(Mutation):
    """
    Uniform mutation.
    
    Replaces selected genes with uniformly random values within
    the variable bounds. This is a more disruptive mutation than
    Gaussian or polynomial.
    
    Args:
        prob: Mutation probability per gene. If None, defaults to 1/n_var.
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Example:
        >>> mutation = UniformMutation(prob=0.05)
        >>> mutated = mutation(population, xl, xu)
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Get mutation mask
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            logits = prob_logits.unsqueeze(0).expand(n_pop, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            prob = self._get_prob(n_var, device)
            mask = (torch.rand(n_pop, n_var, device=device) < prob).float()
        
        # Random values within bounds
        random_vals = xl + (xu - xl) * torch.rand(n_pop, n_var, device=device, dtype=dtype)
        
        # Apply mutation with mask
        y = mask * random_vals + (1.0 - mask) * x
        
        return y
    
    def __repr__(self) -> str:
        prob_str = f"{self.prob.mean().item():.3f}" if self.prob is not None else "1/n_var"
        return f"UniformMutation(prob={prob_str})"


# =============================================================================
# Non-Uniform Mutation
# =============================================================================

class NonUniformMutation(Mutation):
    """
    Non-uniform mutation with decreasing perturbation.
    
    The perturbation magnitude decreases over generations, allowing
    large exploration early and fine-tuning later. Uses the formula:
        delta = (xu - x) * (1 - r^((1 - t/T)^b))  if coin flip
        delta = (x - xl) * (1 - r^((1 - t/T)^b))  otherwise
    
    where t is current generation, T is max generations, r is random,
    and b controls the decay rate.
    
    Args:
        max_generations: Maximum number of generations (T).
        b: Shape parameter controlling decay (higher = faster decay).
        prob: Mutation probability per gene.
        differentiable: If True, use differentiable operations.
        learn_b: If True, b is learnable.
    
    Example:
        >>> mutation = NonUniformMutation(max_generations=500, b=5.0)
        >>> mutation.set_generation(100)
        >>> mutated = mutation(population, xl, xu)
    
    Reference:
        Michalewicz (1996). Genetic Algorithms + Data Structures =
        Evolution Programs.
    """
    
    def __init__(
        self,
        max_generations: int = 500,
        b: float = 5.0,
        prob: Optional[float] = None,
        differentiable: bool = False,
        learn_b: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        self.max_generations = max_generations
        
        # Current generation (updated externally)
        self.register_buffer("_generation", torch.tensor(0))
        
        # B parameter (log for positivity)
        if learn_b and differentiable:
            self._log_b = nn.Parameter(torch.tensor(b).log())
        else:
            self.register_buffer("_log_b", torch.tensor(b).log())
    
    @property
    def b(self) -> Tensor:
        """Current b value."""
        return self._log_b.exp()
    
    @property
    def generation(self) -> int:
        """Current generation."""
        return self._generation.item()
    
    def set_generation(self, gen: int) -> None:
        """Set current generation."""
        self._generation.fill_(gen)
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Get mutation mask
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            logits = prob_logits.unsqueeze(0).expand(n_pop, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            prob = self._get_prob(n_var, device)
            mask = (torch.rand(n_pop, n_var, device=device) < prob).float()
        
        # Time decay factor
        t = self._generation.float()
        T = float(self.max_generations)
        decay = (1.0 - t / T).clamp(min=0.0)
        
        # Random factor
        r = torch.rand(n_pop, n_var, device=device, dtype=dtype)
        delta_factor = 1.0 - r.pow(decay.pow(self.b))
        
        # Direction: towards upper or lower bound
        direction = (torch.rand(n_pop, n_var, device=device) < 0.5).float()
        delta_upper = (xu - x) * delta_factor
        delta_lower = (x - xl) * delta_factor
        
        delta = direction * delta_upper - (1.0 - direction) * delta_lower
        
        # Apply mutation with mask
        y = x + mask * delta
        
        return y
    
    def __repr__(self) -> str:
        return (
            f"NonUniformMutation("
            f"max_gen={self.max_generations}, "
            f"b={self.b.item():.2f}, "
            f"generation={self.generation})"
        )


# =============================================================================
# Boundary Mutation
# =============================================================================

class BoundaryMutation(Mutation):
    """
    Boundary mutation.
    
    Resets selected genes to either the lower or upper bound
    (chosen randomly). Useful for exploring boundary regions
    of the search space.
    
    Args:
        prob: Mutation probability per gene. If None, defaults to 1/n_var.
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_prob: If True, mutation probability is learnable.
    
    Example:
        >>> mutation = BoundaryMutation(prob=0.01)
        >>> mutated = mutation(population, xl, xu)
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_prob: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=None,
        )
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        n_pop, n_var = x.shape
        device = x.device
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Get mutation mask
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            logits = prob_logits.unsqueeze(0).expand(n_pop, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            prob = self._get_prob(n_var, device)
            mask = (torch.rand(n_pop, n_var, device=device) < prob).float()
        
        # Choose lower or upper bound randomly
        use_upper = (torch.rand(n_pop, n_var, device=device) < 0.5).float()
        boundary_vals = use_upper * xu + (1.0 - use_upper) * xl
        
        # Apply mutation with mask
        y = mask * boundary_vals + (1.0 - mask) * x
        
        return y
    
    def __repr__(self) -> str:
        prob_str = f"{self.prob.mean().item():.3f}" if self.prob is not None else "1/n_var"
        return f"BoundaryMutation(prob={prob_str})"


# =============================================================================
# No Mutation (Identity)
# =============================================================================

class NoMutation(Mutation):
    """
    No mutation (identity operator).
    
    Returns input unchanged. Useful as a placeholder or when
    mutation should be disabled.
    
    Example:
        >>> mutation = NoMutation()
        >>> mutated = mutation(population, xl, xu)  # Returns population unchanged
    """
    
    def __init__(self) -> None:
        super().__init__(
            prob=0.0,
            differentiable=False,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        return x
    
    def __repr__(self) -> str:
        return "NoMutation()"


# =============================================================================
# Combined Mutation
# =============================================================================

class CombinedMutation(Mutation):
    """
    Combine multiple mutation operators.
    
    Applies mutations sequentially. Each mutation is applied
    with its own probability to the output of the previous one.
    
    Args:
        mutations: List of mutation operators to combine.
    
    Example:
        >>> mutation = CombinedMutation([
        ...     PolynomialMutation(eta=20, prob=0.1),
        ...     GaussianMutation(sigma=0.01, prob=0.05),
        ... ])
        >>> mutated = mutation(population, xl, xu)
    """
    
    def __init__(self, mutations: list) -> None:
        super().__init__(
            prob=1.0,
            differentiable=False,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        self.mutations = nn.ModuleList(mutations)
    
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
    ) -> Tensor:
        y = x
        for mutation in self.mutations:
            y = mutation(y, xl, xu)
        return y
    
    def __repr__(self) -> str:
        mut_strs = ", ".join(repr(m) for m in self.mutations)
        return f"CombinedMutation([{mut_strs}])"
