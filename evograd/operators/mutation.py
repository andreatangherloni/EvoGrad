"""
Mutation operators for introducing variation.

This module provides mutation operators that introduce random
perturbations to individuals, promoting exploration of the search
space. All operators support both classical and differentiable
(i.e., adaptive) modes.

Available mutations:
    - PolynomialMutation: Bounded polynomial mutation (GA)
    - GaussianMutation: Gaussian/normal perturbation
    - UniformMutation: Uniform random perturbation
    - NonUniformMutation: Decreasing perturbation over time
    - BoundaryMutation: Reset genes to boundary values
    - NoMutation: Identity (no mutation)
    - CombinedMutation: Chain multiple mutations

Differentiable Mode:
    When `adaptive=True`, mutation masks use Binary-Concrete
    (Gumbel-Sigmoid) relaxation, and perturbations use the
    reparameterisation trick for gradient flow.

Per-Individual/Per-Gene Parameters:
    All mutation operators support four parameter configurations via
    optional runtime overrides in forward(). This is essential for
    self-adaptive algorithms like SHADE, jDE, or self-adaptive GAs.
    
    Configurations:
        - Fixed (scalar): Same value for all individuals and genes
        - Per-gene [D]: Different value per gene, same across individuals
        - Per-individual [N]: Different value per individual, same across genes
        - Per-gene + Per-individual [N, D]: Full matrix, different for each
    
    Example:
        >>> # SHADE-style per-individual sigma/F
        >>> sigma_per_ind = torch.rand(pop_size) * 0.5  # [N]
        >>> mutated = mutation(population, xl, xu, sigma=sigma_per_ind)
        >>> 
        >>> # Per-gene mutation probability
        >>> prob_per_gene = torch.rand(n_var) * 0.2  # [D]
        >>> mutated = mutation(population, xl, xu, prob=prob_per_gene)

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
    ...     adaptive=True,
    ...     learn_eta=True,
    ... )
    >>> offspring = mutation(population, xl, xu)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import math
import torch
import torch.nn as nn
from torch import Tensor

from evograd.operators.relaxations import binary_concrete, expand_param, standard_normal, log_param

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = [
    "Mutation",
    "PolynomialMutation",
    "GaussianMutation",
    "UniformMutation",
    "NonUniformMutation",
    "BoundaryMutation",
    "NoMutation",
    "CombinedMutation",
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
        adaptive: If True, use Binary-Concrete for soft masks.
        temperature: Temperature for Binary-Concrete.
        learn_temperature: If True, temperature is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables (for per-gene probability).
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional parameter overrides that
        support four configurations:
        
        - scalar: Fixed value for all individuals and genes
        - [D] tensor: Per-gene values (same across individuals)
        - [N] tensor: Per-individual values (same across genes)
        - [N, D] tensor: Full matrix (different for each individual and gene)
        
        When an override is provided, it takes precedence over the stored
        parameter. This enables self-adaptive algorithms like SHADE.
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self._MIN_TEMPERATURE = 0.05
        self._MAX_TEMPERATURE = 10.0
        
        self.adaptive = adaptive
        self.n_var = n_var
        self._default_prob = prob is None
        
        # Temperature parameter (log for positivity)
        if learn_temperature and adaptive:
            self._log_temperature = log_param(temperature)
        else:
            self.register_buffer(
                "_log_temperature",
                torch.tensor(temperature).log()
            )
        
        # Mutation probability as logits
        # If prob is None, we'll compute 1/n_var at runtime
        if prob is not None:
            prob_logit = self._prob_to_logit(prob)
            if learn_prob and adaptive:
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
            prob = torch.sigmoid(self.prob_logits.to(device))
            if prob.dim() == 0:
                return prob.expand(n_var)
            return prob
        # Default: 1/n_var
        return torch.full((n_var,), 1.0 / n_var, device=device)
    
    def _get_prob_logits(self, n_var: int, device: torch.device) -> Tensor:
        """Get probability logits, computing default if needed."""
        if self.prob_logits is not None:
            logits = self.prob_logits.to(device)
            if logits.dim() == 0:
                return logits.expand(n_var)
            return logits
        # Default: 1/n_var
        default_prob = 1.0 / n_var
        return torch.full(
            (n_var,),
            self._prob_to_logit(default_prob),
            device=device
        )
    
    def _clamp_temperature(self):
        if hasattr(self, "_log_temperature") and self._log_temperature is not None:
            with torch.no_grad():
                self._log_temperature.clamp_(
                    math.log(self._MIN_TEMPERATURE),
                    math.log(self._MAX_TEMPERATURE),
                )
    
    @abstractmethod
    def _mutate(
        self,
        x: Tensor,
        xl: Tensor,
        xu: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Apply mutation to individuals.
        
        Args:
            x: Individuals to mutate [n_pop, n_var].
            xl: Lower bounds [n_var] or scalar.
            xu: Upper bounds [n_var] or scalar.
            **kwargs: Optional per-individual/per-gene parameter overrides.
        
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
        **kwargs,
    ) -> Tensor:
        """
        Apply mutation.
        
        Args:
            x: Individuals to mutate [n_pop, n_var].
            xl: Lower bounds (or provide problem).
            xu: Upper bounds (or provide problem).
            problem: Problem instance with bounds.
            **kwargs: Optional parameter overrides for per-individual or
                per-gene operation. Supported kwargs depend on the specific
                mutation operator (e.g., eta, sigma, prob).
                
                Each parameter can be:
                - scalar: Fixed value for all
                - [D] tensor: Per-gene values
                - [N] tensor: Per-individual values
                - [N, D] tensor: Full matrix
        
        Returns:
            Mutated individuals [n_pop, n_var].
        
        Example:
            >>> # Standard call (uses stored parameters)
            >>> mutated = mutation(population, xl, xu)
            >>> 
            >>> # Per-individual sigma override (for SHADE)
            >>> mutated = mutation(population, xl, xu, sigma=sigma_per_individual)
            >>> 
            >>> # Per-gene eta override
            >>> mutated = mutation(population, xl, xu, eta=eta_per_gene)
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
        
        self._clamp_temperature()
        return self._mutate(x, xl, xu, **kwargs)
    
    # Note: Do NOT override __call__. nn.Module.__call__ dispatches to
    # forward() and fires registered hooks (forward_pre_hooks, forward_hooks,
    # and the autograd profiler). Overriding __call__ would bypass all of these.


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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_eta: If True, eta is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - eta: Distribution index [scalar, D, N, or N×D]
        - prob: Mutation probability [scalar, D, N, or N×D]
    
    Example:
        >>> mutation = PolynomialMutation(eta=20)
        >>> mutated = mutation(population, xl, xu)
        >>> 
        >>> # Per-individual eta (for self-adaptive GA)
        >>> eta_per_ind = torch.rand(pop_size) * 20 + 5  # [N]
        >>> mutated = mutation(population, xl, xu, eta=eta_per_ind)
    
    Reference:
        Deb & Deb (2014). Analysing Mutation Schemes for
        Real-Parameter Genetic Algorithms.
    """
    
    def __init__(
        self,
        eta: float = 20.0,
        prob: Optional[float] = None,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_eta: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        # Eta parameter (log for positivity)
        if learn_eta and adaptive:
            self._log_eta = log_param(eta)
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
        eta: Optional[Tensor] = None,
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply polynomial mutation.
        
        Args:
            x: Individuals to mutate [N, D].
            xl: Lower bounds [D] or scalar.
            xu: Upper bounds [D] or scalar.
            eta: Optional distribution index override [scalar, D, N, or N×D].
            prob: Optional mutation probability override [scalar, D, N, or N×D].
        
        Returns:
            Mutated individuals [N, D].
        """
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors with correct shape
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Expand eta to [N, D]
        eta_expanded = expand_param(eta, self.eta, n_pop, n_var, device, dtype)
        
        # Expand prob to [N, D]
        default_prob = self._get_prob(n_var, device)
        prob_expanded = expand_param(prob, default_prob, n_pop, n_var, device, dtype)
        
        # Get mutation mask
        if self.adaptive:
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pop, n_var, device=device) < prob_expanded).float()
        
        # Compute polynomial perturbation
        u = torch.rand(n_pop, n_var, device=device, dtype=dtype)
        
        # Polynomial distribution
        mut_pow = 1.0 / (eta_expanded + 1.0)
        
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
            f"adaptive={self.adaptive})"
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
        adaptive: If True, use reparameterisation trick.
        temperature: Temperature for Binary-Concrete mask.
        learn_sigma: If True, sigma is learnable.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - sigma: Standard deviation [scalar, D, N, or N×D]
        - prob: Mutation probability [scalar, D, N, or N×D]
        
        This is essential for SHADE where each individual has its
        own F (scale factor) that can be used as sigma.
    
    Example:
        >>> # Fixed sigma
        >>> mutation = GaussianMutation(sigma=0.1)
        >>> 
        >>> # Sigma as fraction of range
        >>> mutation = GaussianMutation(sigma_frac=0.1)  # sigma = 0.1 * (xu - xl)
        >>> 
        >>> # Per-individual sigma (for SHADE/DE)
        >>> F_per_ind = torch.rand(pop_size) * 0.5 + 0.5  # [N]
        >>> mutated = mutation(population, xl, xu, sigma=F_per_ind)
    """
    
    def __init__(
        self,
        sigma: Optional[float] = None,
        sigma_frac: float = 0.1,
        prob: Optional[float] = None,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_sigma: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        self._use_frac = sigma is None
        
        # Sigma parameter (log for positivity)
        sigma_val = sigma if sigma is not None else sigma_frac
        if learn_sigma and adaptive:
            self._log_sigma = log_param(sigma_val)
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
        sigma: Optional[Tensor] = None,
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply Gaussian mutation.
        
        Args:
            x: Individuals to mutate [N, D].
            xl: Lower bounds [D] or scalar.
            xu: Upper bounds [D] or scalar.
            sigma: Optional standard deviation override [scalar, D, N, or N×D].
            prob: Optional mutation probability override [scalar, D, N, or N×D].
        
        Returns:
            Mutated individuals [N, D].
        """
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Expand prob to [N, D]
        default_prob = self._get_prob(n_var, device)
        prob_expanded = expand_param(prob, default_prob, n_pop, n_var, device, dtype)
        
        # Get mutation mask
        if self.adaptive:
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pop, n_var, device=device) < prob_expanded).float()
        
        # Compute sigma (possibly scaled by range)
        if sigma is not None:
            # Use provided sigma
            sigma_expanded = expand_param(sigma, self.sigma, n_pop, n_var, device, dtype)
            if self._use_frac:
                sigma_expanded = sigma_expanded * (xu - xl)
        else:
            # Use stored sigma
            if self._use_frac:
                sigma_expanded = self.sigma * (xu - xl)
                sigma_expanded = sigma_expanded.unsqueeze(0).expand(n_pop, -1)
            else:
                sigma_expanded = expand_param(None, self.sigma, n_pop, n_var, device, dtype)
        
        # Gaussian noise (reparameterised)
        noise = standard_normal(n_pop, n_var, device=device, dtype=dtype) * sigma_expanded
        
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
            f"adaptive={self.adaptive})"
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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_prob: If True, mutation probability is learnable.
        n_var: Number of variables.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - prob: Mutation probability [scalar, D, N, or N×D]
    
    Example:
        >>> mutation = UniformMutation(prob=0.05)
        >>> mutated = mutation(population, xl, xu)
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
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
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply uniform mutation.
        
        Args:
            x: Individuals to mutate [N, D].
            xl: Lower bounds [D] or scalar.
            xu: Upper bounds [D] or scalar.
            prob: Optional mutation probability override [scalar, D, N, or N×D].
        
        Returns:
            Mutated individuals [N, D].
        """
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Expand prob to [N, D]
        default_prob = self._get_prob(n_var, device)
        prob_expanded = expand_param(prob, default_prob, n_pop, n_var, device, dtype)
        
        # Get mutation mask
        if self.adaptive:
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pop, n_var, device=device) < prob_expanded).float()
        
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
    large exploration early and fine-tuning later. Uses the formula::

        delta = (xu - x) * (1 - r^((1 - t/T)^b))  if coin flip
        delta = (x - xl) * (1 - r^((1 - t/T)^b))  otherwise

    where t is current generation, T is max generations, r is random,
    and b controls the decay rate.
    
    Args:
        max_generations: Maximum number of generations (T).
        b: Shape parameter controlling decay (higher = faster decay).
        prob: Mutation probability per gene.
        adaptive: If True, use differentiable operations.
        learn_b: If True, b is learnable.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - prob: Mutation probability [scalar, D, N, or N×D]
        - progress: Progress ratio t/T override [scalar or N]
    
    Example:
        >>> mutation = NonUniformMutation(max_generations=500, b=5.0)
        >>> mutation.set_generation(100)
        >>> mutated = mutation(population, xl, xu)
        >>> 
        >>> # Per-individual progress (for heterogeneous adaptation)
        >>> progress_per_ind = torch.rand(pop_size)  # [N]
        >>> mutated = mutation(population, xl, xu, progress=progress_per_ind)
    
    Reference:
        Michalewicz (1996). Genetic Algorithms + Data Structures =
        Evolution Programs.
    """
    
    def __init__(
        self,
        max_generations: int = 500,
        b: float = 5.0,
        prob: Optional[float] = None,
        adaptive: bool = False,
        learn_b: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        self.max_generations = max_generations
        
        # Current generation (updated externally)
        self.register_buffer("_generation", torch.tensor(0))
        
        # B parameter (log for positivity)
        if learn_b and adaptive:
            self._log_b = log_param(b)
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
        prob: Optional[Tensor] = None,
        progress: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply non-uniform mutation.
        
        Args:
            x: Individuals to mutate [N, D].
            xl: Lower bounds [D] or scalar.
            xu: Upper bounds [D] or scalar.
            prob: Optional mutation probability override [scalar, D, N, or N×D].
            progress: Optional progress ratio (t/T) override [scalar or N].
        
        Returns:
            Mutated individuals [N, D].
        """
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Expand prob to [N, D]
        default_prob = self._get_prob(n_var, device)
        prob_expanded = expand_param(prob, default_prob, n_pop, n_var, device, dtype)
        
        # Get mutation mask
        if self.adaptive:
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pop, n_var, device=device) < prob_expanded).float()
        
        # Compute progress ratio
        if progress is not None:
            if isinstance(progress, Tensor):
                t_ratio = progress.to(device=device, dtype=dtype)
                if t_ratio.dim() == 0:
                    t_ratio = t_ratio.expand(n_pop)
            else:
                t_ratio = torch.full((n_pop,), progress, device=device, dtype=dtype)
        else:
            t_ratio = torch.full(
                (n_pop,),
                self.generation / max(self.max_generations, 1),
                device=device,
                dtype=dtype
            )
        
        # Expand t_ratio to [N, D]
        t_ratio = t_ratio.unsqueeze(1).expand(-1, n_var)
        
        # Compute non-uniform delta
        r = torch.rand(n_pop, n_var, device=device, dtype=dtype)
        
        # decay = (1 - t/T)^b
        decay = (1.0 - t_ratio).pow(self.b)
        
        # delta factor = 1 - r^decay
        delta_factor = 1.0 - r.pow(decay)
        
        # Direction (coin flip per gene)
        direction = (torch.rand(n_pop, n_var, device=device) < 0.5).float()
        
        # Compute perturbation
        delta_up = (xu - x) * delta_factor
        delta_down = (x - xl) * delta_factor
        delta = direction * delta_up - (1.0 - direction) * delta_down
        
        # Apply mutation with mask
        y = x + mask * delta
        
        return y
    
    def __repr__(self) -> str:
        prob_str = f"{self.prob.mean().item():.3f}" if self.prob is not None else "1/n_var"
        return (
            f"NonUniformMutation("
            f"b={self.b.item():.2f}, "
            f"prob={prob_str}, "
            f"max_gen={self.max_generations})"
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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_prob: If True, mutation probability is learnable.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - prob: Mutation probability [scalar, D, N, or N×D]
    
    Example:
        >>> mutation = BoundaryMutation(prob=0.01)
        >>> mutated = mutation(population, xl, xu)
    """
    
    def __init__(
        self,
        prob: Optional[float] = None,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_prob: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
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
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Apply boundary mutation.
        
        Args:
            x: Individuals to mutate [N, D].
            xl: Lower bounds [D] or scalar.
            xu: Upper bounds [D] or scalar.
            prob: Optional mutation probability override [scalar, D, N, or N×D].
        
        Returns:
            Mutated individuals [N, D].
        """
        n_pop, n_var = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure bounds are tensors
        if xl.dim() == 0:
            xl = xl.expand(n_var)
        if xu.dim() == 0:
            xu = xu.expand(n_var)
        
        # Expand prob to [N, D]
        default_prob = self._get_prob(n_var, device)
        prob_expanded = expand_param(prob, default_prob, n_pop, n_var, device, dtype)
        
        # Get mutation mask
        if self.adaptive:
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pop, n_var, device=device) < prob_expanded).float()
        
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
        >>> mutated = mutation(population, xl, xu)  # Returns unchanged
    """
    
    def __init__(self) -> None:
        super().__init__(
            prob=0.0,
            adaptive=False,
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
        **kwargs,
    ) -> Tensor:
        return x
    
    def __repr__(self) -> str:
        return "NoMutation()"


# =============================================================================
# Combined Mutation
# =============================================================================

class CombinedMutation(Mutation):
    """
    Combined mutation applying multiple operators sequentially.
    
    Chains multiple mutation operators together, applying them
    in sequence to the population.
    
    Args:
        mutations: List of mutation operators to chain.
    
    Example:
        >>> combined = CombinedMutation([
        ...     GaussianMutation(sigma=0.1, prob=0.5),
        ...     PolynomialMutation(eta=20, prob=0.1),
        ... ])
        >>> mutated = combined(population, xl, xu)
    
    Note:
        Per-individual parameters are NOT propagated to child
        operators. Use individual operators directly for per-
        individual control.
    """
    
    def __init__(
        self,
        mutations: List[Mutation],
    ) -> None:
        super().__init__(
            prob=1.0,
            adaptive=False,
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
        **kwargs,
    ) -> Tensor:
        """
        Apply all mutations sequentially.
        
        Note: kwargs are NOT passed to child operators.
        """
        y = x
        for mut in self.mutations:
            y = mut(y, xl, xu)
        return y
    
    def __repr__(self) -> str:
        muts_str = ", ".join(repr(m) for m in self.mutations)
        return f"CombinedMutation([{muts_str}])"
