"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for EvoGrad.

This module provides a fully differentiable CMA-ES implementation that
supports both classical and gradient-enabled optimisation modes.

CMA-ES evolves a multivariate Gaussian distribution N(μ, σ²C) through:
    1. Sampling: Generate offspring from the distribution
    2. Selection: Rank solutions by fitness
    3. Recombination: Update mean using weighted average of best solutions
    4. Adaptation: Update covariance matrix and step-size

The key components are:
    - μ (mean): Center of the search distribution
    - σ (sigma): Overall step-size (scale)
    - C (covariance): Shape of the distribution (via Cholesky factor L)
    - Evolution paths: p_σ and p_c for adaptation

Restart Strategies:
    - IPOP-CMA-ES: Restart with increasing population size
    - BIPOP-CMA-ES: Alternate between small (focused) and large (broad) populations

Modes:
    - adaptive=False, differentiable=False: Classical CMA-ES
    - adaptive=True, differentiable=False: Adaptation coefficients
        (cc, cs, c1, cmu, damps) are learnable via backpropagation
    - adaptive=False, differentiable=True: Mean μ is learnable
        via backpropagation
    - adaptive=True, differentiable=True: Both adaptation coefficients
        and mean are learnable

Example:
    >>> from evograd.algorithms import CMAES
    >>> from evograd.core import Problem, minimize
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Classical CMA-ES
    >>> cmaes = CMAES(pop_size=50, sigma=0.5)
    >>> result = minimize(problem, cmaes, max_evals=10000)
    >>> 
    >>> # Adaptive CMA-ES with learnable coefficients
    >>> cmaes = CMAES(pop_size=50, adaptive=True)
    >>> result = minimize(problem, cmaes, max_evals=10000)
    >>> 
    >>> # IPOP-CMA-ES with restarts
    >>> cmaes = CMAES(pop_size=50, restarts=9, incpopsize=2)
    >>> result = minimize(problem, cmaes, max_evals=100000)
    >>> 
    >>> # BIPOP-CMA-ES
    >>> cmaes = CMAES(pop_size=50, restarts=9, bipop=True)
    >>> result = minimize(problem, cmaes, max_evals=100000)

Reference:
    Hansen, N. & Ostermeier, A. (2001). Completely Derandomized 
    Self-Adaptation in Evolution Strategies. Evolutionary Computation.
    
    Hansen, N. (2009). Benchmarking a BI-Population CMA-ES on the 
    BBOB-2009 Function Testbed. GECCO Workshop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from evograd.core.algorithm import Algorithm

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = [
    "CMAES",
    "cmaes_default",
    "cmaes_small",
    "cmaes_large",
    "cmaes_adaptive",
    "cmaes_ipop",
    "cmaes_bipop",
]


def _expected_norm(dim: int) -> float:
    """
    Expected norm of a standard normal vector E[||N(0, I)||].
    
    Accurate to O(1/dim).
    """
    d = float(dim)
    return math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))


class RestartRegime(Enum):
    """Restart regime for BIPOP strategy."""
    LARGE = "large"  # IPOP-like increasing population
    SMALL = "small"  # Small focused population


@dataclass
class RestartState:
    """
    Tracks restart-related state for IPOP/BIPOP strategies.
    
    Attributes:
        n_restarts: Number of restarts performed so far.
        initial_pop_size: Original population size before any restarts.
        current_pop_size: Current population size after restarts.
        best_ever_x: Best solution found across all restarts.
        best_ever_f: Best fitness found across all restarts.
        regime: Current regime for BIPOP (LARGE or SMALL).
        large_evals: Total evaluations used by large populations (BIPOP).
        small_evals: Total evaluations used by small populations (BIPOP).
        small_n_restarts: Number of small-population restarts (BIPOP).
        run_history: History of best fitness per run.
    """
    n_restarts: int = 0
    initial_pop_size: int = 0
    current_pop_size: int = 0
    best_ever_x: Optional[Tensor] = None
    best_ever_f: float = float('inf')
    regime: RestartRegime = RestartRegime.LARGE
    large_evals: int = 0
    small_evals: int = 0
    small_n_restarts: int = 0
    run_history: List[float] = field(default_factory=list)


class CMAES(Algorithm):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    CMA-ES is a state-of-the-art evolutionary algorithm for continuous
    optimisation. It adapts a full covariance matrix to learn the
    structure of the objective function landscape.
    
    The algorithm samples from N(μ, σ²C) and adapts:
        - μ: Distribution mean (search center)
        - σ: Step-size (overall scale)
        - C: Covariance matrix (search shape)
    
    Args:
        pop_size: Population size (lambda). If None, uses 4 + floor(3*ln(n)).
        sigma: Initial step-size. Default: 0.5.
        x0: Initial mean. If None, uses center of bounds.
        cc: Cumulation constant for rank-one update. If None, uses default.
        cs: Cumulation constant for step-size control. If None, uses default.
        c1: Learning rate for rank-one update. If None, uses default.
        cmu: Learning rate for rank-mu update. If None, uses default.
        damps: Damping for step-size update. If None, uses default.
        restarts: Number of restarts with increasing population size (IPOP).
            Set to 0 for no restarts. Default: 0.
        restart_from_best: If True, restart from best-ever solution.
            If False, restart from random point. Default: False.
        incpopsize: Multiplier for population size increase after restart.
            Default: 2.
        bipop: If True, use BIPOP strategy alternating between small and
            large populations. Requires restarts > 0. Default: False.
        tolfun: Tolerance on function value for restart detection.
            Default: 1e-11.
        tolx: Tolerance on x change for restart detection. Default: 1e-11.
        sampling: Operator for initial population generation.
        repair: Repair operator for constraint handling.
        adaptive: If True, adaptation coefficients are learnable.
        differentiable: If True, mean μ is learnable.
        dtype: Tensor dtype.
    
    Attributes:
        mean: Current distribution mean μ.
        sigma: Current step-size σ.
        C: Current covariance matrix.
        L: Cholesky factor of C.
        p_sigma: Evolution path for step-size.
        p_c: Evolution path for covariance.
        restart_state: State tracking for IPOP/BIPOP restarts.
    
    Example:
        >>> # Classical CMA-ES
        >>> cmaes = CMAES(pop_size=50, sigma=0.3)
        >>> 
        >>> # Adaptive CMA-ES
        >>> cmaes = CMAES(adaptive=True)
        >>> 
        >>> # IPOP-CMA-ES with 9 restarts
        >>> cmaes = CMAES(restarts=9, incpopsize=2)
        >>> 
        >>> # BIPOP-CMA-ES
        >>> cmaes = CMAES(restarts=9, bipop=True)
        >>> 
        >>> # Differentiable mean
        >>> cmaes = CMAES(differentiable=True)
        >>> 
        >>> # Fully differentiable
        >>> cmaes = CMAES(adaptive=True, differentiable=True)
    """
    
    def __init__(
        self,
        pop_size: Optional[int] = None,
        sigma: float = 0.5,
        x0: Optional[Tensor] = None,
        cc: Optional[float] = None,
        cs: Optional[float] = None,
        c1: Optional[float] = None,
        cmu: Optional[float] = None,
        damps: Optional[float] = None,
        # Restart parameters
        restarts: int = 0,
        restart_from_best: bool = False,
        incpopsize: int = 2,
        bipop: bool = False,
        tolfun: float = 1e-11,
        tolx: float = 1e-11,
        # Standard parameters
        sampling: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        adaptive: bool = False,
        differentiable: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.adaptive = adaptive
        self._init_sigma = sigma
        self._init_x0 = x0
        self._init_cc = cc
        self._init_cs = cs
        self._init_c1 = c1
        self._init_cmu = cmu
        self._init_damps = damps
        
        # Restart parameters
        self._restarts = restarts
        self._restart_from_best = restart_from_best
        self._incpopsize = incpopsize
        self._bipop = bipop
        self._tolfun = tolfun
        self._tolx = tolx
        
        # Store pop_size for later (will be computed in _setup if None)
        self._requested_pop_size = pop_size
        
        # Use a default pop_size for base class, will be updated in _setup
        effective_pop_size = pop_size if pop_size is not None else 10
        
        # CMA-ES doesn't use standard EA operators
        super().__init__(
            pop_size=effective_pop_size,
            sampling=sampling,
            selection=None,
            crossover=None,
            mutation=None,
            survival=None,
            repair=repair,
            eliminate_duplicates=False,
            n_offsprings=effective_pop_size,
            differentiable=differentiable,
            adaptive=adaptive,
            dtype=dtype,
        )
        
        # Initialize restart state
        self.restart_state = RestartState()
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    def _setup(self) -> None:
        """CMA-ES specific setup after initialization."""
        n_var = self.problem.n_var
        
        # Compute default population size if not provided
        if self._requested_pop_size is None:
            self.pop_size = 4 + int(3 * math.log(n_var))
            self.n_offsprings = self.pop_size
        
        # Initialize restart state
        self.restart_state.initial_pop_size = self.pop_size
        self.restart_state.current_pop_size = self.pop_size
        
        # Number of parents for recombination
        self._mu = self.pop_size // 2
        
        # Compute recombination weights
        self._setup_weights()
        
        # Setup mean
        self._setup_mean(n_var)
        
        # Setup step-size (sigma)
        self._setup_sigma()
        
        # Setup covariance matrix (via Cholesky factor)
        self._setup_covariance(n_var)
        
        # Setup evolution paths
        self._setup_evolution_paths(n_var)
        
        # Setup adaptation coefficients
        self._setup_coefficients(n_var)
        
        # Expected norm of N(0,I)
        self._chi_n = _expected_norm(n_var)
        
        # Small epsilon for numerical stability
        self._eps = 1e-14 * (n_var ** 2)
        self._eps = min(self._eps, 1e-4)
        
        # History for restart detection
        self._fitness_history: List[float] = []
        self._generation_count = 0
    
    def _setup_weights(self) -> None:
        """Setup recombination weights."""
        mu = self._mu
        
        # Log weights: w_i = log(mu + 0.5) - log(i)
        raw_weights = torch.log(
            torch.tensor(mu + 0.5, device=self.device, dtype=self.dtype)
        ) - torch.log(
            torch.arange(1, mu + 1, device=self.device, dtype=self.dtype)
        )
        
        # Normalize
        weights = raw_weights / raw_weights.sum()
        self.register_buffer("_weights", weights)
        
        # Variance effective selection mass
        self._mu_eff = float(1.0 / (weights ** 2).sum())
    
    def _setup_mean(self, n_var: int, restart: bool = False) -> None:
        """Setup distribution mean."""
        if restart and self._restart_from_best and self.restart_state.best_ever_x is not None:
            # Restart from best-ever solution
            mean = self.restart_state.best_ever_x.clone()
        elif self._init_x0 is not None and not restart:
            mean = self._init_x0.to(device=self.device, dtype=self.dtype)
        else:
            # Random point within bounds or center
            if restart:
                # Random initialization for restart
                mean = self.xl + torch.rand(n_var, device=self.device, dtype=self.dtype) * (self.xu - self.xl)
            else:
                # Center of bounds
                mean = 0.5 * (self.xl + self.xu)
        
        # Mean is always a parameter (for differentiable mode)
        # But gradients only flow when differentiable=True
        if self.differentiable:
            if hasattr(self, '_mean') and isinstance(self._mean, nn.Parameter):
                with torch.no_grad():
                    self._mean.copy_(mean)
            else:
                self._mean = nn.Parameter(mean.clone())
        else:
            if hasattr(self, '_mean'):
                self._mean.copy_(mean)
            else:
                self.register_buffer("_mean", mean.clone())
    
    def _setup_sigma(self, restart: bool = False) -> None:
        """Setup step-size."""
        sigma_val = self._init_sigma
        
        if self.adaptive:
            if hasattr(self, '_log_sigma') and isinstance(self._log_sigma, nn.Parameter):
                with torch.no_grad():
                    self._log_sigma.fill_(math.log(sigma_val))
            else:
                self._log_sigma = nn.Parameter(
                    torch.tensor(sigma_val, device=self.device, dtype=self.dtype).log()
                )
        else:
            if hasattr(self, '_sigma_buffer'):
                self._sigma_buffer.fill_(sigma_val)
            else:
                self.register_buffer(
                    "_sigma_buffer",
                    torch.tensor(sigma_val, device=self.device, dtype=self.dtype)
                )
    
    def _setup_covariance(self, n_var: int, restart: bool = False) -> None:
        """Setup covariance matrix via Cholesky factor."""
        # Initialize as identity (C = I, L = I)
        L_init = torch.eye(n_var, device=self.device, dtype=self.dtype)
        
        if self.adaptive:
            if hasattr(self, '_L') and isinstance(self._L, nn.Parameter):
                with torch.no_grad():
                    self._L.copy_(L_init)
            else:
                self._L = nn.Parameter(L_init)
        else:
            if hasattr(self, '_L'):
                self._L.copy_(L_init)
            else:
                self.register_buffer("_L", L_init)
    
    def _setup_evolution_paths(self, n_var: int, restart: bool = False) -> None:
        """Setup evolution paths."""
        zeros = torch.zeros(n_var, device=self.device, dtype=self.dtype)
        
        if hasattr(self, '_p_sigma'):
            self._p_sigma.copy_(zeros)
        else:
            self.register_buffer("_p_sigma", zeros.clone())
        
        if hasattr(self, '_p_c'):
            self._p_c.copy_(zeros)
        else:
            self.register_buffer("_p_c", zeros.clone())
    
    def _setup_coefficients(self, n_var: int) -> None:
        """Setup adaptation coefficients."""
        mu_eff = self._mu_eff
        d = float(n_var)
        
        # Default values following Hansen's recommendations
        cc_default = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs_default = (mu_eff + 2) / (d + mu_eff + 5)
        c1_default = 2 / ((d + 1.3) ** 2 + mu_eff)
        cmu_default = min(
            1 - c1_default,
            2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff)
        )
        damps_default = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (d + 1)) - 1) + cs_default
        
        # Use provided values or defaults
        cc = self._init_cc if self._init_cc is not None else cc_default
        cs = self._init_cs if self._init_cs is not None else cs_default
        c1 = self._init_c1 if self._init_c1 is not None else c1_default
        cmu = self._init_cmu if self._init_cmu is not None else cmu_default
        damps = self._init_damps if self._init_damps is not None else damps_default
        
        if self.adaptive:
            # Store as logits for bounded optimization
            if not hasattr(self, '_cc_logit'):
                self._cc_logit = nn.Parameter(
                    self._to_logit(cc).to(device=self.device, dtype=self.dtype)
                )
                self._cs_logit = nn.Parameter(
                    self._to_logit(cs).to(device=self.device, dtype=self.dtype)
                )
                self._c1_logit = nn.Parameter(
                    self._to_logit(c1).to(device=self.device, dtype=self.dtype)
                )
                self._cmu_logit = nn.Parameter(
                    self._to_logit(cmu).to(device=self.device, dtype=self.dtype)
                )
                # damps is positive, store as log
                self._log_damps = nn.Parameter(
                    torch.tensor(damps, device=self.device, dtype=self.dtype).log()
                )
        else:
            if not hasattr(self, '_cc'):
                self.register_buffer("_cc", torch.tensor(cc, device=self.device, dtype=self.dtype))
                self.register_buffer("_cs", torch.tensor(cs, device=self.device, dtype=self.dtype))
                self.register_buffer("_c1", torch.tensor(c1, device=self.device, dtype=self.dtype))
                self.register_buffer("_cmu", torch.tensor(cmu, device=self.device, dtype=self.dtype))
                self.register_buffer("_damps", torch.tensor(damps, device=self.device, dtype=self.dtype))
    
    @staticmethod
    def _to_logit(p: float, eps: float = 1e-7) -> Tensor:
        """Convert probability/rate to logit."""
        p = max(min(p, 1 - eps), eps)
        return torch.tensor(math.log(p / (1 - p)))
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def population(self) -> Tensor:
        """Current population (sampled points)."""
        return self._population
    
    @property
    def fitness(self) -> Tensor:
        """Current fitness values."""
        return self.state.fitness
    
    @property
    def mean(self) -> Tensor:
        """Current distribution mean."""
        return self._mean
    
    @property
    def sigma(self) -> Tensor:
        """Current step-size."""
        if self.adaptive:
            return self._log_sigma.exp()
        return self._sigma_buffer
    
    @property
    def L(self) -> Tensor:
        """Cholesky factor of covariance matrix."""
        return torch.tril(self._L)
    
    @property
    def C(self) -> Tensor:
        """Covariance matrix C = L @ L.T."""
        L = self.L
        return L @ L.T
    
    @property
    def p_sigma(self) -> Tensor:
        """Evolution path for step-size control."""
        return self._p_sigma
    
    @property
    def p_c(self) -> Tensor:
        """Evolution path for covariance update."""
        return self._p_c
    
    @property
    def cc(self) -> Tensor:
        """Cumulation constant for rank-one update."""
        if self.adaptive:
            return torch.sigmoid(self._cc_logit)
        return self._cc
    
    @property
    def cs(self) -> Tensor:
        """Cumulation constant for step-size control."""
        if self.adaptive:
            return torch.sigmoid(self._cs_logit)
        return self._cs
    
    @property
    def c1(self) -> Tensor:
        """Learning rate for rank-one update."""
        if self.adaptive:
            return torch.sigmoid(self._c1_logit)
        return self._c1
    
    @property
    def cmu(self) -> Tensor:
        """Learning rate for rank-mu update."""
        if self.adaptive:
            return torch.sigmoid(self._cmu_logit)
        return self._cmu
    
    @property
    def damps(self) -> Tensor:
        """Damping for step-size update."""
        if self.adaptive:
            return self._log_damps.exp()
        return self._damps
    
    @property
    def n_restarts_remaining(self) -> int:
        """Number of restarts remaining."""
        return max(0, self._restarts - self.restart_state.n_restarts)
    
    # =========================================================================
    # Core CMA-ES Methods
    # =========================================================================
    
    def _sample(self) -> tuple:
        """
        Sample offspring from N(μ, σ²C) using reparameterization.
        
        x = μ + σ * L @ z, where z ~ N(0, I)
        
        Returns:
            Tuple of (offspring, z_vectors) where z are the standard normal samples.
        """
        N, D = self.pop_size, self.n_var
        
        # Sample standard normal
        z = torch.randn(N, D, device=self.device, dtype=self.dtype)
        
        # Transform: y = L @ z
        L = self.L
        y = (L @ z.T).T  # [N, D]
        
        # Scale and shift: x = μ + σ * y
        offspring = self.mean + self.sigma * y
        
        return offspring, z, y
    
    def _infill(self) -> Tensor:
        """
        Generate offspring through sampling.
        
        Returns:
            Offspring population [pop_size, n_var].
        """
        # Sample from distribution
        offspring, z, y = self._sample()
        
        # Store for later use in adaptation
        self._pending_z = z
        self._pending_y = y
        
        # Repair bounds
        if self.repair is not None:
            offspring = self.repair(offspring, self.xl, self.xu)
        else:
            offspring = torch.clamp(offspring, self.xl, self.xu)
        
        return offspring
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Update CMA-ES state based on offspring evaluation.
        
        Args:
            offspring: Offspring population [pop_size, n_var].
            offspring_fitness: Fitness values [pop_size].
        """
        N, D = self.pop_size, self.n_var
        mu = self._mu
        
        # Sort by fitness (ascending for minimization)
        sorted_indices = torch.argsort(offspring_fitness)
        selected_indices = sorted_indices[:mu]
        
        # Get selected y vectors (in transformed space)
        y_selected = self._pending_y[selected_indices]  # [mu, D]
        
        # Weighted recombination in y-space
        y_w = (self._weights.unsqueeze(-1) * y_selected).sum(dim=0)  # [D]
        
        # Update mean
        new_mean = self.mean + self.sigma * y_w
        
        # Update evolution paths
        new_p_sigma, new_p_c, h_sigma = self._update_evolution_paths(y_w)
        
        # Update covariance
        new_L = self._update_covariance(y_selected, new_p_c, h_sigma)
        
        # Update step-size
        new_sigma = self._update_sigma(new_p_sigma)
        
        # Commit updates
        self._commit_updates(
            new_mean=new_mean,
            new_sigma=new_sigma,
            new_L=new_L,
            new_p_sigma=new_p_sigma,
            new_p_c=new_p_c,
            offspring=offspring,
            offspring_fitness=offspring_fitness,
        )
        
        # Update generation count
        self._generation_count += 1
        
        # Track fitness history for restart detection
        best_fitness = float(offspring_fitness.min())
        self._fitness_history.append(best_fitness)
        
        # Update best-ever solution
        if best_fitness < self.restart_state.best_ever_f:
            best_idx = offspring_fitness.argmin()
            self.restart_state.best_ever_f = best_fitness
            self.restart_state.best_ever_x = offspring[best_idx].clone().detach()
        
        # Check for restart (only if not in differentiable mode)
        if self._restarts > 0 and not self.differentiable:
            self._check_and_perform_restart()
        
        # Cleanup
        del self._pending_z
        del self._pending_y
    
    def _update_evolution_paths(self, y_w: Tensor) -> tuple:
        """
        Update evolution paths p_σ and p_c.
        
        Args:
            y_w: Weighted mean of selected y vectors [n_var].
        
        Returns:
            Tuple of (new_p_sigma, new_p_c, h_sigma).
        """
        D = self.n_var
        mu_eff = self._mu_eff
        
        # Compute C^(-1/2) @ y_w using L^(-1) @ y_w
        L = self.L
        # Solve L @ z_w = y_w for z_w (equivalent to L^(-1) @ y_w)
        z_w = torch.linalg.solve_triangular(L, y_w.unsqueeze(-1), upper=False).squeeze(-1)
        
        # Update p_sigma (conjugate evolution path)
        cs = self.cs
        new_p_sigma = (1 - cs) * self.p_sigma + math.sqrt(cs * (2 - cs) * mu_eff) * z_w
        
        # Heaviside function h_sigma (smooth approximation)
        norm_p_sigma = new_p_sigma.norm()
        threshold = 1.4 + 2.0 / (D + 1)
        h_sigma = torch.sigmoid(10 * (threshold - norm_p_sigma / self._chi_n))
        
        # Update p_c (evolution path for covariance)
        cc = self.cc
        new_p_c = (1 - cc) * self.p_c + h_sigma * math.sqrt(cc * (2 - cc) * mu_eff) * y_w
        
        return new_p_sigma, new_p_c, h_sigma
    
    def _update_covariance(
        self,
        y_selected: Tensor,
        new_p_c: Tensor,
        h_sigma: Tensor,
    ) -> Tensor:
        """
        Update covariance matrix C.
        
        Args:
            y_selected: Selected y vectors [mu, n_var].
            new_p_c: New evolution path [n_var].
            h_sigma: Heaviside indicator.
        
        Returns:
            New Cholesky factor L.
        """
        D = self.n_var
        c1 = self.c1
        cmu = self.cmu
        cc = self.cc
        
        # Current covariance
        L = self.L
        C = L @ L.T
        
        # Rank-one update
        rank_one = torch.outer(new_p_c, new_p_c)
        
        # Rank-mu update
        rank_mu = (
            self._weights.unsqueeze(-1).unsqueeze(-1) * 
            y_selected.unsqueeze(-1) * y_selected.unsqueeze(-2)
        ).sum(dim=0)
        
        # Old C decay correction for h_sigma < 1
        c1_correction = c1 * (1 - h_sigma ** 2) * cc * (2 - cc)
        
        # New covariance
        C_new = (
            (1 - c1 - cmu + c1_correction) * C +
            c1 * rank_one +
            cmu * rank_mu
        )
        
        # Ensure symmetry
        C_new = 0.5 * (C_new + C_new.T)
        
        # Add small diagonal for numerical stability
        C_new = C_new + self._eps * torch.eye(D, device=self.device, dtype=self.dtype)
        
        # Compute new Cholesky factor with robust fallback
        L_new = self._safe_cholesky(C_new)
        
        return L_new
    
    def _safe_cholesky(self, C: Tensor) -> Tensor:
        """
        Compute Cholesky decomposition with robust fallback.
        
        If standard Cholesky fails, applies eigenvalue correction
        and regularization to ensure positive definiteness.
        
        Args:
            C: Covariance matrix [n_var, n_var].
        
        Returns:
            Lower triangular Cholesky factor L where C ≈ L @ L.T.
        """
        D = C.shape[0]
        
        # Attempt 1: Direct Cholesky
        try:
            return torch.linalg.cholesky(C)
        except RuntimeError:
            pass
        
        # Attempt 2: Eigendecomposition with correction
        try:
            # Use eigh for symmetric matrices (more stable than eig)
            eigval, eigvec = torch.linalg.eigh(C)
            
            # Clamp eigenvalues to be positive
            min_eigval = max(1e-10, float(eigval.max()) * 1e-12)
            eigval_fixed = torch.clamp(eigval, min=min_eigval)
            
            # Reconstruct covariance
            C_fixed = eigvec @ torch.diag(eigval_fixed) @ eigvec.T
            
            # Force symmetry (numerical errors can break it)
            C_fixed = 0.5 * (C_fixed + C_fixed.T)
            
            # Add small diagonal regularization
            reg = 1e-8 * eigval_fixed.max() * torch.eye(D, device=C.device, dtype=C.dtype)
            C_fixed = C_fixed + reg
            
            return torch.linalg.cholesky(C_fixed)
        except RuntimeError:
            pass
        
        # Attempt 3: More aggressive regularization
        try:
            eigval, eigvec = torch.linalg.eigh(C)
            eigval_fixed = torch.clamp(eigval, min=1e-6)
            C_fixed = eigvec @ torch.diag(eigval_fixed) @ eigvec.T
            C_fixed = 0.5 * (C_fixed + C_fixed.T)
            
            # Stronger regularization
            reg = 1e-4 * torch.eye(D, device=C.device, dtype=C.dtype)
            C_fixed = C_fixed + reg
            
            return torch.linalg.cholesky(C_fixed)
        except RuntimeError:
            pass
        
        # Attempt 4: Last resort - reset to scaled identity
        # Preserve the trace (total variance) from original matrix
        trace = torch.trace(C).clamp(min=1e-6)
        scale = torch.sqrt(trace / D)
        L_identity = scale * torch.eye(D, device=C.device, dtype=C.dtype)
        
        return L_identity
    
    def _update_sigma(self, new_p_sigma: Tensor) -> Tensor:
        """
        Update step-size using CSA (Cumulative Step-size Adaptation).
        
        Args:
            new_p_sigma: New evolution path for step-size.
        
        Returns:
            New step-size.
        """
        cs = self.cs
        damps = self.damps
        
        # Step-size update factor
        norm_p_sigma = new_p_sigma.norm()
        factor = torch.exp((cs / damps) * (norm_p_sigma / self._chi_n - 1))
        
        new_sigma = self.sigma * factor
        
        return new_sigma
    
    def _commit_updates(
        self,
        new_mean: Tensor,
        new_sigma: Tensor,
        new_L: Tensor,
        new_p_sigma: Tensor,
        new_p_c: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
    ) -> None:
        """Commit all updates to state."""
        with torch.no_grad():
            # Update mean
            if isinstance(self._mean, nn.Parameter):
                self._mean.copy_(new_mean)
            else:
                self._mean.copy_(new_mean)
            
            # Update sigma
            if self.adaptive:
                self._log_sigma.copy_(new_sigma.log())
            else:
                self._sigma_buffer.copy_(new_sigma)
            
            # Update Cholesky factor
            if isinstance(self._L, nn.Parameter):
                self._L.copy_(new_L)
            else:
                self._L.copy_(new_L)
            
            # Update evolution paths
            self._p_sigma.copy_(new_p_sigma)
            self._p_c.copy_(new_p_c)
            
            # Update population
            self._population.copy_(offspring)
        
        # Update fitness
        self.state.fitness = offspring_fitness
        self.state.population = self._population
        
        # Update best solution
        self.state.update_best(offspring, offspring_fitness)
    
    # =========================================================================
    # Restart Methods (IPOP/BIPOP)
    # =========================================================================
    
    def _should_restart(self) -> bool:
        """
        Check if restart conditions are met.
        
        Returns:
            True if algorithm should restart.
        """
        # Need enough history
        if len(self._fitness_history) < 10:
            return False
        
        # Check tolerance on function values (stagnation)
        recent = self._fitness_history[-10:]
        if max(recent) - min(recent) < self._tolfun:
            return True
        
        # Check tolerance on sigma (step-size too small)
        sigma_val = float(self.sigma)
        if sigma_val < self._tolx:
            return True
        
        # Check condition number of C (degenerate distribution)
        try:
            L = self.L
            C = L @ L.T
            eigvals = torch.linalg.eigvalsh(C)
            cond = eigvals.max() / eigvals.min().clamp(min=1e-30)
            if cond > 1e14:
                return True
        except RuntimeError:
            return True
        
        return False
    
    def _check_and_perform_restart(self) -> None:
        """Check restart conditions and perform restart if needed."""
        if not self._should_restart():
            return
        
        # Check if we have restarts remaining
        if self.restart_state.n_restarts >= self._restarts:
            return
        
        # Record this run's result
        if self._fitness_history:
            self.restart_state.run_history.append(min(self._fitness_history))
        
        # Determine new population size and regime
        if self._bipop:
            self._bipop_restart()
        else:
            self._ipop_restart()
    
    def _ipop_restart(self) -> None:
        """Perform IPOP restart (increasing population)."""
        n_var = self.n_var
        
        # Increase population size
        new_pop_size = self.restart_state.current_pop_size * self._incpopsize
        self.restart_state.current_pop_size = new_pop_size
        self.restart_state.n_restarts += 1
        
        # Update population size
        self.pop_size = new_pop_size
        self.n_offsprings = new_pop_size
        self._mu = new_pop_size // 2
        
        # Re-setup weights for new population size
        self._setup_weights()
        
        # Reset CMA-ES state
        self._setup_mean(n_var, restart=True)
        self._setup_sigma(restart=True)
        self._setup_covariance(n_var, restart=True)
        self._setup_evolution_paths(n_var, restart=True)
        
        # Re-initialize population buffer + state tensors
        new_pop = torch.zeros(self.pop_size, n_var, device=self.device, dtype=self.dtype)
        if hasattr(self, "_population") and self._population.shape == new_pop.shape:
            self._population.copy_(new_pop)
        else:
            self._population = new_pop

        self.state.population = self._population
        self.state.fitness = torch.full(
            (self.pop_size,), float("inf"), device=self.device, dtype=self.dtype
        )

        # Clear fitness history
        self._fitness_history = []
        self._generation_count = 0
    
    def _bipop_restart(self) -> None:
        """
        Perform BIPOP restart (alternating small/large populations).
        
        BIPOP alternates between:
        - Large population regime: progressively increasing (like IPOP)
        - Small population regime: small focused search
        
        The regime is chosen based on which has used fewer evaluations.
        """
        n_var = self.n_var
        
        # Update evaluation counts for current regime
        evals_this_run = self._generation_count * self.pop_size
        if self.restart_state.regime == RestartRegime.LARGE:
            self.restart_state.large_evals += evals_this_run
        else:
            self.restart_state.small_evals += evals_this_run
        
        # Decide next regime based on evaluation budget balance
        if self.restart_state.small_evals <= self.restart_state.large_evals:
            # Do small-population restart
            self.restart_state.regime = RestartRegime.SMALL
            self.restart_state.small_n_restarts += 1
            
            # Small population: use default size with some randomization
            # Population size uniform in [2, default_size]
            default_size = 4 + int(3 * math.log(n_var))
            new_pop_size = max(2, int(torch.rand(1).item() * default_size))
            
            # Sigma for small regime: smaller initial step-size for focused search
            small_sigma = self._init_sigma * (0.01 + 0.49 * torch.rand(1).item())
            if self.adaptive:
                with torch.no_grad():
                    self._log_sigma.fill_(math.log(small_sigma))
            else:
                self._sigma_buffer.fill_(small_sigma)
        else:
            # Do large-population restart (IPOP-style)
            self.restart_state.regime = RestartRegime.LARGE
            self.restart_state.n_restarts += 1
            
            # Increase population size
            new_pop_size = self.restart_state.current_pop_size * self._incpopsize
            self.restart_state.current_pop_size = new_pop_size
        
        # Update population size
        self.pop_size = new_pop_size
        self.n_offsprings = new_pop_size
        self._mu = new_pop_size // 2
        
        # Re-setup weights for new population size
        self._setup_weights()
        
        # Reset CMA-ES state
        self._setup_mean(n_var, restart=True)
        if self.restart_state.regime == RestartRegime.LARGE:
            self._setup_sigma(restart=True)
        self._setup_covariance(n_var, restart=True)
        self._setup_evolution_paths(n_var, restart=True)
        
        # Re-initialize population buffer + state tensors
        new_pop = torch.zeros(self.pop_size, n_var, device=self.device, dtype=self.dtype)
        if hasattr(self, "_population") and self._population.shape == new_pop.shape:
            self._population.copy_(new_pop)
        else:
            self._population = new_pop

        self.state.population = self._population
        self.state.fitness = torch.full(
            (self.pop_size,), float("inf"), device=self.device, dtype=self.dtype
        )

        # Clear fitness history
        self._fitness_history = []
        self._generation_count = 0
    
    # =========================================================================
    # Hyperparameter Management
    # =========================================================================
    
    @torch.no_grad()
    def _clamp_hyperparams(self) -> None:
        """Clamp learnable hyperparameters to valid ranges."""
        if self.adaptive:
            # Sigma in (1e-10, 1e10)
            self._log_sigma.clamp_(min=-23, max=23)
            
            # Ensure c1 + cmu <= 1
            c1 = torch.sigmoid(self._c1_logit)
            cmu = torch.sigmoid(self._cmu_logit)
            total = c1 + cmu
            if total > 0.99:
                scale = 0.99 / total
                self._c1_logit.fill_(self._to_logit(float(c1 * scale)))
                self._cmu_logit.fill_(self._to_logit(float(cmu * scale)))
    
    def update_state(self) -> None:
        """Commit pending changes and clamp hyperparameters."""
        super().update_state()
        self._clamp_hyperparams()
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameter values."""
        return {
            'pop_size': self.pop_size,
            'mu': self._mu,
            'mu_eff': self._mu_eff,
            'sigma': float(self.sigma.item()),
            'cc': float(self.cc.item()),
            'cs': float(self.cs.item()),
            'c1': float(self.c1.item()),
            'cmu': float(self.cmu.item()),
            'damps': float(self.damps.item()),
            'adaptive': self.adaptive,
            'differentiable': self.differentiable,
            'restarts': self._restarts,
            'restart_from_best': self._restart_from_best,
            'incpopsize': self._incpopsize,
            'bipop': self._bipop,
            'n_restarts_done': self.restart_state.n_restarts,
            'current_pop_size': self.restart_state.current_pop_size,
            'best_ever_f': self.restart_state.best_ever_f,
        }
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        parts = [
            f"CMAES(pop_size={self.pop_size}",
            f"sigma={float(self._init_sigma):.4f}",
        ]
        if self._restarts > 0:
            if self._bipop:
                parts.append(f"bipop=True")
            else:
                parts.append(f"restarts={self._restarts}")
            parts.append(f"incpopsize={self._incpopsize}")
        parts.append(f"adaptive={self.adaptive}")
        parts.append(f"differentiable={self.differentiable})")
        return ", ".join(parts)


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def cmaes_default(
    pop_size: Optional[int] = None,
    sigma: float = 0.5,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> CMAES:
    """
    Create CMA-ES with default settings.
    
    Population size defaults to 4 + floor(3*ln(n)) where n is
    the number of variables.
    
    Args:
        pop_size: Population size. If None, computed from n_var.
        sigma: Initial step-size.
        adaptive: If True, adaptation coefficients are learnable.
        differentiable: If True, mean is learnable.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance.
    """
    return CMAES(
        pop_size=pop_size,
        sigma=sigma,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def cmaes_small(
    sigma: float = 0.3,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> CMAES:
    """
    Create CMA-ES with small population for fast convergence.
    
    Uses minimum recommended population size.
    
    Args:
        sigma: Initial step-size.
        adaptive: If True, adaptation coefficients are learnable.
        differentiable: If True, mean is learnable.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance.
    """
    # pop_size will be computed as 4 + 3*ln(n) in _setup
    return CMAES(
        pop_size=None,
        sigma=sigma,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def cmaes_large(
    pop_size_factor: float = 2.0,
    sigma: float = 0.5,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> CMAES:
    """
    Create CMA-ES with larger population for more robust search.
    
    Multiplies the default population size by a factor.
    
    Note: pop_size is computed after initialization when n_var is known.
    For now, this creates a CMAES that will use a larger population.
    
    Args:
        pop_size_factor: Multiplier for default population size.
        sigma: Initial step-size.
        adaptive: If True, adaptation coefficients are learnable.
        differentiable: If True, mean is learnable.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance.
    """
    # Store factor for custom handling
    cmaes = CMAES(
        pop_size=None,
        sigma=sigma,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )
    cmaes._pop_size_factor = pop_size_factor
    return cmaes


def cmaes_adaptive(
    pop_size: Optional[int] = None,
    sigma: float = 0.5,
    differentiable: bool = False,
    **kwargs,
) -> CMAES:
    """
    Create CMA-ES with adaptive (learnable) hyperparameters.
    
    The adaptation coefficients (cc, cs, c1, cmu, damps) are
    learned via backpropagation.
    
    Args:
        pop_size: Population size. If None, computed from n_var.
        sigma: Initial step-size.
        differentiable: If True, mean is also learnable.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance with adaptive=True.
    """
    return CMAES(
        pop_size=pop_size,
        sigma=sigma,
        adaptive=True,
        differentiable=differentiable,
        **kwargs,
    )


def cmaes_ipop(
    restarts: int = 9,
    incpopsize: int = 2,
    restart_from_best: bool = False,
    sigma: float = 0.5,
    **kwargs,
) -> CMAES:
    """
    Create IPOP-CMA-ES with increasing population restarts.
    
    IPOP-CMA-ES restarts the algorithm with doubled population
    size after convergence, allowing escape from local optima.
    
    Args:
        restarts: Number of restarts to perform. Default: 9.
        incpopsize: Population size multiplier after restart. Default: 2.
        restart_from_best: If True, restart from best solution found.
            If False, restart from random point. Default: False.
        sigma: Initial step-size.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance with IPOP restart strategy.
    
    Reference:
        Auger, A. & Hansen, N. (2005). A Restart CMA Evolution Strategy
        With Increasing Population Size. CEC 2005.
    """
    return CMAES(
        sigma=sigma,
        restarts=restarts,
        restart_from_best=restart_from_best,
        incpopsize=incpopsize,
        bipop=False,
        **kwargs,
    )


def cmaes_bipop(
    restarts: int = 9,
    incpopsize: int = 2,
    sigma: float = 0.5,
    **kwargs,
) -> CMAES:
    """
    Create BIPOP-CMA-ES with alternating population sizes.
    
    BIPOP-CMA-ES alternates between:
    - Small populations: Focused search for exploiting local structure
    - Large populations: Broad search (IPOP-style) for exploration
    
    This strategy performs well on both functions with many regularly
    or irregularly arranged local optima.
    
    Args:
        restarts: Number of large-population restarts. Default: 9.
        incpopsize: Population size multiplier for large regime. Default: 2.
        sigma: Initial step-size.
        **kwargs: Additional arguments passed to CMAES.
    
    Returns:
        Configured CMAES instance with BIPOP restart strategy.
    
    Reference:
        Hansen, N. (2009). Benchmarking a BI-Population CMA-ES on the
        BBOB-2009 Function Testbed. GECCO Workshop.
    """
    return CMAES(
        sigma=sigma,
        restarts=restarts,
        incpopsize=incpopsize,
        bipop=True,
        **kwargs,
    )