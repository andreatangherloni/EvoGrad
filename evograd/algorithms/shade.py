"""
SHADE (Success-History based Adaptive Differential Evolution) for EvoGrad.

This module provides SHADE and L-SHADE (with Linear Population Size Reduction),
which are self-adaptive DE variants that use historical memory of successful
F and CR parameters to guide adaptation.

Key Features:
    - Uses "current-to-pbest/1" mutation strategy
    - F values sampled from Cauchy distribution centered at memory values
    - CR values sampled from Normal distribution centered at memory values
    - Successful parameters are stored using weighted Lehmer mean
    - External archive stores replaced inferior solutions for diversity

Variants:
    - SHADE: Standard success-history based adaptation
    - L-SHADE: SHADE with Linear Population Size Reduction (LPSR)

Modes:
    - adaptive=False, differentiable=False: Classical SHADE
    - adaptive=True, differentiable=False: Learnable memory/operators via backprop
    - adaptive=False, differentiable=True: Learnable population via backprop
    - adaptive=True, differentiable=True: Full end-to-end differentiable

Note on adaptive vs differentiable:
    - `adaptive=True`: Enables backpropagation for OPERATORS (memory M_F, M_CR,
      selection temperature, crossover parameters become learnable)
    - `differentiable=True`: Enables backpropagation for POPULATION (the
      population tensor becomes an nn.Parameter, selection uses Gumbel-Softmax)

References:
    Tanabe, R. & Fukunaga, A. (2013). Success-History Based Parameter Adaptation
    for Differential Evolution. CEC 2013.
    
    Tanabe, R. & Fukunaga, A. (2014). Improving the Search Performance of SHADE
    Using Linear Population Size Reduction. CEC 2014.

Example:
    >>> from evograd.algorithms import SHADE, LSHADE
    >>> from evograd.core import Problem, minimize
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Standard SHADE
    >>> shade = SHADE(pop_size=100, memory_size=100)
    >>> result = minimize(problem, shade, max_evals=100000)
    >>> 
    >>> # L-SHADE with population reduction
    >>> lshade = LSHADE(pop_size_init=18*30, pop_size_min=4)
    >>> result = minimize(problem, lshade, max_evals=100000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from evograd.core.algorithm import Algorithm

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = [
    "SHADE",
    "LSHADE", 
    "SHADEMemory",
    "shade_default",
    "shade_adaptive",
    "lshade_default",
    "lshade_adaptive",
]


# =============================================================================
# SHADE Memory Container
# =============================================================================

@dataclass
class SHADEMemory:
    """
    Success-history memory for SHADE parameter adaptation.
    
    Stores historical successful F and CR values that guide the generation
    of new parameter values via Cauchy and Normal distributions.
    
    Attributes:
        M_F: Memory of successful mutation scale factors [H].
        M_CR: Memory of successful crossover rates [H].
        k: Current memory index for update (circular).
        H: Memory size.
        archive: External archive of replaced inferior solutions.
        archive_size: Current archive size.
        max_archive_size: Maximum archive size.
    """
    M_F: Tensor
    M_CR: Tensor
    k: int = 0
    H: int = 100
    archive: Optional[Tensor] = None
    archive_size: int = 0
    max_archive_size: int = 100
    
    @classmethod
    def create(
        cls,
        H: int = 100,
        max_archive_size: int = 100,
        init_F: float = 0.5,
        init_CR: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SHADEMemory":
        """
        Create initial SHADE memory.
        
        Args:
            H: Memory size.
            max_archive_size: Maximum external archive size.
            init_F: Initial F memory values.
            init_CR: Initial CR memory values.
            device: Computation device.
            dtype: Tensor dtype.
        
        Returns:
            Initialized SHADEMemory.
        """
        return cls(
            M_F=torch.full((H,), init_F, device=device, dtype=dtype),
            M_CR=torch.full((H,), init_CR, device=device, dtype=dtype),
            k=0,
            H=H,
            archive=None,
            archive_size=0,
            max_archive_size=max_archive_size,
        )
    
    def sample_F(self, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Sample F values from Cauchy distribution centered at random memory cells.
        
        F_i = cauchy(M_F[r_i], 0.1), truncated to (0, 1]
        
        Args:
            n: Number of samples.
            device: Computation device.
            dtype: Tensor dtype.
        
        Returns:
            Sampled F values [n].
        """
        # Select random memory indices
        r_idx = torch.randint(0, self.H, (n,), device=device)
        mu_F = self.M_F.to(device)[r_idx]
        
        # Sample from Cauchy distribution (using inverse CDF)
        # Cauchy(mu, gamma) = mu + gamma * tan(pi * (u - 0.5))
        u = torch.rand(n, device=device, dtype=dtype)
        gamma = 0.1  # Standard SHADE scale parameter
        F = mu_F + gamma * torch.tan(torch.pi * (u - 0.5))
        
        # Truncate to (0, 1]
        # Values > 1 are set to 1 in original SHADE
        F = torch.where(F > 1.0, torch.ones_like(F), F)
        
        # Values <= 0 are regenerated (we approximate by reflecting)
        F = torch.where(F <= 0, torch.abs(F) + 1e-4, F)
        F = torch.clamp(F, min=1e-4, max=1.0)
        
        return F
    
    def sample_CR(self, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Sample CR values from Normal distribution centered at random memory cells.
        
        CR_i = N(M_CR[r_i], 0.1), truncated to [0, 1]
        
        Args:
            n: Number of samples.
            device: Computation device.
            dtype: Tensor dtype.
        
        Returns:
            Sampled CR values [n].
        """
        # Select random memory indices
        r_idx = torch.randint(0, self.H, (n,), device=device)
        mu_CR = self.M_CR.to(device)[r_idx]
        
        # Sample from Normal distribution
        sigma = 0.1  # Standard SHADE scale parameter
        CR = mu_CR + sigma * torch.randn(n, device=device, dtype=dtype)
        
        # Truncate to [0, 1]
        CR = torch.clamp(CR, min=0.0, max=1.0)
        
        return CR
    
    def update(
        self,
        S_F: Tensor,
        S_CR: Tensor,
        weights: Tensor,
    ) -> None:
        """
        Update memory with successful F and CR values.
        
        Uses weighted Lehmer mean for F and weighted arithmetic mean for CR.
        
        Args:
            S_F: Successful F values [n_success].
            S_CR: Successful CR values [n_success].
            weights: Improvement weights [n_success].
        """
        if len(S_F) == 0:
            return
        
        # Normalise weights
        w = weights / (weights.sum() + 1e-10)
        
        # Weighted Lehmer mean for F (reduces bias towards small values)
        # mean_WL = sum(w * F^2) / sum(w * F)
        mean_F = (w * S_F * S_F).sum() / ((w * S_F).sum() + 1e-10)
        
        # Weighted arithmetic mean for CR
        mean_CR = (w * S_CR).sum()
        
        # Update memory at position k
        device = self.M_F.device
        self.M_F[self.k] = mean_F.to(device)
        self.M_CR[self.k] = mean_CR.to(device)
        
        # Circular increment
        self.k = (self.k + 1) % self.H
    
    def add_to_archive(self, solutions: Tensor) -> None:
        """
        Add replaced solutions to external archive.
        
        The archive maintains diversity by storing inferior solutions
        that were replaced during selection.
        
        Args:
            solutions: Solutions to add [n, n_var].
        """
        if solutions.numel() == 0:
            return
        
        if self.archive is None:
            self.archive = solutions.clone()
            self.archive_size = solutions.shape[0]
        else:
            # Concatenate new solutions
            self.archive = torch.cat([self.archive, solutions], dim=0)
            self.archive_size = self.archive.shape[0]
        
        # If archive exceeds max size, randomly remove excess
        if self.archive_size > self.max_archive_size:
            perm = torch.randperm(self.archive_size, device=self.archive.device)
            self.archive = self.archive[perm[:self.max_archive_size]]
            self.archive_size = self.max_archive_size


# =============================================================================
# SHADE Algorithm
# =============================================================================

class SHADE(Algorithm):
    """
    Success-History based Adaptive Differential Evolution (SHADE).
    
    SHADE adapts F and CR parameters using a success-history mechanism.
    F values are sampled from Cauchy distributions and CR values from
    Normal distributions, both centered at memory values updated with
    successful parameters.
    
    Args:
        pop_size: Population size.
        memory_size: Size of success-history memory (H). Default: 100.
        p_best_rate: Fraction of top individuals for pbest selection.
            Default: 0.1 (10%).
        archive_rate: Archive size as fraction of pop_size. Default: 1.0.
        init_F: Initial F memory values. Default: 0.5.
        init_CR: Initial CR memory values. Default: 0.5.
        sampling: Operator for initial population generation.
        repair: Repair operator for constraint handling.
        adaptive: If True, memory and operator parameters become learnable
            via backpropagation (learnable hyperparameters).
        differentiable: If True, population becomes an nn.Parameter and
            selection uses Gumbel-Softmax (learnable population).
        selection_temperature: Temperature for Gumbel-Softmax selection.
        seed: Random seed for reproducibility.
        device: Computation device.
        dtype: Tensor dtype.
    
    Attributes:
        memory: SHADEMemory containing M_F, M_CR, and archive.
        p_best_rate: Rate for pbest selection.
    
    Example:
        >>> # Standard SHADE
        >>> shade = SHADE(pop_size=100, memory_size=100)
        >>> 
        >>> # SHADE with larger archive
        >>> shade = SHADE(pop_size=100, archive_rate=2.0)
        >>> 
        >>> # Differentiable SHADE for meta-learning
        >>> shade = SHADE(pop_size=100, adaptive=True, differentiable=True)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        memory_size: int = 100,
        p_best_rate: float = 0.1,
        archive_rate: float = 1.0,
        init_F: float = 0.5,
        init_CR: float = 0.5,
        sampling: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        adaptive: bool = False,
        differentiable: bool = False,
        selection_temperature: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.archive_rate = archive_rate
        self._init_F = init_F
        self._init_CR = init_CR
        self.adaptive = adaptive
        self._selection_temperature = selection_temperature
        
        # Create pbest selection operator using TruncationSelection
        # Selects from top p_best_rate fraction of population
        pbest_selection = self._create_pbest_selection(p_best_rate, adaptive, selection_temperature)
        
        # Create random selection for r1 and r2
        random_selection = self._create_random_selection(adaptive, selection_temperature)
        
        # Create BinomialCrossover that supports per-individual CR via forward(cr=...)
        # - If adaptive=True: crossover is differentiable
        crossover = self._create_crossover(adaptive)
        
        # Call base class
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=random_selection,
            crossover=crossover,
            mutation=None,  # SHADE mutation is handled internally
            survival=None,  # SHADE uses greedy selection
            repair=repair,
            eliminate_duplicates=False,
            n_offsprings=pop_size,
            differentiable=differentiable,  # Controls whether population is learnable
            seed=seed,
            device=device,
            dtype=dtype,
        )
        
        # Register additional selection operators as submodules
        self._register_operator("pbest_selection", pbest_selection)
        self.pbest_selection = pbest_selection
    
    
    def _create_pbest_selection(
        self,
        p_best_rate: float,
        adaptive: bool,
        temperature: float,
    ) -> nn.Module:
        """
        Create pbest selection operator using TruncationSelection.
        
        Selects from top p_best_rate fraction of population.
        
        Args:
            p_best_rate: Fraction of top individuals to consider.
            adaptive: If True, temperature is learnable and use Gumbel-Softmax selection.
            temperature: Temperature for soft selection.
        
        Returns:
            TruncationSelection operator.
        """
        
        from evograd.operators.selection import TruncationSelection
        return TruncationSelection(
            truncation_ratio=p_best_rate,
            differentiable=adaptive,
            temperature=temperature,
            learn_temperature=adaptive,  # Only learn if adaptive
            minimize=True,
        )
    
    def _create_random_selection(
        self,
        adaptive: bool,
        temperature: float,
    ) -> nn.Module:
        """
        Create random selection operator.
        
        Args:
            adaptive: If True, use differentiable selection and use Gumbel-Softmax selection.
            temperature: Temperature for soft selection.
        
        Returns:
            RandomSelection operator.
        """
        
        from evograd.operators.selection import RandomSelection
        return RandomSelection(replacement=True,
                               differentiable=adaptive,
                               temperature=temperature,
                               )
    
    def _create_crossover(self, adaptive: bool) -> nn.Module:
        """
        Create binomial crossover operator.
        
        Uses BinomialCrossover from evograd.operators which supports
        per-individual CR via the `cr` parameter in forward().
        
        Args:
            adaptive: If True, crossover is differentiable.
        
        Returns:
            BinomialCrossover operator.
        """
        from evograd.operators.crossover import BinomialCrossover
        return BinomialCrossover(
            cr=0.5,  # Default CR, will be overridden per-individual
            differentiable=adaptive,  # Differentiable if adaptive
            learn_cr=False,  # CR comes from memory sampling, not learned directly
        )
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    def _setup(self) -> None:
        """SHADE-specific setup after initialization."""
        # Create success-history memory
        max_archive_size = int(self.archive_rate * self.pop_size)
        self.memory = SHADEMemory.create(
            H=self.memory_size,
            max_archive_size=max_archive_size,
            init_F=self._init_F,
            init_CR=self._init_CR,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Make memory learnable if adaptive mode
        if self.adaptive:
            self._M_F_param = nn.Parameter(self.memory.M_F.clone())
            self._M_CR_param = nn.Parameter(self.memory.M_CR.clone())
        
        # Store per-individual F and CR for current generation
        self._current_F: Optional[Tensor] = None
        self._current_CR: Optional[Tensor] = None
    
    @property
    def M_F(self) -> Tensor:
        """Current F memory."""
        if self.adaptive:
            return self._M_F_param
        return self.memory.M_F
    
    @property
    def M_CR(self) -> Tensor:
        """Current CR memory."""
        if self.adaptive:
            return self._M_CR_param
        return self.memory.M_CR
    
    @property
    def population(self) -> Tensor:
        """Current population."""
        return self._population
    
    @property
    def fitness(self) -> Tensor:
        """Current fitness values."""
        return self.state.fitness
    
    # =========================================================================
    # Core SHADE Methods
    # =========================================================================
    
    def _sample_parameters(self) -> Tuple[Tensor, Tensor]:
        """
        Sample F and CR values from memory distributions.
        
        If adaptive=True, uses reparameterization trick for gradient flow.
        
        Returns:
            Tuple of (F_values, CR_values), each [pop_size].
        """
        N = self.pop_size
        
        if self.adaptive:
            # Use learnable memory with reparameterization
            # Select random memory indices
            r_idx = torch.randint(0, self.memory_size, (N,), device=self.device)
            mu_F = self.M_F[r_idx]
            mu_CR = self.M_CR[r_idx]
            
            # Reparameterized Cauchy for F (using inverse CDF)
            u = torch.rand(N, device=self.device, dtype=self.dtype)
            gamma = 0.1
            F = mu_F + gamma * torch.tan(torch.pi * (u - 0.5))
            F = torch.clamp(F, min=1e-4, max=1.0)
            
            # Reparameterized Normal for CR
            sigma = 0.1
            eps = torch.randn(N, device=self.device, dtype=self.dtype)
            CR = mu_CR + sigma * eps
            CR = torch.clamp(CR, min=0.0, max=1.0)
        else:
            # Standard sampling from memory
            F = self.memory.sample_F(N, self.device, self.dtype)
            CR = self.memory.sample_CR(N, self.device, self.dtype)
        
        return F, CR
    
    def _select_pbest(self: float) -> Tensor:
        """
        Select random pbest individuals from top p% of population.
        
        Uses TruncationSelection operator which handles both hard and
        soft (Gumbel-Softmax) selection modes.
        
        Args:
            p_rate: Fraction of top individuals to consider.
        
        Returns:
            Selected pbest individuals [pop_size, n_var].
        """
        N = self.pop_size
        
        # Use TruncationSelection to select from top p_rate fraction
        # The operator handles differentiable vs hard selection internally
        pbest = self.pbest_selection(self.population, self.fitness, n_select=N)
        
        return pbest
    
    def _select_random_from_union(self) -> Tensor:
        """
        Select random individuals from population ∪ archive.
        
        Uses RandomSelection operator which handles both hard and
        soft (Gumbel-Softmax uniform) selection modes.
        
        Returns:
            Selected individuals [pop_size, n_var].
        """
        N = self.pop_size
        
        # Combine population and archive
        if self.memory.archive is not None and self.memory.archive_size > 0:
            union = torch.cat([self.population, self.memory.archive], dim=0)
        else:
            union = self.population
        
        # Create dummy fitness for random selection (RandomSelection ignores fitness)
        union_fitness = torch.zeros(union.shape[0], device=self.device, dtype=self.dtype)
        
        # Use RandomSelection operator for uniform selection from union
        selected = self.selection(union, union_fitness, n_select=N)
        
        return selected
    
    def _mutate(self) -> Tensor:
        """
        Generate donor vectors using current-to-pbest/1 mutation.
        
        v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
        
        Returns:
            Donor vectors [pop_size, n_var].
        """
        N = self.pop_size
        
        # Sample F and CR for this generation
        self._current_F, self._current_CR = self._sample_parameters()
        
        # Select pbest (random from top p%)
        x_pbest = self._select_pbest()
        
        # Select r1 from population (random, different from current)
        x_r1 = self.selection(self.population, self.fitness, n_select=N)
        
        # Select r2 from population ∪ archive
        x_r2 = self._select_random_from_union()
        
        # Ensure F has correct shape for broadcasting [N, 1]
        F = self._current_F.unsqueeze(-1)
        
        # current-to-pbest/1 mutation
        # v_i = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        donor = self.population + F * (x_pbest - self.population) + F * (x_r1 - x_r2)
        
        return donor
    
    def _infill(self) -> Tensor:
        """
        Generate trial vectors through mutation and crossover.
        
        Returns:
            Trial vectors [pop_size, n_var].
        """
        # 1. Mutation: create donor vectors
        donor = self._mutate()
        
        # 2. Crossover: binomial with per-individual CR
        # Use our BinomialCrossover with per-individual CR override
        trial = self.crossover(self.population, donor, cr=self._current_CR)
        
        # 3. Repair bounds
        if self.repair is not None:
            trial = self.repair(trial, self.xl, self.xu)
        else:
            trial = torch.clamp(trial, self.xl, self.xu)
        
        return trial
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Apply greedy selection and update memory.
        
        Args:
            offspring: Trial vectors [pop_size, n_var].
            offspring_fitness: Fitness of trials [pop_size].
        """
        # Identify successful trials (trial better than target)
        improved = offspring_fitness < self.fitness
        
        # Collect successful F and CR values
        if improved.any():
            S_F = self._current_F[improved]
            S_CR = self._current_CR[improved]
            
            # Weights based on fitness improvement (delta f)
            delta_f = self.fitness[improved] - offspring_fitness[improved]
            weights = delta_f
            
            # Update memory with successful parameters
            # In adaptive mode, gradients update memory directly, so skip
            if not self.adaptive:
                self.memory.update(S_F, S_CR, weights)
            
            # Add replaced solutions to archive
            replaced_solutions = self.population[improved].detach()
            self.memory.add_to_archive(replaced_solutions)
        
        # Greedy selection: keep trial if better, else keep target
        new_pop = torch.where(
            improved.unsqueeze(-1),
            offspring,
            self.population
        )
        new_fitness = torch.where(improved, offspring_fitness, self.fitness)
        
        # Update internal state
        self._update_population(new_pop, new_fitness)
        
        # Update best solution tracking
        self.state.update_best(self.population, self.state.fitness)
    
    def _update_population(self, new_pop: Tensor, new_fitness: Tensor) -> None:
        """
        Update population and fitness tensors.
        
        Args:
            new_pop: New population tensor [pop_size, n_var].
            new_fitness: New fitness tensor [pop_size].
        """
        with torch.no_grad():
            self._population.copy_(new_pop)
        self.state.fitness = new_fitness
        self.state.population = self._population
    
    # =========================================================================
    # Hyperparameter Access
    # =========================================================================
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameter values."""
        params = {
            'pop_size': self.pop_size,
            'memory_size': self.memory_size,
            'p_best_rate': self.p_best_rate,
            'archive_rate': self.archive_rate,
            'adaptive': self.adaptive,
            'differentiable': self.differentiable,
            'M_F_mean': float(self.M_F.mean().detach()),
            'M_CR_mean': float(self.M_CR.mean().detach()),
            'archive_size': self.memory.archive_size,
        }
        return params
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    @torch.no_grad()
    def _clamp_hyperparams(self) -> None:
        """Clamp learnable hyperparameters to valid ranges."""
        if self.adaptive:
            # F memory in (0, 1]
            self._M_F_param.clamp_(min=1e-4, max=1.0)
            # CR memory in [0, 1]
            self._M_CR_param.clamp_(min=0.0, max=1.0)
    
    def update_state(self) -> None:
        """Commit pending changes and clamp hyperparameters."""
        super().update_state()
        self._clamp_hyperparams()
        
        # Sync learnable memory back to SHADEMemory structure
        if self.adaptive:
            self.memory.M_F = self._M_F_param.detach().clone()
            self.memory.M_CR = self._M_CR_param.detach().clone()
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"SHADE(pop_size={self.pop_size}, "
            f"memory_size={self.memory_size}, "
            f"p_best_rate={self.p_best_rate:.2f}, "
            f"adaptive={self.adaptive}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# L-SHADE Algorithm (SHADE with Linear Population Size Reduction)
# =============================================================================

class LSHADE(SHADE):
    """
    L-SHADE: SHADE with Linear Population Size Reduction (LPSR).
    
    L-SHADE extends SHADE by linearly reducing the population size during
    optimisation. This allows early exploration with a large population
    and later exploitation with a focused small population.
    
    Population size at generation g:
        N_g = round((N_min - N_init) / max_evals * n_evals + N_init)
    
    Args:
        pop_size_init: Initial population size. Default: 18 * n_var.
        pop_size_min: Minimum population size. Default: 4.
        memory_size: Size of success-history memory (H). Default: 100.
        p_best_rate: Fraction of top individuals for pbest selection.
        archive_rate: Archive size as fraction of pop_size.
        init_F: Initial F memory values.
        init_CR: Initial CR memory values.
        sampling: Operator for initial population generation.
        repair: Repair operator for constraint handling.
        adaptive: If True, memory becomes learnable.
        differentiable: If True, population becomes learnable.
        selection_temperature: Temperature for differentiable selection.
        seed: Random seed.
        device: Computation device.
        dtype: Tensor dtype.
    
    Attributes:
        pop_size_init: Initial population size.
        pop_size_min: Minimum population size.
        max_evals: Maximum evaluations (set via set_max_evals).
    
    Example:
        >>> # Standard L-SHADE for 30D problem
        >>> lshade = LSHADE(pop_size_init=18*30, pop_size_min=4)
        >>> lshade.set_max_evals(100000)
        >>> result = minimize(problem, lshade, max_evals=100000)
    """
    
    def __init__(
        self,
        pop_size_init: Optional[int] = None,
        pop_size_min: int = 4,
        memory_size: int = 100,
        p_best_rate: float = 0.1,
        archive_rate: float = 2.6,  # L-SHADE default
        init_F: float = 0.5,
        init_CR: float = 0.5,
        sampling: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        adaptive: bool = False,
        differentiable: bool = False,
        selection_temperature: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.pop_size_init = pop_size_init  # Will be set in _setup if None
        self.pop_size_min = pop_size_min
        self._max_evals: Optional[int] = None
        
        # Use pop_size_init as initial pop_size, or default of 100
        init_pop_size = pop_size_init if pop_size_init is not None else 100
        
        super().__init__(
            pop_size=init_pop_size,
            memory_size=memory_size,
            p_best_rate=p_best_rate,
            archive_rate=archive_rate,
            init_F=init_F,
            init_CR=init_CR,
            sampling=sampling,
            repair=repair,
            adaptive=adaptive,
            differentiable=differentiable,
            selection_temperature=selection_temperature,
            seed=seed,
            device=device,
            dtype=dtype,
        )
    
    def _setup(self) -> None:
        """L-SHADE-specific setup after initialization."""
        # Set default pop_size_init based on problem dimension
        if self.pop_size_init is None:
            self.pop_size_init = 18 * self.problem.n_var
            self._pop_size = self.pop_size_init
        
        # Create success-history memory with L-SHADE archive size
        self.memory = SHADEMemory.create(
            H=self.memory_size,
            max_archive_size=int(self.archive_rate * self.pop_size_init),
            init_F=self._init_F,
            init_CR=self._init_CR,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Make memory learnable if adaptive mode
        if self.adaptive:
            self._M_F_param = nn.Parameter(self.memory.M_F.clone())
            self._M_CR_param = nn.Parameter(self.memory.M_CR.clone())
        
        # Store per-individual F and CR
        self._current_F = None
        self._current_CR = None
    
    def set_max_evals(self, max_evals: int) -> None:
        """
        Set maximum evaluations for population size reduction.
        
        Must be called before running the algorithm.
        
        Args:
            max_evals: Maximum fitness evaluations.
        """
        self._max_evals = max_evals
    
    @property
    def target_pop_size(self) -> int:
        """
        Calculate target population size based on current evaluations.
        
        Returns:
            Target population size for current generation.
        """
        if self._max_evals is None:
            return self._pop_size
        
        n_evals = self.n_evals
        N_init = self.pop_size_init
        N_min = self.pop_size_min
        
        # Linear reduction formula
        N_g = round((N_min - N_init) / self._max_evals * n_evals + N_init)
        N_g = max(N_g, N_min)
        
        return N_g
    
    def _reduce_population(self) -> None:
        """
        Reduce population size according to LPSR schedule.
        
        Removes worst individuals to reach target population size.
        """
        target_size = self.target_pop_size
        current_size = self._pop_size
        
        if target_size >= current_size:
            return
        
        n_remove = current_size - target_size
        
        # Get indices of best individuals to keep
        keep_idx = torch.argsort(self.fitness)[:target_size]
        
        # Keep only best individuals
        with torch.no_grad():
            new_pop = self.population[keep_idx].clone()
            new_fitness = self.fitness[keep_idx].clone()
            
            # Resize population tensor
            if self.differentiable:
                self._population = nn.Parameter(new_pop)
            else:
                # Re-register buffer with new size
                delattr(self, '_population')
                self.register_buffer('_population', new_pop)
            
            self.state.fitness = new_fitness
            self.state.population = self._population
            self._pop_size = target_size
        
        # Also reduce archive if needed
        self.memory.max_archive_size = int(self.archive_rate * target_size)
        if self.memory.archive_size > self.memory.max_archive_size:
            perm = torch.randperm(self.memory.archive_size, device=self.memory.archive.device)
            self.memory.archive = self.memory.archive[perm[:self.memory.max_archive_size]]
            self.memory.archive_size = self.memory.max_archive_size
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Apply greedy selection, update memory, and reduce population.
        
        Args:
            offspring: Trial vectors.
            offspring_fitness: Fitness of trials.
        """
        # Standard SHADE advance
        super()._advance(offspring, offspring_fitness)
        
        # Apply population size reduction
        self._reduce_population()
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameter values."""
        params = super()._get_hyperparams()
        params.update({
            'pop_size_init': self.pop_size_init,
            'pop_size_min': self.pop_size_min,
            'target_pop_size': self.target_pop_size,
            'max_evals': self._max_evals,
        })
        return params
    
    def __repr__(self) -> str:
        return (
            f"LSHADE(pop_size={self.pop_size}, "
            f"pop_size_init={self.pop_size_init}, "
            f"pop_size_min={self.pop_size_min}, "
            f"memory_size={self.memory_size}, "
            f"adaptive={self.adaptive}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def shade_default(
    pop_size: int = 100,
    memory_size: int = 100,
    p_best_rate: float = 0.1,
    **kwargs,
) -> SHADE:
    """
    Create standard SHADE with default parameters.
    
    Args:
        pop_size: Population size.
        memory_size: Size of success-history memory.
        p_best_rate: Fraction of top individuals for pbest.
        **kwargs: Additional arguments passed to SHADE.
    
    Returns:
        Configured SHADE instance.
    """
    return SHADE(
        pop_size=pop_size,
        memory_size=memory_size,
        p_best_rate=p_best_rate,
        **kwargs,
    )


def shade_adaptive(
    pop_size: int = 100,
    memory_size: int = 100,
    adaptive: bool = True,
    differentiable: bool = True,
    **kwargs,
) -> SHADE:
    """
    Create SHADE with learnable memory and differentiable population.
    
    Args:
        pop_size: Population size.
        memory_size: Size of success-history memory.
        adaptive: If True, memory is learnable.
        differentiable: If True, population is learnable.
        **kwargs: Additional arguments passed to SHADE.
    
    Returns:
        Configured SHADE instance.
    """
    return SHADE(
        pop_size=pop_size,
        memory_size=memory_size,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def lshade_default(
    pop_size_init: Optional[int] = None,
    pop_size_min: int = 4,
    memory_size: int = 100,
    **kwargs,
) -> LSHADE:
    """
    Create standard L-SHADE with default parameters.
    
    If pop_size_init is None, it defaults to 18 * n_var during setup.
    
    Args:
        pop_size_init: Initial population size (None = 18*n_var).
        pop_size_min: Minimum population size.
        memory_size: Size of success-history memory.
        **kwargs: Additional arguments passed to LSHADE.
    
    Returns:
        Configured LSHADE instance.
    """
    return LSHADE(
        pop_size_init=pop_size_init,
        pop_size_min=pop_size_min,
        memory_size=memory_size,
        **kwargs,
    )


def lshade_adaptive(
    pop_size_init: Optional[int] = None,
    pop_size_min: int = 4,
    adaptive: bool = True,
    differentiable: bool = True,
    **kwargs,
) -> LSHADE:
    """
    Create L-SHADE with learnable memory and differentiable population.
    
    Args:
        pop_size_init: Initial population size (None = 18*n_var).
        pop_size_min: Minimum population size.
        adaptive: If True, memory is learnable.
        differentiable: If True, population is learnable.
        **kwargs: Additional arguments passed to LSHADE.
    
    Returns:
        Configured LSHADE instance.
    """
    return LSHADE(
        pop_size_init=pop_size_init,
        pop_size_min=pop_size_min,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )