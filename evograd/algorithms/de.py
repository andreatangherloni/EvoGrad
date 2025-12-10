"""
Differential Evolution (DE) implementation for EvoGrad.

This module provides a fully differentiable Differential Evolution algorithm
that supports both classical and gradient-enabled optimisation modes.

DE evolves a population through:
    1. Mutation: Create donor vectors using difference of population members
    2. Crossover: Combine target and donor to create trial vectors
    3. Selection: Greedy one-to-one replacement

All operators are pluggable via dependency injection (pymoo-style). The
crossover operator uses the existing BinomialCrossover or ExponentialCrossover
from the operators module.

Variants:
    The variant string (e.g., "DE/rand/1/bin") specifies:
    - Mutation base: rand, best, current-to-best, current-to-rand
    - Number of difference vectors: 1 or 2
    - Crossover type: bin (binomial) or exp (exponential)

Modes:
    - adaptive=False, differentiable=False: Classical DE
    - adaptive=True, differentiable=False: Operators are differentiable,
        hyperparameters (F, CR, temperatures) learned via backprop
    - adaptive=False, differentiable=True: Population is differentiable,
        learned via backprop
    - adaptive=True, differentiable=True: Both operators and population
        are differentiable

Example:
    >>> from evograd.algorithms import DE
    >>> from evograd.core import Problem, minimize
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Classical DE
    >>> de = DE(pop_size=100, variant="DE/rand/1/bin", F=0.5, CR=0.9)
    >>> result = minimize(problem, de, max_evals=10000)
    >>> 
    >>> # Adaptive DE with learnable hyperparameters
    >>> de = DE(pop_size=100, variant="DE/best/1/bin", adaptive=True)
    >>> result = minimize(problem, de, max_evals=10000)

Reference:
    Storn, R. & Price, K. (1997). Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from evograd.core.algorithm import Algorithm

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = ["DE", "DEVariant"]


# =============================================================================
# DE Variant Parser
# =============================================================================

@dataclass
class DEVariant:
    """
    Parsed DE variant specification.
    
    Attributes:
        mutation: Mutation strategy (rand, best, current-to-best, current-to-rand)
        n_diff: Number of difference vectors (1 or 2)
        crossover: Crossover type (bin, exp, or None for current-to-rand)
    """
    mutation: str
    n_diff: int
    crossover: Optional[str]
    
    # Pattern: DE/mutation/n_diff/crossover
    _PATTERN = re.compile(
        r"^DE/(RAND|BEST|CURRENT-TO-BEST|CURRENT-TO-RAND)/([12])(?:/(BIN|EXP))?$",
        re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, variant: str) -> "DEVariant":
        """
        Parse a DE variant string.
        
        Args:
            variant: Variant string like "DE/rand/1/bin"
        
        Returns:
            Parsed DEVariant instance.
        
        Raises:
            ValueError: If variant string is invalid.
        """
        # Normalise: replace underscores with hyphens
        normalised = variant.replace("_", "-").upper()
        
        match = cls._PATTERN.match(normalised)
        if not match:
            valid = [
                "DE/rand/1/bin", "DE/rand/1/exp", "DE/rand/2/bin", "DE/rand/2/exp",
                "DE/best/1/bin", "DE/best/1/exp", "DE/best/2/bin", "DE/best/2/exp",
                "DE/current-to-best/1/bin", "DE/current-to-best/1/exp",
                "DE/current-to-rand/1"
            ]
            raise ValueError(
                f"Invalid DE variant '{variant}'. "
                f"Valid variants: {', '.join(valid)}"
            )
        
        mutation = match.group(1).lower().replace("-", "_")
        n_diff = int(match.group(2))
        crossover = match.group(3).lower() if match.group(3) else None
        
        # current-to-rand doesn't use crossover
        if mutation == "current_to_rand" and crossover is not None:
            raise ValueError(
                f"DE/current-to-rand does not use crossover. "
                f"Use 'DE/current-to-rand/1' without crossover suffix."
            )
        
        # Other variants require crossover
        if mutation != "current_to_rand" and crossover is None:
            raise ValueError(
                f"Variant '{variant}' requires crossover type. "
                f"Use 'DE/{mutation}/{n_diff}/bin' or 'DE/{mutation}/{n_diff}/exp'."
            )
        
        return cls(mutation=mutation, n_diff=n_diff, crossover=crossover)
    
    def __str__(self) -> str:
        mutation_str = self.mutation.replace("_", "-")
        if self.crossover:
            return f"DE/{mutation_str}/{self.n_diff}/{self.crossover}"
        return f"DE/{mutation_str}/{self.n_diff}"


# =============================================================================
# Differential Evolution Algorithm
# =============================================================================

class DE(Algorithm):
    """
    Differential Evolution (DE) for continuous optimisation.
    
    DE evolves a population through mutation (using difference vectors),
    crossover, and greedy selection. Supports multiple mutation strategies
    and both binomial and exponential crossover.
    
    Args:
        pop_size: Population size.
        variant: DE variant string (e.g., "DE/rand/1/bin").
            See DEVariant for valid options.
        F: Mutation scale factor in (0, 2]. Default: 0.5.
        CR: Crossover rate in [0, 1]. Default: 0.9.
        sampling: Operator for initial population generation.
        crossover: Crossover operator. If None, created from variant.
        repair: Repair operator for constraint handling.
        dither: F randomisation strategy (classical mode only):
            - None: Fixed F
            - "scalar": Randomise F once per generation
            - "vector": Randomise F per individual
        jitter: If True, add small per-dimension noise to F (classical only).
        adaptive: If True, operators are differentiable and hyperparameters
            (F, CR, temperatures) are learned via backpropagation.
        differentiable: If True, population is differentiable and
            learned via backpropagation.
        selection_temperature: Initial temperature for Gumbel-Softmax selection.
        seed: Random seed for reproducibility.
        device: Computation device.
        dtype: Tensor dtype.
    
    Attributes:
        variant: Parsed DEVariant.
        F: Current mutation scale factor.
        CR: Current crossover rate.
    
    Example:
        >>> # Classical DE/rand/1/bin
        >>> de = DE(pop_size=100, variant="DE/rand/1/bin")
        >>> 
        >>> # Adaptive DE with learnable hyperparameters
        >>> de = DE(variant="DE/best/1/bin", adaptive=True)
        >>> 
        >>> # Differentiable population
        >>> de = DE(variant="DE/rand/1/bin", differentiable=True)
        >>> 
        >>> # Both adaptive and differentiable
        >>> de = DE(variant="DE/current-to-best/1/bin", adaptive=True, differentiable=True)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        variant: str = "DE/rand/1/bin",
        F: float = 0.5,
        CR: float = 0.9,
        sampling: Optional[nn.Module] = None,
        crossover: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        dither: Optional[str] = None,
        jitter: bool = False,
        adaptive: bool = False,
        differentiable: bool = False,
        selection_temperature: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Parse variant
        self.variant = DEVariant.parse(variant)
        self.dither = dither
        self.jitter = jitter
        self.adaptive = adaptive
        self._init_F = F
        self._init_CR = CR
        self._selection_temperature = selection_temperature
        
        # Create crossover operator if not provided
        if crossover is None and self.variant.crossover is not None:
            crossover = self._create_crossover(CR, adaptive)
        
        # Create selection operator for parent selection in mutation
        # Selection is differentiable when adaptive=True
        selection = self._create_selection(adaptive, selection_temperature)
        
        # Call base class
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=None,  # DE mutation is handled internally
            survival=None,  # DE uses greedy one-to-one selection
            repair=repair,
            eliminate_duplicates=False,  # DE doesn't eliminate duplicates
            n_offsprings=pop_size,  # DE creates one trial per individual
            differentiable=differentiable,
            seed=seed,
            device=device,
            dtype=dtype,
        )
    
    def _create_crossover(
        self,
        CR: float,
        adaptive: bool,
    ) -> nn.Module:
        """
        Create the appropriate crossover operator.
        
        When adaptive=True, crossover is differentiable with learnable CR.
        """
        if self.variant.crossover == "bin":
            from evograd.operators.crossover import BinomialCrossover
            return BinomialCrossover(
                cr=CR,
                differentiable=adaptive,  # Differentiable when adaptive
                learn_cr=adaptive,  # Learn CR when adaptive
            )
        elif self.variant.crossover == "exp":
            from evograd.operators.crossover import ExponentialCrossover
            return ExponentialCrossover(
                cr=CR,
                differentiable=adaptive,  # Differentiable when adaptive
                learn_cr=adaptive,  # Learn CR when adaptive
            )
        return None
    
    def _create_selection(
        self,
        adaptive: bool,
        temperature: float,
    ) -> nn.Module:
        """
        Create selection operator for parent selection in mutation.
        
        When adaptive=True, selection is differentiable with learnable temperature.
        """
        if adaptive:
            # Use fitness-proportionate selection with Gumbel-Softmax
            from evograd.operators.selection import RouletteSelection
            return RouletteSelection(
                differentiable=True,
                temperature=temperature,
                learn_temperature=True,  # Learn temperature when adaptive
                minimize=True,
            )
        else:
            # Use random selection in classical mode
            from evograd.operators.selection import RandomSelection
            return RandomSelection(replacement=True)
    
    # =========================================================================
    # Setup and Hyperparameters
    # =========================================================================
    
    def _setup(self) -> None:
        """DE-specific setup after initialization."""
        n_var = self.problem.n_var
        
        # Setup F parameter
        if self.adaptive:
            # Learnable F stored as log(F) for positivity
            self._log_F = nn.Parameter(
                torch.tensor(self._init_F, device=self.device, dtype=self.dtype).log()
            )
        else:
            self.register_buffer(
                "_F_buffer",
                torch.tensor(self._init_F, device=self.device, dtype=self.dtype)
            )
    
    @property
    def F(self) -> Tensor:
        """Current mutation scale factor."""
        if self.adaptive:
            return self._log_F.exp()
        return self._F_buffer
    
    @property
    def CR(self) -> Tensor:
        """Current crossover rate."""
        if self.crossover is not None and hasattr(self.crossover, 'cr'):
            return self.crossover.cr
        return torch.tensor(self._init_CR, device=self.device)
    
    # =========================================================================
    # Core DE Methods
    # =========================================================================
    
    def _get_F_values(self, n: int) -> Tensor:
        """
        Get F values, optionally with dither/jitter (classical mode only).
        
        Args:
            n: Number of F values needed.
        
        Returns:
            F values tensor of shape [n] or [n, n_var].
        """
        base_F = self.F
        
        # Dither and jitter only in classical mode (not adaptive)
        if not self.adaptive:
            if self.dither == "scalar":
                # Same random F for all individuals this generation
                F_val = base_F + 0.1 * (2 * torch.rand(1, device=self.device) - 1)
                F_val = F_val.expand(n)
            elif self.dither == "vector":
                # Different random F for each individual
                F_val = 0.5 + 0.5 * torch.rand(n, device=self.device)
            else:
                F_val = base_F.expand(n)
            
            if self.jitter:
                # Add small per-dimension noise
                n_var = self.n_var
                jitter_noise = 0.001 * (2 * torch.rand(n, n_var, device=self.device) - 1)
                F_val = F_val.unsqueeze(-1) + jitter_noise
        else:
            # Adaptive mode: use learnable F directly
            F_val = base_F.expand(n)
        
        return F_val
    
    def _select_parents(
        self,
        n_select: int,
    ) -> Tensor:
        """
        Select parents for mutation using the selection operator.
        
        Args:
            n_select: Number of parents to select.
        
        Returns:
            Selected individuals [n_select, n_var].
        """
        return self.selection(
            self.population,
            self.fitness,
            n_select=n_select,
        )
    
    def _mutate(self) -> Tensor:
        """
        Generate donor vectors using the mutation strategy.
        
        Returns:
            Donor vectors [pop_size, n_var].
        """
        N = self.pop_size
        F = self._get_F_values(N)
        
        # Ensure F has correct shape for broadcasting
        if F.dim() == 1:
            F = F.unsqueeze(-1)  # [N, 1] for broadcasting
        
        mutation_type = self.variant.mutation
        
        if mutation_type == "rand":
            # DE/rand: v = x_r1 + F * (x_r2 - x_r3)
            r1 = self._select_parents(N)
            r2 = self._select_parents(N)
            r3 = self._select_parents(N)
            
            if self.variant.n_diff == 1:
                donor = r1 + F * (r2 - r3)
            else:  # n_diff == 2
                r4 = self._select_parents(N)
                r5 = self._select_parents(N)
                donor = r1 + F * (r2 - r3) + F * (r4 - r5)
        
        elif mutation_type == "best":
            # DE/best: v = x_best + F * (x_r1 - x_r2)
            best_idx = torch.argmin(self.fitness)
            x_best = self.population[best_idx].unsqueeze(0).expand(N, -1)
            
            r1 = self._select_parents(N)
            r2 = self._select_parents(N)
            
            if self.variant.n_diff == 1:
                donor = x_best + F * (r1 - r2)
            else:  # n_diff == 2
                r3 = self._select_parents(N)
                r4 = self._select_parents(N)
                donor = x_best + F * (r1 - r2) + F * (r3 - r4)
        
        elif mutation_type == "current_to_best":
            # DE/current-to-best: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            best_idx = torch.argmin(self.fitness)
            x_best = self.population[best_idx].unsqueeze(0).expand(N, -1)
            
            r1 = self._select_parents(N)
            r2 = self._select_parents(N)
            
            donor = self.population + F * (x_best - self.population) + F * (r1 - r2)
        
        elif mutation_type == "current_to_rand":
            # DE/current-to-rand: v = x_i + K * (x_r1 - x_i) + F * (x_r2 - x_r3)
            # K is typically random in [0, 1]
            K = torch.rand(N, 1, device=self.device, dtype=self.dtype)
            
            r1 = self._select_parents(N)
            r2 = self._select_parents(N)
            r3 = self._select_parents(N)
            
            donor = self.population + K * (r1 - self.population) + F * (r2 - r3)
        
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
        
        return donor
    
    def _infill(self) -> Tensor:
        """
        Generate trial vectors through mutation and crossover.
        
        Returns:
            Trial vectors [pop_size, n_var].
        """
        # 1. Mutation: create donor vectors
        donor = self._mutate()
        
        # 2. Crossover: combine target (population) and donor
        if self.crossover is not None:
            trial = self.crossover(self.population, donor)
        else:
            # current-to-rand: no crossover, donor is the trial
            trial = donor
        
        # 3. Repair bounds
        if self.repair is not None:
            trial = self.repair(trial, self.xl, self.xu)
        else:
            # Default: clamp to bounds
            trial = torch.clamp(trial, self.xl, self.xu)
        
        return trial
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Apply greedy one-to-one selection.
        
        Each trial vector replaces the corresponding target if it has
        better (lower for minimisation) fitness.
        
        Args:
            offspring: Trial vectors [pop_size, n_var].
            offspring_fitness: Fitness of trial vectors [pop_size].
        """
        # Greedy selection: trial replaces target if better
        improved = offspring_fitness < self.fitness
        
        # Update population
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
        """Update population and fitness tensors."""
        with torch.no_grad():
            self._population.copy_(new_pop)
        self.state.fitness = new_fitness
        self.state.population = self._population
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def population(self) -> Tensor:
        """Current population."""
        return self._population
    
    @property
    def fitness(self) -> Tensor:
        """Current fitness values."""
        return self.state.fitness
    
    # =========================================================================
    # Hyperparameter Access
    # =========================================================================
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameter values."""
        params = {
            'pop_size': self.pop_size,
            'variant': str(self.variant),
            'F': float(self.F),
            'adaptive': self.adaptive,
            'differentiable': self.differentiable,
        }
        
        # Add CR from crossover operator
        if self.crossover is not None and hasattr(self.crossover, 'cr'):
            cr = self.crossover.cr
            if isinstance(cr, Tensor):
                params['CR'] = float(cr.mean())
            else:
                params['CR'] = float(cr)
        
        # Add selection temperature
        if hasattr(self.selection, 'temperature'):
            params['selection_temperature'] = float(self.selection.temperature)
        
        # Add crossover temperature
        if self.crossover is not None and hasattr(self.crossover, 'temperature'):
            params['crossover_temperature'] = float(self.crossover.temperature)
        
        return params
    
    # =========================================================================
    # State Management for Adaptive Mode
    # =========================================================================
    
    @torch.no_grad()
    def _clamp_hyperparams(self) -> None:
        """Clamp learnable hyperparameters to valid ranges."""
        if self.adaptive:
            # F in (0.01, 2.0) -> log(F) in (log(0.01), log(2.0))
            self._log_F.clamp_(min=-4.6, max=0.7)
    
    def update_state(self) -> None:
        """Commit pending changes and clamp hyperparameters."""
        super().update_state()
        self._clamp_hyperparams()
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"DE(pop_size={self.pop_size}, "
            f"variant='{self.variant}', "
            f"F={float(self.F):.3f}, "
            f"adaptive={self.adaptive}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def de_rand_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> DE:
    """
    Create DE/rand/1/bin - the classic DE variant.
    
    Args:
        pop_size: Population size.
        F: Mutation scale factor.
        CR: Crossover rate.
        adaptive: If True, operators are differentiable with learnable hyperparams.
        differentiable: If True, population is learnable.
        **kwargs: Additional arguments passed to DE.
    
    Returns:
        Configured DE instance.
    """
    return DE(
        pop_size=pop_size,
        variant="DE/rand/1/bin",
        F=F,
        CR=CR,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def de_best_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> DE:
    """
    Create DE/best/1/bin - greedy variant using best individual.
    
    Args:
        pop_size: Population size.
        F: Mutation scale factor.
        CR: Crossover rate.
        adaptive: If True, operators are differentiable with learnable hyperparams.
        differentiable: If True, population is learnable.
        **kwargs: Additional arguments passed to DE.
    
    Returns:
        Configured DE instance.
    """
    return DE(
        pop_size=pop_size,
        variant="DE/best/1/bin",
        F=F,
        CR=CR,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def de_current_to_best_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> DE:
    """
    Create DE/current-to-best/1/bin - balances exploration and exploitation.
    
    Args:
        pop_size: Population size.
        F: Mutation scale factor.
        CR: Crossover rate.
        adaptive: If True, operators are differentiable with learnable hyperparams.
        differentiable: If True, population is learnable.
        **kwargs: Additional arguments passed to DE.
    
    Returns:
        Configured DE instance.
    """
    return DE(
        pop_size=pop_size,
        variant="DE/current-to-best/1/bin",
        F=F,
        CR=CR,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )