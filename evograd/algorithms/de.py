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
from evograd.operators.repair import clamp_to_bounds
from evograd.operators.relaxations import log_param

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = ["DE", "DEVariant", "de_default", "de_rand_1_bin", "de_best_1_bin", "de_current_to_best_1_bin"]

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
        per_individual_coeffs: If True, sample F and CR independently for
            each individual. In classical mode, sampled from Uniform(0.5, 1.0).
            In adaptive mode, sampled around the learned base values using
            reparameterization (gradients flow to learned parameters).
        adaptive: If True, operators are differentiable and hyperparameters
            (F, CR, temperatures) are learned via backpropagation.
        differentiable: If True, population is differentiable and
            learned via backpropagation.
        selection_temperature: Initial temperature for Gumbel-Softmax selection.
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
        >>>
        >>> # Per-individual F and CR (jDE-style)
        >>> de = DE(variant="DE/rand/1/bin", per_individual_coeffs=True)
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
        per_individual_coeffs: bool = False,
        adaptive: bool = False,
        differentiable: bool = False,
        selection_temperature: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Parse variant
        self.variant = DEVariant.parse(variant)
        self.dither = dither
        self.jitter = jitter
        self.per_individual_coeffs = per_individual_coeffs
        self.adaptive = adaptive
        self._init_F = F
        self._init_CR = CR
        self._selection_temperature = selection_temperature
        
        # Create crossover operator if not provided
        if crossover is None and self.variant.crossover is not None:
            crossover = self._create_crossover(CR, adaptive)
        
        # Create selection operator for parent selection in mutation
        # Selection is differentiable when adaptive=True
        selection = self._create_random_selection(adaptive, selection_temperature)
        
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
            adaptive=adaptive,
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
                adaptive=adaptive,  # Differentiable when adaptive
                learn_cr=adaptive,  # Learn CR when adaptive
            )
        elif self.variant.crossover == "exp":
            from evograd.operators.crossover import ExponentialCrossover
            return ExponentialCrossover(
                cr=CR,
                adaptive=adaptive,  # Differentiable when adaptive
                learn_cr=adaptive,  # Learn CR when adaptive
            )
        return None
    
    def _create_random_selection(self, adaptive: bool, temperature: float) -> nn.Module:
        """
        Create selection operator for parent selection in mutation.
        When adaptive=True, selection is differentiable with learnable temperature.
        """
        from evograd.operators.selection import RandomSelection
        return RandomSelection(replacement=True,
                               adaptive=adaptive,
                               temperature=temperature,
                               )
        
    # =========================================================================
    # Setup and Hyperparameters
    # =========================================================================
    
    def _setup(self) -> None:
        """DE-specific setup after initialization."""
        n_var = self.problem.n_var
        
        # Setup F parameter
        if self.adaptive:
            # Learnable F stored as log(F) for positivity
            self._log_F = log_param(self._init_F, device=self.device, dtype=self.dtype)
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
        Get F values, optionally with dither/jitter/per_individual.
        
        In adaptive mode, noise is added around the learned base_F using
        reparameterization so gradients flow to the learnable parameter.
        
        Args:
            n: Number of F values needed.
        
        Returns:
            F values tensor of shape [n] or [n, n_var].
        """
        base_F = self.F
        
        # Per-individual coefficients: sample F around base (or Uniform if classical)
        if self.per_individual_coeffs:
            if self.adaptive:
                # Reparameterized: noise around learned base_F (gradients flow)
                # F_i = base_F + 0.25 * (2u - 1), u ~ Uniform(0,1) -> F_i ~ Uniform(base_F-0.25, base_F+0.25)
                noise = 0.25 * (2 * torch.rand(n, device=self.device, dtype=self.dtype) - 1)
                F_val = base_F + noise
            else:
                # Classical: F ~ Uniform(0.5, 1.0)
                F_val = 0.5 + 0.5 * torch.rand(n, device=self.device, dtype=self.dtype)
            
            F_val = F_val.clamp(0.01, 2.0)
            
            if self.jitter:
                n_var = self.n_var
                jitter_noise = 0.001 * (2 * torch.rand(n, n_var, device=self.device, dtype=self.dtype) - 1)
                F_val = F_val.unsqueeze(-1) + jitter_noise
            
            return F_val
        
        # Dither: randomize F per-generation or per-individual
        if self.dither == "scalar":
            # Same random F for all individuals this generation
            if self.adaptive:
                # Noise around learned base_F
                noise = 0.1 * (2 * torch.rand(1, device=self.device, dtype=self.dtype) - 1)
                F_val = (base_F + noise).expand(n)
            else:
                F_val = base_F + 0.1 * (2 * torch.rand(1, device=self.device, dtype=self.dtype) - 1)
                F_val = F_val.expand(n)
        elif self.dither == "vector":
            # Different random F for each individual
            if self.adaptive:
                # Noise around learned base_F
                noise = 0.25 * (2 * torch.rand(n, device=self.device, dtype=self.dtype) - 1)
                F_val = base_F + noise
            else:
                F_val = 0.5 + 0.5 * torch.rand(n, device=self.device, dtype=self.dtype)
        else:
            # No dither: use base_F directly
            F_val = base_F.expand(n)
        
        F_val = F_val.clamp(0.01, 2.0)
        
        # Jitter: add small per-dimension noise
        if self.jitter:
            n_var = self.n_var
            jitter_noise = 0.001 * (2 * torch.rand(n, n_var, device=self.device, dtype=self.dtype) - 1)
            if F_val.dim() == 1:
                F_val = F_val.unsqueeze(-1) + jitter_noise
            else:
                F_val = F_val + jitter_noise
        
        return F_val
    
    def _get_CR_values(self, n: int) -> Optional[Tensor]:
        """
        Get CR values for per-individual crossover.
        
        In adaptive mode, noise is added around the learned CR using
        reparameterization so gradients flow to the learnable parameter.
        
        Args:
            n: Number of CR values needed.
        
        Returns:
            CR values tensor of shape [n], or None if not using per_individual_coeffs.
        """
        if not self.per_individual_coeffs:
            return None
        
        if self.adaptive:
            # Reparameterized: noise around learned CR (gradients flow)
            # CR_i = base_CR + 0.25 * (2u - 1), u ~ Uniform(0,1)
            base_CR = self.CR
            noise = 0.25 * (2 * torch.rand(n, device=self.device, dtype=self.dtype) - 1)
            CR_val = base_CR + noise
        else:
            # Classical: CR ~ Uniform(0.5, 1.0)
            CR_val = 0.5 + 0.5 * torch.rand(n, device=self.device, dtype=self.dtype)
        
        return CR_val.clamp(0.0, 1.0)
    
    def _select_parents(
        self,
        n_select: int,
    ) -> Tensor:
        """
        Select parents for mutation using the selection operator.

        In adaptive (differentiable) mode, uses the soft Gumbel-Softmax
        selection operator so that gradients flow through parent selection.

        Args:
            n_select: Number of parents to select.

        Returns:
            Selected individuals [n_select, n_var].
        """
        return self.selection(self.population, self.fitness, n_select=n_select)

    @staticmethod
    def _sample_distinct_indices(
        N: int,
        n_needed: int,
        device: torch.device,
        exclude: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Sample ``n_needed`` mutually exclusive random index vectors of
        length ``N``, each different from the optional ``exclude`` indices.

        This is the canonical DE requirement: for each target *i* the
        selected donor indices r1, r2, … must be distinct from each other
        and from *i*.

        Args:
            N: Population size.
            n_needed: How many distinct index vectors to draw (e.g. 3
                for DE/rand/1).
            device: Target device.
            exclude: Optional ``[N]`` tensor of indices to avoid
                (typically ``torch.arange(N)`` for the target vector).

        Returns:
            List of ``n_needed`` tensors, each of shape ``[N]``.
        """
        # Build a pool of candidate indices for each target
        # For each row i we need n_needed indices from {0..N-1} \ {exclude[i]}
        all_indices: List[Tensor] = []
        for _ in range(n_needed):
            idx = torch.randint(0, N, (N,), device=device)
            all_indices.append(idx)

        # Rejection-resample collisions (vectorised, one pass per pair)
        targets = exclude if exclude is not None else torch.full((N,), -1, device=device)

        for k in range(len(all_indices)):
            # Avoid target index
            collides = all_indices[k] == targets
            while collides.any():
                all_indices[k][collides] = torch.randint(0, N, (int(collides.sum()),), device=device)
                collides = all_indices[k] == targets

            # Avoid previously selected indices
            for j in range(k):
                collides = all_indices[k] == all_indices[j]
                while collides.any():
                    all_indices[k][collides] = torch.randint(0, N, (int(collides.sum()),), device=device)
                    # Re-check all constraints for the resampled positions
                    collides = all_indices[k] == targets
                    for jj in range(k):
                        collides = collides | (all_indices[k] == all_indices[jj])

        return all_indices

    def _mutate(self) -> Tensor:
        """
        Generate donor vectors using the mutation strategy.

        In classical mode, parent indices are sampled to be mutually
        exclusive and different from the target index (canonical DE).
        In adaptive mode, the soft selection operator is used instead
        so that gradients can flow through the selection process; the
        exclusion constraint is relaxed in that case.

        Returns:
            Donor vectors [pop_size, n_var].
        """
        N = self.pop_size
        F = self._get_F_values(N)

        # Ensure F has correct shape for broadcasting
        if F.dim() == 1:
            F = F.unsqueeze(-1)  # [N, 1] for broadcasting

        mutation_type = self.variant.mutation

        # -----------------------------------------------------------------
        # Helper: pick parents (hard distinct indices or soft selection)
        # -----------------------------------------------------------------
        def _hard_parents(n_needed: int, exclude: Optional[Tensor] = None) -> List[Tensor]:
            """Return list of n_needed parent tensors [N, n_var] via hard distinct sampling."""
            idx_list = self._sample_distinct_indices(N, n_needed, self.device, exclude=exclude)
            return [self.population[idx] for idx in idx_list]

        def _soft_parents(n_needed: int) -> List[Tensor]:
            """Return list of n_needed parent tensors [N, n_var] via soft selection."""
            return [self._select_parents(N) for _ in range(n_needed)]

        use_soft = self.adaptive
        target_idx = torch.arange(N, device=self.device)

        if mutation_type == "rand":
            # DE/rand: v = x_r1 + F * (x_r2 - x_r3)
            n_parents = 3 if self.variant.n_diff == 1 else 5
            if use_soft:
                parents = _soft_parents(n_parents)
            else:
                parents = _hard_parents(n_parents, exclude=target_idx)

            if self.variant.n_diff == 1:
                donor = parents[0] + F * (parents[1] - parents[2])
            else:
                donor = parents[0] + F * (parents[1] - parents[2]) + F * (parents[3] - parents[4])

        elif mutation_type == "best":
            # DE/best: v = x_best + F * (x_r1 - x_r2)
            best_idx = torch.argmin(self.fitness)
            x_best = self.population[best_idx].unsqueeze(0).expand(N, -1)

            n_parents = 2 if self.variant.n_diff == 1 else 4
            if use_soft:
                parents = _soft_parents(n_parents)
            else:
                parents = _hard_parents(n_parents, exclude=target_idx)

            if self.variant.n_diff == 1:
                donor = x_best + F * (parents[0] - parents[1])
            else:
                donor = x_best + F * (parents[0] - parents[1]) + F * (parents[2] - parents[3])

        elif mutation_type == "current_to_best":
            # DE/current-to-best: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            best_idx = torch.argmin(self.fitness)
            x_best = self.population[best_idx].unsqueeze(0).expand(N, -1)

            if use_soft:
                parents = _soft_parents(2)
            else:
                parents = _hard_parents(2, exclude=target_idx)

            donor = self.population + F * (x_best - self.population) + F * (parents[0] - parents[1])

        elif mutation_type == "current_to_rand":
            # DE/current-to-rand: v = x_i + K * (x_r1 - x_i) + F * (x_r2 - x_r3)
            K = torch.rand(N, 1, device=self.device, dtype=self.dtype)

            if use_soft:
                parents = _soft_parents(3)
            else:
                parents = _hard_parents(3, exclude=target_idx)

            donor = self.population + K * (parents[0] - self.population) + F * (parents[1] - parents[2])

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
            # Get per-individual CR if enabled
            cr_values = self._get_CR_values(self.pop_size)
            if cr_values is not None:
                trial = self.crossover(self.population, donor, cr=cr_values)
            else:
                trial = self.crossover(self.population, donor)
        else:
            # current-to-rand: no crossover, donor is the trial
            trial = donor
        
        # 3. Repair bounds
        if self.repair is not None:
            trial = self.repair(trial, self.xl, self.xu)
        else:
            # Default: clamp to bounds
            trial = clamp_to_bounds(trial, self.xl, self.xu)
        
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
            'F': float(self.F.item()),
            'per_individual_coeffs': self.per_individual_coeffs,
            'adaptive': self.adaptive,
            'differentiable': self.differentiable,
        }
        
        # Add CR from crossover operator
        if self.crossover is not None and hasattr(self.crossover, 'cr'):
            cr = self.crossover.cr
            if isinstance(cr, Tensor):
                params['CR'] = float(cr.mean().item())
            else:
                params['CR'] = float(cr)
        
        # Add selection temperature
        if hasattr(self.selection, 'temperature'):
            params['selection_temperature'] = float(self.selection.temperature.item())
        
        # Add crossover temperature
        if self.crossover is not None and hasattr(self.crossover, 'temperature'):
            params['crossover_temperature'] = float(self.crossover.temperature.item())
        
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
            f"F={float(self.F.item()):.3f}, "
            f"per_individual_coeffs={self.per_individual_coeffs}, "
            f"adaptive={self.adaptive}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def de_default(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    per_individual_coeffs: bool = False,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> "DE":
    """
    Create a default Differential Evolution instance (DE/rand/1/bin).

    This is the canonical DE configuration and the recommended starting point.

    Args:
        pop_size: Population size.
        F: Mutation scale factor.
        CR: Crossover rate.
        per_individual_coeffs: If True, sample F and CR per individual.
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
        per_individual_coeffs=per_individual_coeffs,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def de_rand_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    per_individual_coeffs: bool = False,
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
        per_individual_coeffs: If True, sample F and CR per individual.
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
        per_individual_coeffs=per_individual_coeffs,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def de_best_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    per_individual_coeffs: bool = False,
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
        per_individual_coeffs: If True, sample F and CR per individual.
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
        per_individual_coeffs=per_individual_coeffs,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def de_current_to_best_1_bin(
    pop_size: int = 100,
    F: float = 0.5,
    CR: float = 0.9,
    per_individual_coeffs: bool = False,
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
        per_individual_coeffs: If True, sample F and CR per individual.
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
        per_individual_coeffs=per_individual_coeffs,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )