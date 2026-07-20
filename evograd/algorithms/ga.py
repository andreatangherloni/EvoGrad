"""
Genetic Algorithm (GA) implementation for EvoGrad.

This module provides a fully differentiable Genetic Algorithm that
supports both classical and gradient-enabled optimisation modes.

The GA follows the standard evolutionary cycle:
    1. Selection: Choose parents based on fitness
    2. Crossover: Recombine parents to create offspring
    3. Mutation: Introduce random variation
    4. Survival: Select individuals for next generation

All operators are pluggable via dependency injection (pymoo-style),
allowing flexible customisation of the algorithm behavior.

Differentiable Mode:
    When `differentiable=True`, the population is stored as nn.Parameter,
    enabling gradient flow through the entire evolutionary cycle.
    
    When operators have `adaptive=True`, their internal parameters
    (temperature, eta, prob, etc.) are also learnable nn.Parameters.

Example:
    >>> from evograd.algorithms import GA
    >>> from evograd.core import Problem, minimize
    >>> from evograd.operators import (
    ...     FloatRandomSampling,
    ...     TournamentSelection,
    ...     SBXCrossover,
    ...     PolynomialMutation,
    ...     MergeSurvival,
    ... )
    >>> 
    >>> # Define problem
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Create GA with operators
    >>> ga = GA(
    ...     pop_size=100,
    ...     sampling=FloatRandomSampling(),
    ...     selection=TournamentSelection(tournament_size=3),
    ...     crossover=SBXCrossover(eta=15, prob=0.9),
    ...     mutation=PolynomialMutation(eta=20),
    ...     survival=MergeSurvival(elitism=True, n_elite=1),
    ...     differentiable=True,
    ... )
    >>> 
    >>> # Run optimization (pymoo-style)
    >>> result = minimize(problem, ga, max_evals=10000, seed=42)
    >>> print(f"Best fitness: {result.best_fitness}")

Reference:
    Holland, J. H. (1992). Genetic Algorithms. Scientific American.
    Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization,
    and Machine Learning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from evograd.core.algorithm import Algorithm
from evograd.operators.repair import clamp_to_bounds

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = ["GA", "ga_default", "ga_steady_state", "ga_comma"]


class GA(Algorithm):
    """
    Genetic Algorithm (GA) for continuous optimisation.
    
    A generational evolutionary algorithm that evolves a population
    of candidate solutions through selection, crossover, mutation,
    and survival selection. Supports both classical and differentiable
    operation modes.
    
    All operators are injected via constructor (pymoo-style), enabling
    flexible algorithm configuration.
    
    Args:
        pop_size: Population size.
        sampling: Operator for initial population generation.
            Default: FloatRandomSampling()
        selection: Parent selection operator.
            Default: TournamentSelection(tournament_size=3)
        crossover: Crossover/recombination operator.
            Default: SBXCrossover(eta=15, prob=0.9)
        mutation: Mutation operator.
            Default: PolynomialMutation(eta=20)
        survival: Survival selection operator.
            Default: MergeSurvival(elitism=True, n_elite=1)
        repair: Repair operator for constraint handling.
        n_offsprings: Number of offspring per generation.
            Default: pop_size (generational GA).
        eliminate_duplicates: Duplicate handling strategy.        
        differentiable: Enable gradient flow through population.
        adaptive: Enable learnable operator parameters.
        dtype: Tensor dtype (default: torch.float32).
    
    Attributes:
        pop_size: Population size.
        n_offsprings: Number of offspring per generation.
        pop: Current population (property).
        fitness: Current fitness values (property).
        best_fitness: Best fitness found so far.
        best_solution: Best solution found so far.
        generation: Current generation number.
        n_evals: Total fitness evaluations.
    
    Example:
        >>> # Minimal setup with defaults
        >>> ga = GA(pop_size=50)
        >>> ga.initialize(problem)
        >>> 
        >>> # With custom operators
        >>> from evograd.operators import (
        ...     RouletteSelection, BlendCrossover,
        ...     GaussianMutation, CommaSurvival,
        ... )
        >>> ga = GA(
        ...     pop_size=100,
        ...     selection=RouletteSelection(),
        ...     crossover=BlendCrossover(alpha=0.5),
        ...     mutation=GaussianMutation(sigma=0.1),
        ...     survival=CommaSurvival(elitism=True, n_elite=2),
        ...     n_offsprings=200,  # Required: >= pop_size for comma
        ... )
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        sampling: Optional[nn.Module] = None,
        selection: Optional[nn.Module] = None,
        crossover: Optional[nn.Module] = None,
        mutation: Optional[nn.Module] = None,
        survival: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        n_offsprings: Optional[int] = None,
        eliminate_duplicates: bool = True,
        differentiable: bool = True,
        adaptive: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Create default operators if not provided
        if selection is None:
            from evograd.operators.selection import TournamentSelection
            selection = TournamentSelection(
                tournament_size=3,
                adaptive=adaptive,
            )
        
        if crossover is None:
            from evograd.operators.crossover import SBXCrossover
            crossover = SBXCrossover(
                eta=15,
                prob=0.9,
                adaptive=adaptive,
            )
        
        if mutation is None:
            from evograd.operators.mutation import PolynomialMutation
            mutation = PolynomialMutation(
                eta=20,
                prob=None,  # Default: 1/n_var
                adaptive=adaptive,
            )
        
        if survival is None:
            from evograd.operators.survival import MergeSurvival
            survival = MergeSurvival(
                n_survive=pop_size,
                elitism=True,
                n_elite=1,
                adaptive=adaptive,
            )
        
        is_adaptive = selection.adaptive or crossover.adaptive or mutation.adaptive
        
        if not is_adaptive and survival.adaptive:
            adaptive = False
            survival.adaptive = False
            for p in survival.parameters():
                p.requires_grad = False
            
        
        # Call super().__init__() first before any attribute assignments
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            repair=repair,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            differentiable=differentiable,
            adaptive=adaptive,
            dtype=dtype,
        )
            
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
    # Core GA Methods
    # =========================================================================
    
    def _setup(self) -> None:
        """
        GA-specific setup after initialization.
        
        Called by initialize() after population is created.
        """
        # Update survival operator's n_survive if not set
        if hasattr(self.survival, 'n_survive'):
            if self.survival.n_survive is None:
                self.survival.n_survive = self.pop_size
    
    def _infill(self) -> Tensor:
        """
        Generate offspring through selection, crossover, and mutation.
        
        This implements the main GA variation operators:
        1. Select parents based on fitness
        2. Apply crossover to create offspring
        3. Apply mutation to introduce variation
        
        Returns:
            Offspring population tensor [n_offsprings, n_var].
        """
        n_offspring = self.n_offsprings
        
        # Number of parent pairs needed
        n_pairs = n_offspring
        
        # 1. SELECTION: Choose parents
        # Select 2 * n_pairs parents (to form pairs)
        parents = self.selection(
            self.population,
            self.fitness,
            n_select=2 * n_pairs,
        )
        
        # Split into parent pairs
        parent1 = parents[:n_pairs]
        parent2 = parents[n_pairs:2*n_pairs]
        
        # 2. CROSSOVER: Recombine parents
        offspring = self.crossover(parent1, parent2)
        
        # 3. MUTATION: Introduce variation
        offspring = self.mutation(offspring, self.xl, self.xu)

        # 4. Bound safety: when no repair operator is configured, clamp to the
        #    box so out-of-bounds points are never evaluated regardless of the
        #    mutation/crossover operators (mirrors DE/PSO/CMA-ES/SHADE). When a
        #    repair IS configured the base step/forward applies it. This is a
        #    no-op for the bounded default PolynomialMutation.
        if self.repair is None:
            offspring = clamp_to_bounds(offspring, self.xl, self.xu)

        return offspring
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Update population using survival selection.
        
        Delegates to the survival operator to select which individuals
        survive to the next generation.
        
        Args:
            offspring: Offspring population tensor.
            offspring_fitness: Fitness values of offspring.
        """
        # Use survival operator
        survivors, survivor_fitness = self.survival(
            self.population,
            self.fitness,
            offspring,
            offspring_fitness,
            n_survive=self.pop_size,
        )
        
        # Update population
        self._update_population(survivors, survivor_fitness)
        
        # Update best solution tracking
        self.state.update_best(self.population, self.state.fitness)
    
    def _update_population(self, new_population: Tensor, new_fit: Tensor) -> None:
        """
        Update population and fitness tensors.
        
        Args:
            new_pop: New population tensor.
            new_fit: New fitness tensor.
        """
        with torch.no_grad():
            self._population.copy_(new_population)
        self.state.fitness = new_fit
        self.state.population = self._population
    
    # =========================================================================
    # Hyperparameter Access
    # =========================================================================
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """
        Return current hyperparameter values.
        
        Collects hyperparameters from all operators for logging.
        
        Returns:
            Dictionary of hyperparameter names to values.
        """
        params = {
            'pop_size': self.pop_size,
            'n_offsprings': self.n_offsprings,
        }
        
        # Add selection parameters
        if hasattr(self.selection, 'temperature'):
            params['selection_temperature'] = self.selection.temperature.item()
        if hasattr(self.selection, 'tournament_size'):
            params['tournament_size'] = self.selection.tournament_size
        
        # Add crossover parameters
        if hasattr(self.crossover, 'eta'):
            params['crossover_eta'] = self.crossover.eta.item()
        if hasattr(self.crossover, 'prob'):
            prob = self.crossover.prob
            if isinstance(prob, Tensor):
                params['crossover_prob'] = prob.mean().item()
            else:
                params['crossover_prob'] = prob
        if hasattr(self.crossover, 'temperature'):
            params['crossover_temperature'] = self.crossover.temperature.item()
        
        # Add mutation parameters
        if hasattr(self.mutation, 'eta'):
            params['mutation_eta'] = self.mutation.eta.item()
        if hasattr(self.mutation, 'prob'):
            prob = self.mutation.prob
            if prob is not None:
                if isinstance(prob, Tensor):
                    params['mutation_prob'] = prob.mean().item()
                else:
                    params['mutation_prob'] = prob
        if hasattr(self.mutation, 'temperature'):
            params['mutation_temperature'] = self.mutation.temperature.item()
        
        # Add survival parameters
        if hasattr(self.survival, 'elitism'):
            params['elitism'] = self.survival.elitism
        if hasattr(self.survival, 'n_elite'):
            params['n_elite'] = self.survival.n_elite
        if hasattr(self.survival, 'temperature'):
            params['survival_temperature'] = self.survival.temperature.item()
        
        return params
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        survival_name = type(self.survival).__name__
        return (
            f"GA(pop_size={self.pop_size}, "
            f"n_offsprings={self.n_offsprings}, "
            f"survival={survival_name}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def ga_default(
    pop_size: int = 100,
    differentiable: bool = True,
    adaptive: bool = True,
    **kwargs,
) -> GA:
    """
    Create a GA with sensible default operators.
    
    Uses:
        - TournamentSelection (k=3)
        - SBXCrossover (eta=15, prob=0.9)
        - PolynomialMutation (eta=20)
        - MergeSurvival with elitism (n_elite=1)
    
    Args:
        pop_size: Population size.
        differentiable: Enable gradient flow.
        adaptive: Enable learnable operator parameters.
        **kwargs: Additional arguments passed to GA.
    
    Returns:
        Configured GA instance.
    
    Example:
        >>> ga = ga_default(pop_size=50)
        >>> result = minimize(problem, ga)
    """
    from evograd.operators.survival import MergeSurvival
    
    return GA(
        pop_size=pop_size,
        survival=MergeSurvival(
            n_survive=pop_size,
            elitism=True,
            n_elite=1,
            adaptive=adaptive,
        ),
        differentiable=differentiable,
        adaptive=adaptive,
        **kwargs,
    )


def ga_steady_state(
    pop_size: int = 100,
    n_offsprings: int = 2,
    differentiable: bool = True,
    adaptive: bool = True,
    **kwargs,
) -> GA:
    """
    Create a steady-state GA.
    
    In steady-state GA, only a few offspring are created per
    generation and they replace the worst individuals.
    
    Args:
        pop_size: Population size.
        n_offsprings: Number of offspring per generation.
        differentiable: Enable gradient flow.
        adaptive: Enable learnable operator parameters.
        **kwargs: Additional arguments passed to GA.
    
    Returns:
        Configured GA instance.
    
    Example:
        >>> ga = ga_steady_state(pop_size=50, n_offsprings=2)
        >>> result = minimize(problem, ga)
    """
    from evograd.operators.survival import ReplaceWorstSurvival
    
    return GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        survival=ReplaceWorstSurvival(
            n_survive=pop_size,
            elitism=True,
            n_elite=1,
            adaptive=adaptive,
        ),
        differentiable=differentiable,
        adaptive=adaptive,
        **kwargs,
    )


def ga_comma(
    pop_size: int = 50,
    n_offsprings: int = 100,
    differentiable: bool = True,
    adaptive: bool = True,
    **kwargs,
) -> GA:
    """
    Create a (μ, λ) style GA.
    
    In (μ, λ) selection, parents are discarded and the next
    generation is selected only from offspring. This can help
    escape local optima but requires n_offsprings >= pop_size.
    
    Args:
        pop_size: Population size (μ).
        n_offsprings: Number of offspring (λ).
        differentiable: Enable gradient flow.
        adaptive: Enable learnable operator parameters.
        **kwargs: Additional arguments passed to GA.
    
    Returns:
        Configured GA instance.
    
    Example:
        >>> ga = ga_comma(pop_size=50, n_offsprings=100)
        >>> result = minimize(problem, ga)
    """
    from evograd.operators.survival import CommaSurvival
    
    if n_offsprings < pop_size:
        raise ValueError(
            f"For (μ,λ) GA, n_offsprings ({n_offsprings}) must be >= "
            f"pop_size ({pop_size})"
        )
    
    return GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        survival=CommaSurvival(
            n_survive=pop_size,
            elitism=True,  # Still keep elitism to preserve best
            n_elite=1,
            adaptive=adaptive,
        ),
        differentiable=differentiable,
        adaptive=adaptive,
        **kwargs,
    )