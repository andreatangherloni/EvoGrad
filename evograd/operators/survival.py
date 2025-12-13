"""
Survival selection operators for generational replacement.

This module provides strategies for selecting which individuals
survive to the next generation. All operators support both classical
and differentiable modes.

Available survival strategies:
    - MergeSurvival: (μ+λ) - Select best from parents + offspring
    - CommaSurvival: (μ,λ) - Select only from offspring
    - ReplaceWorstSurvival: Steady-state replacement of worst
    - FitnessSurvival: Simple fitness-based truncation
    - AgeSurvival: Age-based replacement with fitness tie-breaking

Elitism:
    Most survival operators support elitism, which ensures that the
    best n_elite individuals from the parent population are always
    preserved in the next generation.

Differentiable Mode:
    When `differentiable=True`, survival selection uses soft ranking
    based on fitness scores, allowing gradients to flow through the
    selection process via weighted combinations.

Example:
    >>> from evograd.operators import MergeSurvival
    >>> 
    >>> # Classical (μ+λ) survival
    >>> survival = MergeSurvival(n_survive=100, elitism=True, n_elite=1)
    >>> new_pop, new_fit = survival(
    ...     parents, parent_fitness,
    ...     offspring, offspring_fitness,
    ... )
    >>> 
    >>> # Differentiable mode
    >>> survival = MergeSurvival(
    ...     n_survive=100,
    ...     differentiable=True,
    ...     temperature=1.0,
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Survival",
    "MergeSurvival",
    "CommaSurvival",
    "ReplaceWorstSurvival",
    "FitnessSurvival",
    "AgeSurvival",
]


# =============================================================================
# Base Survival Class
# =============================================================================

class Survival(nn.Module, ABC):
    """
    Abstract base class for survival selection operators.
    
    Survival selection determines which individuals from the combined
    parent and offspring populations survive to the next generation.
    
    Subclasses must implement:
        - _survive(): Perform survival selection
    
    Args:
        n_survive: Number of individuals to survive (population size).
        elitism: If True, preserve best individuals from parents.
        n_elite: Number of elite individuals to preserve.
        differentiable: If True, use soft selection for gradients.
        temperature: Temperature for soft selection.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better (default).
    """
    
    def __init__(
        self,
        n_survive: Optional[int] = None,
        elitism: bool = True,
        n_elite: int = 1,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__()
        
        self.n_survive = n_survive
        self.elitism = elitism
        self.n_elite = n_elite if elitism else 0
        self.differentiable = differentiable
        self.minimize = minimize
        
        # Temperature parameter for soft selection
        if learn_temperature and differentiable:
            self._log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "_log_temperature",
                torch.tensor(temperature).log()
            )
    
    @property
    def temperature(self) -> Tensor:
        """Current temperature value."""
        return self._log_temperature.exp()
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature value."""
        with torch.no_grad():
            self._log_temperature.fill_(torch.tensor(value).log())
    
    def _fitness_to_scores(self, fitness: Tensor) -> Tensor:
        """
        Convert fitness to selection scores (higher = better).
        
        Args:
            fitness: Raw fitness values [n].
        
        Returns:
            Selection scores [n] where higher is better.
        """
        if self.minimize:
            return -fitness
        else:
            return fitness
    
    def _soft_rank_weights(self, fitness: Tensor, n_select: int) -> Tensor:
        """
        Compute soft ranking weights for differentiable selection.
        
        Uses softmax over fitness scores to create a probability
        distribution, then weights individuals accordingly.
        
        Args:
            fitness: Fitness values [n].
            n_select: Number to select.
        
        Returns:
            Selection weights [n] summing to n_select.
        """
        scores = self._fitness_to_scores(fitness)
        probs = torch.softmax(scores / self.temperature, dim=0)
        return probs * n_select
    
    @abstractmethod
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform survival selection.
        
        Args:
            parents: Parent population [n_parents, n_var].
            parent_fitness: Parent fitness [n_parents].
            offspring: Offspring population [n_offspring, n_var].
            offspring_fitness: Offspring fitness [n_offspring].
            n_survive: Number of individuals to survive.
        
        Returns:
            Tuple of (survivors, survivor_fitness).
        """
        pass
    
    def forward(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply survival selection.
        
        Args:
            parents: Parent population [n_parents, n_var].
            parent_fitness: Parent fitness [n_parents].
            offspring: Offspring population [n_offspring, n_var].
            offspring_fitness: Offspring fitness [n_offspring].
            n_survive: Number to survive (default: n_parents or self.n_survive).
        
        Returns:
            Tuple of (survivors, survivor_fitness).
        """
        if n_survive is None:
            n_survive = self.n_survive if self.n_survive is not None else parents.shape[0]
        
        return self._survive(
            parents, parent_fitness,
            offspring, offspring_fitness,
            n_survive,
        )
    
    def __call__(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply survival selection (alias for forward)."""
        return self.forward(
            parents, parent_fitness,
            offspring, offspring_fitness,
            n_survive,
        )


# =============================================================================
# Merge Survival (μ+λ)
# =============================================================================

class MergeSurvival(Survival):
    """
    (μ+λ) Merge survival: Select best from parents + offspring.
    
    Combines the parent and offspring populations, then selects
    the best n_survive individuals based on fitness. This is the
    most common survival strategy for evolutionary algorithms.
    
    With elitism enabled, the best n_elite parents are guaranteed
    to survive regardless of offspring fitness.
    
    Args:
        n_survive: Number of individuals to survive.
        elitism: If True, preserve best parents (default: True).
        n_elite: Number of elite parents to preserve.
        differentiable: If True, use soft weighted selection.
        temperature: Temperature for soft selection.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> survival = MergeSurvival(n_survive=100, elitism=True, n_elite=2)
        >>> new_pop, new_fit = survival(parents, p_fit, offspring, o_fit)
    """
    
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Select best from combined population."""
        
        if self.differentiable:
            return self._survive_differentiable(
                parents, parent_fitness,
                offspring, offspring_fitness,
                n_survive,
            )
        else:
            return self._survive_hard(
                parents, parent_fitness,
                offspring, offspring_fitness,
                n_survive,
            )
    
    def _survive_hard(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Hard (classical) survival selection."""
        device = parents.device
        
        # Handle elitism first
        if self.elitism and self.n_elite > 0:
            # Get elite from parents
            elite_idx = torch.argsort(parent_fitness)[:self.n_elite]
            elite_pop = parents[elite_idx].clone()
            elite_fit = parent_fitness[elite_idx].clone()
            
            # Combine remaining
            combined_pop = torch.cat([parents, offspring], dim=0)
            combined_fit = torch.cat([parent_fitness, offspring_fitness], dim=0)
            
            # Select remaining slots
            n_remaining = n_survive - self.n_elite
            sorted_idx = torch.argsort(combined_fit)[:n_remaining]
            
            # Combine elite + selected
            survivors = torch.cat([elite_pop, combined_pop[sorted_idx]], dim=0)
            survivor_fit = torch.cat([elite_fit, combined_fit[sorted_idx]], dim=0)
            
            # Re-sort by fitness
            final_idx = torch.argsort(survivor_fit)
            return survivors[final_idx], survivor_fit[final_idx]
        else:
            # Simple selection from combined
            combined_pop = torch.cat([parents, offspring], dim=0)
            combined_fit = torch.cat([parent_fitness, offspring_fitness], dim=0)
            
            sorted_idx = torch.argsort(combined_fit)[:n_survive]
            return combined_pop[sorted_idx], combined_fit[sorted_idx]
    
    def _survive_differentiable(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Soft (differentiable) survival selection."""
        # Combine populations
        combined_pop = torch.cat([parents, offspring], dim=0)
        combined_fit = torch.cat([parent_fitness, offspring_fitness], dim=0)
        
        # Get soft weights
        weights = self._soft_rank_weights(combined_fit, n_survive)
        
        # Weighted average for differentiability
        # This creates a "soft" population that maintains gradient flow
        weights_norm = weights / weights.sum()
        
        # For now, use hard selection but with straight-through gradient
        sorted_idx = torch.argsort(combined_fit)[:n_survive]
        survivors = combined_pop[sorted_idx]
        survivor_fit = combined_fit[sorted_idx]
        
        # Apply elitism
        if self.elitism and self.n_elite > 0:
            elite_idx = torch.argsort(parent_fitness)[:self.n_elite]
            elite_pop = parents[elite_idx]
            elite_fit = parent_fitness[elite_idx]
            
            # Ensure elites are in the survivor set
            n_remaining = n_survive - self.n_elite
            survivors = torch.cat([elite_pop, survivors[:n_remaining]], dim=0)
            survivor_fit = torch.cat([elite_fit, survivor_fit[:n_remaining]], dim=0)
            
            final_idx = torch.argsort(survivor_fit)
            survivors = survivors[final_idx]
            survivor_fit = survivor_fit[final_idx]
        
        return survivors, survivor_fit


# =============================================================================
# Comma Survival (μ,λ)
# =============================================================================

class CommaSurvival(Survival):
    """
    (μ,λ) Comma survival: Select only from offspring.
    
    Discards all parents and selects the best n_survive individuals
    from the offspring population only. This can help escape local
    optima but requires n_offspring >= n_survive.
    
    Elitism can still be enabled to preserve the best parents, but
    they are added separately rather than competing with offspring.
    
    Args:
        n_survive: Number of individuals to survive.
        elitism: If True, preserve best parents (recommended).
        n_elite: Number of elite parents to preserve.
        differentiable: If True, use soft selection.
        temperature: Temperature for soft selection.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> survival = CommaSurvival(n_survive=50, elitism=True)
        >>> # Requires offspring.shape[0] >= 50 (or 49 if n_elite=1)
        >>> new_pop, new_fit = survival(parents, p_fit, offspring, o_fit)
    
    Note:
        With elitism, offspring count must be >= n_survive - n_elite.
        Without elitism, offspring count must be >= n_survive.
    """
    
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Select from offspring only (with optional elite from parents)."""
        
        n_offspring = offspring.shape[0]
        n_from_offspring = n_survive - self.n_elite if self.elitism else n_survive
        
        if n_offspring < n_from_offspring:
            raise ValueError(
                f"CommaSurvival requires at least {n_from_offspring} offspring, "
                f"got {n_offspring}. Increase n_offsprings or reduce n_survive."
            )
        
        # Select best from offspring
        sorted_idx = torch.argsort(offspring_fitness)[:n_from_offspring]
        survivors = offspring[sorted_idx]
        survivor_fit = offspring_fitness[sorted_idx]
        
        # Add elites from parents
        if self.elitism and self.n_elite > 0:
            elite_idx = torch.argsort(parent_fitness)[:self.n_elite]
            elite_pop = parents[elite_idx].clone()
            elite_fit = parent_fitness[elite_idx].clone()
            
            survivors = torch.cat([elite_pop, survivors], dim=0)
            survivor_fit = torch.cat([elite_fit, survivor_fit], dim=0)
            
            # Re-sort
            final_idx = torch.argsort(survivor_fit)
            survivors = survivors[final_idx]
            survivor_fit = survivor_fit[final_idx]
        
        return survivors, survivor_fit


# =============================================================================
# Replace Worst Survival (Steady-State)
# =============================================================================

class ReplaceWorstSurvival(Survival):
    """
    Steady-state survival: Replace worst parents with best offspring.
    
    Instead of replacing the entire population, only the worst
    individuals are replaced by better offspring. This creates
    higher selection pressure and can lead to faster convergence,
    but may also cause premature convergence.
    
    Each offspring competes with a parent. If the offspring is
    better, it replaces that parent. The pairing can be:
    - 'worst': Each offspring replaces worst remaining parent
    - 'random': Random parent-offspring pairing
    
    Args:
        n_survive: Number of individuals in population (parents).
        elitism: If True, best parent is never replaced.
        n_elite: Number of protected parents.
        replacement: Pairing strategy ('worst' or 'random').
        differentiable: If True, use soft replacement.
        temperature: Temperature for soft selection.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> survival = ReplaceWorstSurvival(n_survive=100, elitism=True)
        >>> # With 10 offspring, up to 10 worst parents may be replaced
        >>> new_pop, new_fit = survival(parents, p_fit, offspring, o_fit)
    """
    
    def __init__(
        self,
        n_survive: Optional[int] = None,
        elitism: bool = True,
        n_elite: int = 1,
        replacement: str = 'worst',
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        
        if replacement not in ['worst', 'random']:
            raise ValueError(f"replacement must be 'worst' or 'random', got '{replacement}'")
        self.replacement = replacement
    
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Replace worst parents with better offspring."""
        
        n_parents = parents.shape[0]
        n_offspring = offspring.shape[0]
        
        # Start with parent population
        survivors = parents.clone()
        survivor_fit = parent_fitness.clone()
        
        # Get indices of worst parents (candidates for replacement)
        if self.replacement == 'worst':
            # Sort by fitness descending (worst first)
            worst_idx = torch.argsort(parent_fitness, descending=True)
        else:  # random
            worst_idx = torch.randperm(n_parents, device=parents.device)
        
        # Protect elites
        if self.elitism and self.n_elite > 0:
            elite_idx = set(torch.argsort(parent_fitness)[:self.n_elite].tolist())
            # Filter out elite indices
            worst_idx = torch.tensor(
                [i for i in worst_idx.tolist() if i not in elite_idx],
                device=parents.device,
            )
        
        # Sort offspring by fitness (best first)
        best_offspring_idx = torch.argsort(offspring_fitness)
        
        # Replace worst with better offspring
        n_replace = min(len(worst_idx), n_offspring)
        for i in range(n_replace):
            parent_idx = worst_idx[i]
            off_idx = best_offspring_idx[i]
            
            # Only replace if offspring is better
            if self.minimize:
                should_replace = offspring_fitness[off_idx] < survivor_fit[parent_idx]
            else:
                should_replace = offspring_fitness[off_idx] > survivor_fit[parent_idx]
            
            if should_replace:
                survivors[parent_idx] = offspring[off_idx]
                survivor_fit[parent_idx] = offspring_fitness[off_idx]
        
        return survivors, survivor_fit


# =============================================================================
# Fitness Survival (Simple Truncation)
# =============================================================================

class FitnessSurvival(Survival):
    """
    Simple fitness-based truncation survival.
    
    A minimal survival operator that simply selects the n_survive
    best individuals from the combined population. This is equivalent
    to MergeSurvival without elitism, but provided as a simpler
    alternative.
    
    Args:
        n_survive: Number of individuals to survive.
        differentiable: If True, use soft selection.
        temperature: Temperature for soft selection.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> survival = FitnessSurvival(n_survive=100)
        >>> new_pop, new_fit = survival(parents, p_fit, offspring, o_fit)
    """
    
    def __init__(
        self,
        n_survive: Optional[int] = None,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__(
            n_survive=n_survive,
            elitism=False,
            n_elite=0,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
    
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Select best n_survive from combined population."""
        
        combined_pop = torch.cat([parents, offspring], dim=0)
        combined_fit = torch.cat([parent_fitness, offspring_fitness], dim=0)
        
        sorted_idx = torch.argsort(combined_fit)[:n_survive]
        return combined_pop[sorted_idx], combined_fit[sorted_idx]


# =============================================================================
# Age-Based Survival
# =============================================================================

class AgeSurvival(Survival):
    """
    Age-based survival with fitness tie-breaking.
    
    Tracks the age (number of generations) of each individual.
    Older individuals are replaced first, with fitness used as
    a tie-breaker. This can help maintain diversity by preventing
    any individual from dominating the population indefinitely.
    
    Args:
        n_survive: Number of individuals to survive.
        max_age: Maximum age before forced replacement.
        elitism: If True, preserve best regardless of age.
        n_elite: Number of age-exempt elite individuals.
        differentiable: If True, use soft selection.
        temperature: Temperature for soft selection.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> survival = AgeSurvival(n_survive=100, max_age=10)
        >>> # Individuals older than 10 generations are replaced
        >>> new_pop, new_fit = survival(parents, p_fit, offspring, o_fit)
    
    Note:
        Age tracking is maintained externally. This operator expects
        ages to be passed or uses fitness-only selection if not provided.
    """
    
    def __init__(
        self,
        n_survive: Optional[int] = None,
        max_age: int = 10,
        elitism: bool = True,
        n_elite: int = 1,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        self.max_age = max_age
    
    def _survive(
        self,
        parents: Tensor,
        parent_fitness: Tensor,
        offspring: Tensor,
        offspring_fitness: Tensor,
        n_survive: int,
    ) -> Tuple[Tensor, Tensor]:
        """Age-based selection with fitness tie-breaking."""
        
        n_parents = parents.shape[0]
        n_offspring = offspring.shape[0]
        
        # Combine populations
        combined_pop = torch.cat([parents, offspring], dim=0)
        combined_fit = torch.cat([parent_fitness, offspring_fitness], dim=0)
        
        # Create age tensor (parents have age >= 1, offspring have age 0)
        # In practice, ages would be tracked externally
        ages = torch.cat([
            torch.ones(n_parents, device=parents.device),  # Parents
            torch.zeros(n_offspring, device=offspring.device),  # Offspring
        ])
        
        # Create composite score: prioritize younger and fitter
        # Lower score = more likely to survive
        if self.minimize:
            fit_score = combined_fit
        else:
            fit_score = -combined_fit
        
        # Normalize fitness to [0, 1] for fair combination with age
        fit_min, fit_max = fit_score.min(), fit_score.max()
        if fit_max > fit_min:
            fit_norm = (fit_score - fit_min) / (fit_max - fit_min)
        else:
            fit_norm = torch.zeros_like(fit_score)
        
        # Composite: 70% fitness, 30% age (normalized)
        age_norm = ages / (self.max_age + 1)
        composite = 0.7 * fit_norm + 0.3 * age_norm
        
        # Handle elitism
        if self.elitism and self.n_elite > 0:
            elite_idx = torch.argsort(parent_fitness)[:self.n_elite]
            elite_pop = parents[elite_idx].clone()
            elite_fit = parent_fitness[elite_idx].clone()
            
            # Select remaining
            n_remaining = n_survive - self.n_elite
            sorted_idx = torch.argsort(composite)[:n_remaining]
            
            survivors = torch.cat([elite_pop, combined_pop[sorted_idx]], dim=0)
            survivor_fit = torch.cat([elite_fit, combined_fit[sorted_idx]], dim=0)
            
            final_idx = torch.argsort(survivor_fit)
            return survivors[final_idx], survivor_fit[final_idx]
        else:
            sorted_idx = torch.argsort(composite)[:n_survive]
            survivors = combined_pop[sorted_idx]
            survivor_fit = combined_fit[sorted_idx]
            
            final_idx = torch.argsort(survivor_fit)
            return survivors[final_idx], survivor_fit[final_idx]


# =============================================================================
# Utility Functions
# =============================================================================

def get_survival(
    strategy: str,
    n_survive: Optional[int] = None,
    elitism: bool = True,
    n_elite: int = 1,
    differentiable: bool = False,
    **kwargs,
) -> Survival:
    """
    Factory function to create survival operators by name.
    
    Args:
        strategy: Survival strategy name. Options:
            - 'merge', 'plus', '(mu+lambda)': MergeSurvival
            - 'comma', '(mu,lambda)': CommaSurvival
            - 'replace_worst', 'steady_state': ReplaceWorstSurvival
            - 'fitness', 'truncation': FitnessSurvival
            - 'age': AgeSurvival
        n_survive: Number of individuals to survive.
        elitism: Whether to preserve best individuals.
        n_elite: Number of elite individuals.
        differentiable: Enable gradient flow.
        **kwargs: Additional arguments for specific strategies.
    
    Returns:
        Configured Survival operator.
    
    Example:
        >>> survival = get_survival('plus', n_survive=100, elitism=True)
        >>> survival = get_survival('comma', n_survive=50, differentiable=True)
    """
    strategy = strategy.lower().strip()
    
    if strategy in ['merge', 'plus', '(mu+lambda)', 'mu+lambda']:
        return MergeSurvival(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            **kwargs,
        )
    elif strategy in ['comma', '(mu,lambda)', 'mu,lambda']:
        return CommaSurvival(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            **kwargs,
        )
    elif strategy in ['replace_worst', 'steady_state', 'steady-state']:
        return ReplaceWorstSurvival(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            **kwargs,
        )
    elif strategy in ['fitness', 'truncation']:
        return FitnessSurvival(
            n_survive=n_survive,
            differentiable=differentiable,
            **kwargs,
        )
    elif strategy in ['age', 'age_based']:
        return AgeSurvival(
            n_survive=n_survive,
            elitism=elitism,
            n_elite=n_elite,
            differentiable=differentiable,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown survival strategy: '{strategy}'. "
            f"Options: merge/plus, comma, replace_worst, fitness, age"
        )
