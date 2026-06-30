"""
Selection operators for parent selection.

This module provides strategies for selecting parents from the
population for recombination. All selectors support both classical
(hard) and differentiable (i.e., adaptive) (Gumbel-Softmax) modes.

Available selectors:
    - TournamentSelection: Tournament-based selection
    - RouletteSelection: Fitness-proportionate selection
    - RankSelection: Rank-based selection
    - RandomSelection: Uniform random selection
    - TruncationSelection: Select *from* the top fraction (samples within truncated set)
    - TopKSelection: Deterministic top-k WITHOUT replacement (hard + differentiable)

Differentiable Mode:
    When `adaptive=True`, selection uses Gumbel-Softmax
    relaxation with straight-through estimator, allowing gradients
    to flow through the selection process.

Example:
    >>> from evograd.operators import TournamentSelection
    >>> 
    >>> # Classical mode
    >>> selector = TournamentSelection(tournament_size=3)
    >>> parents = selector(population, fitness, n_parents=50)
    >>> 
    >>> # Differentiable mode
    >>> selector = TournamentSelection(
    ...     tournament_size=3,
    ...     adaptive=True,
    ...     temperature=1.0,
    ... )
    >>> parents = selector(population, fitness, n_parents=50)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch import Tensor

from evograd.operators.relaxations import gumbel_softmax, log_param

__all__ = [
    "Selection",
    "TournamentSelection",
    "RouletteSelection",
    "RankSelection",
    "RandomSelection",
    "TopKSelection",
    "TruncationSelection",
    "StochasticUniversalSampling",
]


# =============================================================================
# Base Selection Class
# =============================================================================

class Selection(nn.Module, ABC):
    """
    Abstract base class for selection operators.
    
    Subclasses must implement:
        - _select(): Perform selection and return indices
    
    Args:
        adaptive: If True, use Gumbel-Softmax for soft selection.
        temperature: Temperature for Gumbel-Softmax (lower = harder).
        learn_temperature: If True, temperature is a learnable parameter.
        minimize: If True, lower fitness is better (default).
    """
    
    def __init__(
        self,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__()
        
        self._MIN_TEMPERATURE = 0.05
        self._MAX_TEMPERATURE = 10.0
        
        self.adaptive = adaptive
        self.minimize = minimize
        
        # Temperature parameter
        if learn_temperature and adaptive:
            # Store as log for positivity
            self._log_temperature = log_param(temperature)
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
            fitness: Raw fitness values [n_pop].
        
        Returns:
            Selection scores [n_pop] where higher is better.
        """
        if self.minimize:
            # Negate so lower fitness = higher score
            return -fitness
        else:
            return fitness
    
    
    def _clamp_temperature(self):
        # Clamp temperature for numerical stability (operator responsibility)
        if hasattr(self, "_log_temperature") and self._log_temperature is not None:
            with torch.no_grad():
                self._log_temperature.clamp_(
                    math.log(self._MIN_TEMPERATURE),
                    math.log(self._MAX_TEMPERATURE),
                )
    
    @abstractmethod
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Perform selection.
        
        Args:
            population: Population tensor [n_pop, n_var].
            fitness: Fitness values [n_pop].
            n_select: Number of individuals to select.
        
        Returns:
            Selected individuals [n_select, n_var] or
            tuple of (selected, indices) if return_indices=True.
        """
        pass
    
    def forward(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: Optional[int] = None,
        return_indices: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Select parents from population.
        
        Args:
            population: Population tensor [n_pop, n_var].
            fitness: Fitness values [n_pop].
            n_select: Number to select (default: population size).
            return_indices: If True, also return selection indices.
        
        Returns:
            Selected individuals [n_select, n_var], or
            tuple (selected, indices) if return_indices=True.
        """
        
        if n_select is None:
            n_select = population.shape[0]
        
        self._clamp_temperature()
        result = self._select(population, fitness, n_select)
        
        if return_indices:
            return result
        elif isinstance(result, tuple):
            return result[0]
        return result
    
    # Note: Do NOT override __call__. nn.Module.__call__ dispatches to
    # forward() and fires registered hooks (forward_pre_hooks, forward_hooks,
    # and the autograd profiler). Overriding __call__ would bypass all of these.


# =============================================================================
# Tournament Selection
# =============================================================================

class TournamentSelection(Selection):
    """
    Tournament selection.
    
    For each selection, randomly pick `tournament_size` individuals
    and select the best one. This is the most common selection
    operator for genetic algorithms.
    
    In differentiable mode, tournament winners are selected using
    Gumbel-Softmax over the tournament participants.
    
    Args:
        tournament_size: Number of individuals per tournament.
        adaptive: If True, use Gumbel-Softmax selection.
        temperature: Temperature for Gumbel-Softmax.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.
        replacement: If True, allow same individual in tournament.
    
    Example:
        >>> selector = TournamentSelection(tournament_size=3)
        >>> parents = selector(population, fitness, n_select=50)
    """
    
    def __init__(
        self,
        tournament_size: int = 3,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
        replacement: bool = True,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        
        if tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {tournament_size}")
        
        self.tournament_size = tournament_size
        self.replacement = replacement
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop, n_var = population.shape
        device = population.device
        k = self.tournament_size
        
        scores = self._fitness_to_scores(fitness)
        
        # Sample tournament participants
        # Shape: [n_select, tournament_size]
        if self.replacement:
            tournament_idx = torch.randint(
                0, n_pop, (n_select, k), device=device
            )
        else:
            # Without replacement (slower)
            tournament_idx = torch.stack([
                torch.randperm(n_pop, device=device)[:k]
                for _ in range(n_select)
            ])
        
        # Get scores for tournament participants
        tournament_scores = scores[tournament_idx]  # [n_select, k]
        
        if self.adaptive:
            # Gumbel-Softmax selection within each tournament
            # Use scores as logits, PASS TEMPERATURE
            weights = gumbel_softmax(
                tournament_scores, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            
            # Get tournament participants
            tournament_pop = population[tournament_idx]  # [n_select, k, n_var]
            
            # Weighted combination (hard weights = one-hot)
            selected = torch.einsum('nk,nkd->nd', weights, tournament_pop)
            
            # For indices, use argmax of weights
            relative_idx = weights.argmax(dim=-1)  # [n_select]
            indices = tournament_idx.gather(1, relative_idx.unsqueeze(-1)).squeeze(-1)
        else:
            # Hard selection: pick best in each tournament
            relative_idx = tournament_scores.argmax(dim=-1)  # [n_select]
            indices = tournament_idx.gather(1, relative_idx.unsqueeze(-1)).squeeze(-1)
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return (
            f"TournamentSelection("
            f"tournament_size={self.tournament_size}, "
            f"adaptive={self.adaptive}, "
            f"temperature={self.temperature.item():.3f})"
        )


# =============================================================================
# Roulette (Fitness Proportionate) Selection
# =============================================================================

class RouletteSelection(Selection):
    """
    Roulette wheel (fitness proportionate) selection.
    
    Selection probability is proportional to fitness. Better
    individuals have higher probability of being selected.
    
    In differentiable mode, uses Gumbel-Softmax over the
    entire population with fitness-based logits.
    
    Args:
        adaptive: If True, use Gumbel-Softmax selection.
        temperature: Temperature for Gumbel-Softmax.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.
        eps: Small constant to avoid division by zero.
    
    Example:
        >>> selector = RouletteSelection()
        >>> parents = selector(population, fitness, n_select=50)
    
    Note:
        For minimisation problems, fitness is transformed to
        ensure positive selection probabilities.
    """
    
    def __init__(
        self,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        self.eps = eps
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop = population.shape[0]
        device = population.device
        
        scores = self._fitness_to_scores(fitness)
        
        # Shift scores to be positive (for probability calculation)
        shifted_scores = scores - scores.min() + self.eps
        
        if self.adaptive:
            # Use log-probabilities as logits
            logits = torch.log(shifted_scores + self.eps)
            
            # Expand logits for n_select samples
            logits_expanded = logits.unsqueeze(0).expand(n_select, -1)
            
            # Gumbel-Softmax selection, PASS TEMPERATURE
            weights = gumbel_softmax(
                logits_expanded, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            
            # Weighted combination
            selected = torch.matmul(weights, population)  # [n_select, n_var]
            indices = weights.argmax(dim=-1)
        else:
            # Classical roulette selection
            probs = shifted_scores / shifted_scores.sum()
            indices = torch.multinomial(probs, n_select, replacement=True)
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return (
            f"RouletteSelection("
            f"adaptive={self.adaptive}, "
            f"temperature={self.temperature.item():.3f})"
        )


# =============================================================================
# Rank Selection
# =============================================================================

class RankSelection(Selection):
    """
    Rank-based selection.
    
    Selection probability is based on rank rather than raw fitness.
    This reduces selection pressure compared to fitness-proportionate
    selection and is more robust to fitness scaling.
    
    Two ranking schemes are available:
        - 'linear': Probability proportional to rank
        - 'exponential': Probability decays exponentially with rank
    
    Args:
        scheme: Ranking scheme ('linear' or 'exponential').
        selection_pressure: Controls selection intensity.
            For 'linear': in [1.0, 2.0], higher = more pressure.
            For 'exponential': decay factor, higher = more pressure.
        adaptive: If True, use Gumbel-Softmax selection.
        temperature: Temperature for Gumbel-Softmax.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> selector = RankSelection(scheme='linear', selection_pressure=1.5)
        >>> parents = selector(population, fitness, n_select=50)
    """
    
    def __init__(
        self,
        scheme: str = "linear",
        selection_pressure: float = 1.5,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        
        if scheme not in ("linear", "exponential"):
            raise ValueError(f"scheme must be 'linear' or 'exponential', got '{scheme}'")
        
        self.scheme = scheme
        self.selection_pressure = selection_pressure
    
    def _compute_rank_probabilities(
        self,
        n_pop: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Compute selection probabilities based on rank."""
        # Ranks from 1 (best) to n_pop (worst)
        ranks = torch.arange(1, n_pop + 1, device=device, dtype=dtype)
        
        if self.scheme == "linear":
            # Linear ranking: P(rank=i) = (2-s)/n + 2*(s-1)*(n-i)/(n*(n-1))
            # where s is selection pressure in [1, 2]
            s = self.selection_pressure
            n = float(n_pop)
            probs = (2 - s) / n + 2 * (s - 1) * (n - ranks) / (n * (n - 1))
        else:  # exponential
            # Exponential ranking: P(rank=i) = exp(-c * (i-1))
            c = self.selection_pressure
            probs = torch.exp(-c * (ranks - 1))
        
        # Normalise
        return probs / probs.sum()
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop = population.shape[0]
        device = population.device
        dtype = population.dtype
        
        scores = self._fitness_to_scores(fitness)
        
        # Sort by score (descending, so best first)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Compute rank-based probabilities
        rank_probs = self._compute_rank_probabilities(n_pop, device, dtype)
        
        if self.adaptive:
            # Use log-probabilities as logits
            logits = torch.log(rank_probs + 1e-10)
            logits_expanded = logits.unsqueeze(0).expand(n_select, -1)
            
            # Gumbel-Softmax selection (in rank space), PASS TEMPERATURE
            weights = gumbel_softmax(
                logits_expanded, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            
            # Map back to original indices
            sorted_pop = population[sorted_indices]
            selected = torch.matmul(weights, sorted_pop)
            
            # Get indices in original population
            rank_indices = weights.argmax(dim=-1)
            indices = sorted_indices[rank_indices]
        else:
            # Classical sampling by rank
            rank_indices = torch.multinomial(rank_probs, n_select, replacement=True)
            indices = sorted_indices[rank_indices]
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return (
            f"RankSelection("
            f"scheme='{self.scheme}', "
            f"selection_pressure={self.selection_pressure}, "
            f"adaptive={self.adaptive})"
        )


# =============================================================================
# Random Selection
# =============================================================================

class RandomSelection(Selection):
    """
    Uniform random selection (baseline).
    
    Selects individuals uniformly at random, ignoring fitness.
    Useful as a baseline or for algorithms that don't use
    fitness-based selection.
    
    Args:
        replacement: If True, allow selecting same individual multiple times.
        adaptive: If True, use Gumbel-Softmax (uniform logits).
        temperature: Temperature for Gumbel-Softmax.
    
    Example:
        >>> selector = RandomSelection()
        >>> parents = selector(population, fitness, n_select=50)
    """
    
    def __init__(
        self,
        replacement: bool = True,
        adaptive: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=False,
            minimize=True,
        )
        self.replacement = replacement
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop = population.shape[0]
        device = population.device
        
        if self.adaptive:
            # Uniform logits
            logits = torch.zeros(n_select, n_pop, device=device)
            weights = gumbel_softmax(
                logits, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            
            selected = torch.matmul(weights, population)
            indices = weights.argmax(dim=-1)
        else:
            if self.replacement:
                indices = torch.randint(0, n_pop, (n_select,), device=device)
            else:
                if n_select > n_pop:
                    raise ValueError(
                        f"Cannot select {n_select} from {n_pop} without replacement"
                    )
                indices = torch.randperm(n_pop, device=device)[:n_select]
            
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return f"RandomSelection(replacement={self.replacement})"



# =============================================================================
# Top-K Selection (Deterministic / Without Replacement)
# =============================================================================


class TopKSelection(Selection):
    """
    Top-k (elitist) selection without replacement.

    Selects the best `k` individuals according to fitness. This is the
    canonical "take the best" operator. Unlike most parent selectors,
    this returns a *subset* of the population (no duplicates).

    In differentiable mode, uses a sequential (without replacement)
    straight-through Gumbel-Softmax relaxation.

    Args:
        adaptive: If True, use differentiable top-k selection.
        temperature: Temperature for Gumbel-Softmax.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.

    Example:
        >>> # Keep best 10 individuals
        >>> selector = TopKSelection()
        >>> best10, idx = selector(population, fitness, n_select=10)
    """


    @staticmethod
    def best_indices(
        fitness: Tensor,
        k: int,
        minimize: bool = True,
    ) -> Tensor:
        """Return indices of the best-k individuals (no replacement)."""
        scores = -fitness if minimize else fitness
        return torch.topk(scores, k=k, largest=True).indices

    @staticmethod
    def worst_indices(
        fitness: Tensor,
        k: int,
        minimize: bool = True,
    ) -> Tensor:
        """Return indices of the worst-k individuals (no replacement)."""
        scores = -fitness if minimize else fitness
        return torch.topk(scores, k=k, largest=False).indices

    @staticmethod
    def sort_indices_best_first(
        fitness: Tensor,
        minimize: bool = True,
    ) -> Tensor:
        """Return indices sorting the population best->worst."""
        scores = -fitness if minimize else fitness
        return torch.argsort(scores, descending=True)

    @staticmethod
    def gumbel_topk(
        logits: Tensor,
        k: int,
        temperature: Union[float, Tensor] = 1.0,
        dim: int = -1,
        eps: float = 1e-10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Differentiable top-k without replacement using sequential Gumbel-Softmax.

        Forward pass returns hard one-hot selections (straight-through),
        and we mask out previously selected items to avoid duplicates.

        Args:
            logits: Logits over items [..., n_items].
            k: Number of items to select.
            temperature: Gumbel-Softmax temperature.
            dim: Dimension along which to select.
            eps: Numerical stability constant.

        Returns:
            weights: Selection weights [..., k, n_items] (one-hot ST).
            indices: Selected indices [..., k].
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        n_items = logits.shape[dim]
        if k > n_items:
            raise ValueError(f"Cannot select k={k} from n_items={n_items}")

        # Work on a copy we can mask in-place
        masked_logits = logits.clone()

        weights_list = []
        indices_list = []

        for _ in range(k):
            w = gumbel_softmax(
                masked_logits,
                temperature=temperature,
                dim=dim,
                eps=eps,
            )
            idx = w.argmax(dim=dim)

            weights_list.append(w)
            indices_list.append(idx)

            # Mask selected items so they cannot be chosen again
            # (use a large negative number instead of -inf for safety)
            if dim != -1:
                # Move selection dim to end for consistent masking
                perm = list(range(masked_logits.dim()))
                perm[dim], perm[-1] = perm[-1], perm[dim]
                masked_logits = masked_logits.permute(perm)

                # idx now refers to last dim
                scatter_idx = idx.unsqueeze(-1)
                masked_logits = masked_logits.scatter(-1, scatter_idx, -1e9)

                # Restore original dim order
                inv = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv[p] = i
                masked_logits = masked_logits.permute(inv)
            else:
                masked_logits = masked_logits.scatter(
                    -1, idx.unsqueeze(-1), -1e9
                )

        weights = torch.stack(weights_list, dim=-2)  # [..., k, n_items]
        indices = torch.stack(indices_list, dim=-1)  # [..., k]
        return weights, indices

    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop, n_var = population.shape
        device = population.device

        if n_select > n_pop:
            raise ValueError(f"Cannot select {n_select} from {n_pop} without replacement")

        scores = self._fitness_to_scores(fitness)

        if self.adaptive:
            # Sequential Gumbel top-k in score space
            logits = scores.unsqueeze(0)  # [1, n_pop]
            weights, idx = self.gumbel_topk(
                logits,
                k=n_select,
                temperature=self.temperature,
                dim=-1,
            )

            # weights: [1, k, n_pop]
            # Build k selected individuals as soft/hard mixtures
            selected = torch.matmul(weights.squeeze(0), population)  # [k, n_var]
            indices = idx.squeeze(0)  # [k]
        else:
            # Hard deterministic top-k
            indices = torch.topk(scores, n_select, largest=True).indices
            selected = population[indices]

        return selected, indices

    def __repr__(self) -> str:
        return (
            f"TopKSelection("
            f"adaptive={self.adaptive}, "
            f"temperature={self.temperature.item():.3f})"
        )

# =============================================================================
# Truncation Selection
# =============================================================================

class TruncationSelection(Selection):
    """
    Truncation (elitist) selection.
    
    Selects only from the top fraction of the population.
    This provides strong selection pressure.
    
    Args:
        truncation_ratio: Fraction of population to select from (0, 1].
        adaptive: If True, use softmax over truncated population.
        temperature: Temperature for soft selection.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.
    
    Example:
        >>> # Select from top 20% of population
        >>> selector = TruncationSelection(truncation_ratio=0.2)
        >>> parents = selector(population, fitness, n_select=50)
    """
    
    def __init__(
        self,
        truncation_ratio: float = 0.5,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        
        if not 0 < truncation_ratio <= 1:
            raise ValueError(
                f"truncation_ratio must be in (0, 1], got {truncation_ratio}"
            )
        
        self.truncation_ratio = truncation_ratio
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop = population.shape[0]
        device = population.device
        
        scores = self._fitness_to_scores(fitness)
        
        # Determine truncation size
        n_truncated = max(1, int(n_pop * self.truncation_ratio))
        
        # Get indices of top individuals
        _, top_indices = torch.topk(scores, n_truncated)
        
        if self.adaptive:
            # Soft selection within truncated set
            top_scores = scores[top_indices]
            logits = top_scores.unsqueeze(0).expand(n_select, -1)
            
            weights = gumbel_softmax(
                logits, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            
            top_pop = population[top_indices]
            selected = torch.matmul(weights, top_pop)
            
            relative_idx = weights.argmax(dim=-1)
            indices = top_indices[relative_idx]
        else:
            # Random selection from truncated set
            relative_idx = torch.randint(0, n_truncated, (n_select,), device=device)
            indices = top_indices[relative_idx]
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return (
            f"TruncationSelection("
            f"truncation_ratio={self.truncation_ratio}, "
            f"adaptive={self.adaptive})"
        )


# =============================================================================
# Stochastic Universal Sampling (SUS)
# =============================================================================

class StochasticUniversalSampling(Selection):
    """
    Stochastic Universal Sampling (SUS).
    
    Similar to roulette selection but uses evenly spaced pointers
    on the wheel, reducing variance and ensuring a more uniform
    sampling of fit individuals.
    
    Args:
        adaptive: If True, use Gumbel-Softmax approximation.
        temperature: Temperature for Gumbel-Softmax.
        learn_temperature: If True, temperature is learnable.
        minimize: If True, lower fitness is better.
        eps: Small constant for numerical stability.
    
    Example:
        >>> selector = StochasticUniversalSampling()
        >>> parents = selector(population, fitness, n_select=50)
    """
    
    def __init__(
        self,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        minimize: bool = True,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=learn_temperature,
            minimize=minimize,
        )
        self.eps = eps
    
    def _select(
        self,
        population: Tensor,
        fitness: Tensor,
        n_select: int,
    ) -> Tuple[Tensor, Tensor]:
        n_pop = population.shape[0]
        device = population.device
        dtype = population.dtype
        
        scores = self._fitness_to_scores(fitness)
        
        # Shift to positive
        shifted_scores = scores - scores.min() + self.eps
        total = shifted_scores.sum()
        
        if self.adaptive:
            # Fall back to Gumbel-Softmax (SUS is inherently discrete)
            logits = torch.log(shifted_scores + self.eps)
            logits_expanded = logits.unsqueeze(0).expand(n_select, -1)
            
            weights = gumbel_softmax(
                logits_expanded, 
                temperature=self.temperature,  # Pass temperature
                dim=-1
            )
            selected = torch.matmul(weights, population)
            indices = weights.argmax(dim=-1)
        else:
            # Classical SUS
            # Compute cumulative sum
            cumsum = torch.cumsum(shifted_scores, dim=0)
            
            # Distance between pointers
            pointer_distance = total / n_select
            
            # Random starting point
            start = torch.rand(1, device=device, dtype=dtype) * pointer_distance
            
            # Pointers
            pointers = start + pointer_distance * torch.arange(
                n_select, device=device, dtype=dtype
            )
            
            # Find indices where pointers land
            indices = torch.searchsorted(cumsum, pointers)
            indices = torch.clamp(indices, 0, n_pop - 1)
            
            selected = population[indices]
        
        return selected, indices
    
    def __repr__(self) -> str:
        return (
            f"StochasticUniversalSampling("
            f"adaptive={self.adaptive})"
        )
