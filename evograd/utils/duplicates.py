"""
Duplicate handling utilities for EvoGrad.

This module provides mechanisms to detect and repair duplicate individuals
in a population. It is intended to be used by algorithms *after* offspring
generation and boundary repair, and *before* the next objective evaluation.

Design Goals
------------
- Torch-only, GPU-friendly operations
- Pluggable detection strategies (epsilon distance, hashing)
- Minimal, well-defined API
- Support for both differentiable and non-differentiable modes

Example
-------
>>> from evograd.utils.duplicates import DuplicateEliminator, DuplicateMethod
>>>
>>> eliminator = DuplicateEliminator(
...     method=DuplicateMethod.EPSILON_L2,
...     epsilon=1e-8,
... )
>>>
>>> # Remove duplicates by resampling
>>> new_pop = eliminator(pop, lower_bounds, upper_bounds)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple, Union

import torch

from evograd.utils.device import ensure_tensor


class DuplicateMethod(Enum):
    """Strategies for duplicate detection.

    EPSILON_L2:
        Two individuals are considered duplicates if their L2 (Euclidean)
        distance is below `epsilon`.

    EPSILON_LINF:
        Two individuals are considered duplicates if their L-infinity
        (max absolute coordinate difference) is below `epsilon`.

    HASH:
        Individuals are rounded to a fixed number of decimal places
        and compared via row-wise uniqueness. Faster for high dimensions
        but less precise.

    NONE:
        Do not perform any duplicate handling.
    """

    EPSILON_L2 = auto()
    EPSILON_LINF = auto()
    HASH = auto()
    NONE = auto()


class DuplicateEliminator:
    """Eliminate duplicates in a population by resampling them.

    The class is callable:

    >>> elim = DuplicateEliminator(method=DuplicateMethod.EPSILON_L2, epsilon=1e-8)
    >>> new_pop = elim(pop, lower, upper)

    It does **not** touch fitness values or call the objective function.
    Algorithms should re-evaluate any individuals that were resampled (duplicates).
    If you call this with `return_indices=True`, you can re-evaluate only those.

    Parameters
    ----------
    method :
        Detection strategy. See `DuplicateMethod` for options.
    epsilon :
        Distance threshold for EPSILON_L2 and EPSILON_LINF methods.
    decimals :
        Number of decimal places for rounding in HASH method.
    max_resamples :
        Maximum number of resampling attempts to resolve duplicates.
        After this many attempts, remaining duplicates are accepted.

    Attributes
    ----------
    n_duplicates_found : int
        Number of duplicates found in the last call.
    n_duplicates_resolved : int
        Number of duplicates successfully resolved in the last call.
    """

    def __init__(
        self,
        method: DuplicateMethod = DuplicateMethod.EPSILON_L2,
        epsilon: float = 1e-8,
        decimals: int = 8,
        max_resamples: int = 5,
    ) -> None:
        self.method = method
        self.epsilon = float(epsilon)
        self.decimals = int(decimals)
        self.max_resamples = int(max_resamples)

        # Statistics from last call
        self.n_duplicates_found = 0
        self.n_duplicates_resolved = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        pop: torch.Tensor,
        lower: Union[float, torch.Tensor],
        upper: Union[float, torch.Tensor],
        return_indices: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return a new population where duplicates have been resampled.

        Parameters
        ----------
        pop :
            Population tensor of shape (N, D).
        lower :
            Lower bounds (scalar or 1D tensor of length D).
        upper :
            Upper bounds (scalar or 1D tensor of length D).

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If `return_indices=False` (default), returns a tensor of shape (N, D).
            If `return_indices=True`, returns (new_pop, changed_indices) where
            `changed_indices` is a 1D int64 tensor containing the indices of
            individuals that were resampled at least once.
        """
        # Reset statistics
        self.n_duplicates_found = 0
        self.n_duplicates_resolved = 0

        if self.method == DuplicateMethod.NONE:
            return (pop.clone(), torch.empty(0, device=pop.device, dtype=torch.long)) if return_indices else pop.clone()

        if pop.ndim != 2:
            raise ValueError(f"Expected population of shape (N, D), got {tuple(pop.shape)}")

        N, D = pop.shape
        device = pop.device
        dtype = pop.dtype

        # Ensure bounds are tensors
        lb = ensure_tensor(lower, dim=D, device=device, dtype=dtype)
        ub = ensure_tensor(upper, dim=D, device=device, dtype=dtype)

        if torch.any(lb > ub):
            raise ValueError("Lower bounds must be <= upper bounds elementwise.")

        # Find which indices are duplicates
        dup_mask = self._find_duplicates(pop)
        initial_dups = dup_mask.sum().item()
        self.n_duplicates_found = initial_dups

        if initial_dups == 0:
            return (pop.clone(), torch.empty(0, device=pop.device, dtype=torch.long)) if return_indices else pop.clone()

        new_pop = pop.clone()
        dup_indices = dup_mask.nonzero(as_tuple=False).view(-1)
        changed_indices = dup_indices.clone()
        span = ub - lb

        # Resample duplicates with multiple attempts
        for attempt in range(self.max_resamples):
            if dup_indices.numel() == 0:
                break

            n_dup = dup_indices.numel()

            # Generate random candidates within bounds
            candidates = lb + span * torch.rand(
                (n_dup, D), device=device, dtype=dtype
            )

            # Insert candidates
            new_pop[dup_indices] = candidates

            # Check which indices are still duplicates
            dup_mask = self._find_duplicates(new_pop)
            dup_indices = dup_mask.nonzero(as_tuple=False).view(-1)

        # Calculate resolved count
        final_dups = dup_indices.numel()
        self.n_duplicates_resolved = initial_dups - final_dups

        return (new_pop, changed_indices) if return_indices else new_pop

    def find_duplicates(self, pop: torch.Tensor) -> torch.Tensor:
        """Find duplicate individuals in a population.

        This is a public wrapper for inspection purposes.

        Parameters
        ----------
        pop :
            Population tensor of shape (N, D).

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (N,) where True indicates a duplicate.
            The first occurrence of each unique individual is NOT marked.
        """
        return self._find_duplicates(pop)

    # ------------------------------------------------------------------
    # Duplicate detection strategies
    # ------------------------------------------------------------------
    def _find_duplicates(self, pop: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of shape (N,) indicating duplicates.

        The *first* occurrence of each unique individual is considered
        non-duplicate; later occurrences are marked as duplicates.
        """
        if self.method == DuplicateMethod.EPSILON_L2:
            return self._find_duplicates_epsilon(pop, p=2)
        elif self.method == DuplicateMethod.EPSILON_LINF:
            return self._find_duplicates_epsilon(pop, p=float("inf"))
        elif self.method == DuplicateMethod.HASH:
            return self._find_duplicates_hash(pop)
        elif self.method == DuplicateMethod.NONE:
            return torch.zeros(pop.shape[0], dtype=torch.bool, device=pop.device)
        else:
            raise ValueError(f"Unhandled duplicate method: {self.method}")

    def _find_duplicates_epsilon(self, pop: torch.Tensor, p: float) -> torch.Tensor:
        """Detect duplicates based on an epsilon distance threshold.

        Parameters
        ----------
        pop :
            Population tensor of shape (N, D).
        p :
            Norm order: 2 for L2 (Euclidean), float('inf') for L-infinity.

        Returns
        -------
        torch.Tensor
            Boolean mask (N,) where True marks a duplicate individual.
        """
        N = pop.shape[0]
        device = pop.device

        if N <= 1:
            return torch.zeros(N, dtype=torch.bool, device=device)

        # Compute pairwise distances
        # For typical population sizes this is fine; can optimize later if needed
        dist = torch.cdist(pop, pop, p=p)

        # Set diagonal to infinity (ignore self-distance)
        dist.fill_diagonal_(float("inf"))

        dup_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # For each individual i, check if any previous j < i is within epsilon
        for i in range(1, N):
            if (dist[i, :i] <= self.epsilon).any():
                dup_mask[i] = True

        return dup_mask

    def _find_duplicates_hash(self, pop: torch.Tensor) -> torch.Tensor:
        """Detect duplicates by rounding and using row-wise uniqueness.

        This is usually faster than epsilon-based distance when the
        dimensionality is high, but depends on rounding precision.

        Parameters
        ----------
        pop :
            Population tensor of shape (N, D).

        Returns
        -------
        torch.Tensor
            Boolean mask (N,) where True marks a duplicate individual.
        """
        N = pop.shape[0]
        device = pop.device

        if N <= 1:
            return torch.zeros(N, dtype=torch.bool, device=device)

        # Round to specified decimal places
        scale = 10.0 ** self.decimals
        rounded = torch.round(pop * scale) / scale

        # Find unique rows
        _, inverse, counts = torch.unique(
            rounded,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        dup_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # For each group with count > 1, mark all but the first as duplicates
        for group_idx, count in enumerate(counts.tolist()):
            if count <= 1:
                continue

            indices = (inverse == group_idx).nonzero(as_tuple=False).view(-1)

            # Keep the first occurrence, mark the rest as duplicates
            if indices.numel() > 1:
                dup_mask[indices[1:]] = True

        return dup_mask

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DuplicateEliminator("
            f"method={self.method.name}, "
            f"epsilon={self.epsilon}, "
            f"decimals={self.decimals}, "
            f"max_resamples={self.max_resamples})"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def eliminate_duplicates(
    pop: torch.Tensor,
    lower: Union[float, torch.Tensor],
    upper: Union[float, torch.Tensor],
    method: DuplicateMethod = DuplicateMethod.EPSILON_L2,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Convenience function to eliminate duplicates from a population.

    Parameters
    ----------
    pop :
        Population tensor of shape (N, D).
    lower :
        Lower bounds.
    upper :
        Upper bounds.
    method :
        Detection method.
    epsilon :
        Distance threshold.

    Returns
    -------
    torch.Tensor
        Population with duplicates replaced by random samples.
    """
    eliminator = DuplicateEliminator(method=method, epsilon=epsilon)
    return eliminator(pop, lower, upper)


def has_duplicates(
    pop: torch.Tensor,
    method: DuplicateMethod = DuplicateMethod.EPSILON_L2,
    epsilon: float = 1e-8,
) -> bool:
    """Check if a population contains duplicates.

    Parameters
    ----------
    pop :
        Population tensor of shape (N, D).
    method :
        Detection method.
    epsilon :
        Distance threshold.

    Returns
    -------
    bool
        True if duplicates are found.
    """
    eliminator = DuplicateEliminator(method=method, epsilon=epsilon)
    mask = eliminator.find_duplicates(pop)
    return mask.any().item()


def count_duplicates(
    pop: torch.Tensor,
    method: DuplicateMethod = DuplicateMethod.EPSILON_L2,
    epsilon: float = 1e-8,
) -> int:
    """Count the number of duplicate individuals in a population.

    Parameters
    ----------
    pop :
        Population tensor of shape (N, D).
    method :
        Detection method.
    epsilon :
        Distance threshold.

    Returns
    -------
    int
        Number of duplicate individuals.
    """
    eliminator = DuplicateEliminator(method=method, epsilon=epsilon)
    mask = eliminator.find_duplicates(pop)
    return mask.sum().item()