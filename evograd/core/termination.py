"""
Termination criteria for EvoGrad optimisation.

This module provides termination conditions that determine when
the optimisation loop should stop. Criteria can be combined
using logical operators (AND, OR).

Available termination criteria:
    - MaxEvaluations: Stop after N fitness evaluations
    - MaxGenerations: Stop after N generations
    - TargetReached: Stop when fitness reaches target value
    - ToleranceReached: Stop when improvement falls below threshold
    - TimeLimit: Stop after N seconds
    - NoTermination: Never terminates (use with caution)

Termination criteria are passed to minimize/maximize functions,
not to the algorithm itself.

Example:
    >>> from evograd.core import MaxEvaluations, TargetReached
    >>> 
    >>> # Single criterion
    >>> termination = MaxEvaluations(10000)
    >>> 
    >>> # Combined criteria (stop when ANY is met)
    >>> termination = MaxEvaluations(10000) | TargetReached(1e-6)
    >>> 
    >>> # Combined criteria (stop when ALL are met)
    >>> termination = MaxGenerations(100) & ToleranceReached(1e-8)
    >>> 
    >>> # Use in minimize
    >>> result = minimize(algorithm, problem, termination=termination)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from evograd.core.algorithm import Algorithm

__all__ = [
    "Termination",
    "MaxEvaluations",
    "MaxGenerations",
    "TargetReached",
    "ToleranceReached",
    "TimeLimit",
    "NoTermination",
    "TerminationCollection",
    "default_termination",
]


# =============================================================================
# Base Termination Class
# =============================================================================

class Termination(ABC):
    """
    Abstract base class for termination criteria.
    
    Subclasses must implement:
        - _should_terminate(): Check if criterion is met
    
    Optionally override:
        - _update(): Update internal state each generation
        - _reset(): Reset state for new optimisation run
    
    Termination criteria can be combined using | (OR) and & (AND):
        - criterion1 | criterion2: Stop when either is met
        - criterion1 & criterion2: Stop when both are met
    """
    
    def __init__(self) -> None:
        self._is_terminated = False
        self._termination_reason: Optional[str] = None
    
    @abstractmethod
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        """
        Check if termination criterion is met.
        
        Args:
            algorithm: The algorithm instance to check.
        
        Returns:
            True if should terminate, False otherwise.
        """
        pass
    
    def _update(self, algorithm: Algorithm) -> None:
        """
        Update internal state. Called each generation.
        
        Override in subclasses that need to track history.
        
        Args:
            algorithm: The algorithm instance.
        """
        pass
    
    def _reset(self) -> None:
        """
        Reset internal state for new optimisation run.
        
        Override in subclasses with internal state.
        """
        self._is_terminated = False
        self._termination_reason = None
    
    def should_terminate(self, algorithm: Algorithm) -> bool:
        """
        Check termination and update internal state.
        
        Args:
            algorithm: The algorithm instance to check.
        
        Returns:
            True if should terminate, False otherwise.
        """
        self._update(algorithm)
        
        if self._should_terminate(algorithm):
            self._is_terminated = True
            return True
        
        return False
    
    def reset(self) -> "Termination":
        """
        Reset for new optimisation run.
        
        Returns:
            Self for method chaining.
        """
        self._reset()
        return self
    
    @property
    def is_terminated(self) -> bool:
        """Whether termination has been triggered."""
        return self._is_terminated
    
    @property
    def reason(self) -> Optional[str]:
        """Reason for termination, if terminated."""
        return self._termination_reason
    
    def __or__(self, other: "Termination") -> "TerminationCollection":
        """Combine with OR: stop when either criterion is met."""
        return TerminationCollection([self, other], mode="or")
    
    def __and__(self, other: "Termination") -> "TerminationCollection":
        """Combine with AND: stop when both criteria are met."""
        return TerminationCollection([self, other], mode="and")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Max Evaluations
# =============================================================================

class MaxEvaluations(Termination):
    """
    Terminate after maximum number of fitness evaluations.
    
    Args:
        max_evals: Maximum number of fitness evaluations.
    
    Example:
        >>> termination = MaxEvaluations(10000)
    """
    
    def __init__(self, max_evals: int) -> None:
        super().__init__()
        
        if max_evals < 1:
            raise ValueError(f"n_evals must be >= 1, got {max_evals}")
        
        self.max_evals = max_evals
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        if algorithm.n_evals >= self.max_evals:
            self._termination_reason = (
                f"Maximum evaluations reached: {algorithm.n_evals} >= {self.max_evals}"
            )
            return True
        return False
    
    def progress(self, algorithm: Algorithm) -> float:
        """Return progress as fraction [0, 1]."""
        return min(1.0, algorithm.n_evals / self.max_evals)
    
    def __repr__(self) -> str:
        return f"MaxEvaluations({self.max_evals})"


# =============================================================================
# Max Generations
# =============================================================================

class MaxGenerations(Termination):
    """
    Terminate after maximum number of generations.
    
    Args:
        max_gens: Maximum number of generations.
    
    Example:
        >>> termination = MaxGenerations(500)
    """
    
    def __init__(self, max_gens: int) -> None:
        super().__init__()
        
        if max_gens < 1:
            raise ValueError(f"n_gen must be >= 1, got {max_gens}")
        
        self.max_gens = max_gens
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        if algorithm.generation >= self.max_gens:
            self._termination_reason = (
                f"Maximum generations reached: {algorithm.generation} >= {self.max_gens}"
            )
            return True
        return False
    
    def progress(self, algorithm: Algorithm) -> float:
        """Return progress as fraction [0, 1]."""
        return min(1.0, algorithm.generation / self.max_gens)
    
    def __repr__(self) -> str:
        return f"MaxGenerations({self.max_gens})"


# =============================================================================
# Target Reached
# =============================================================================

class TargetReached(Termination):
    """
    Terminate when fitness reaches target value.
    
    For minimisation: stop when best_fitness <= target
    For maximisation: stop when best_fitness >= target
    
    Args:
        target_fitness: Target fitness value.
        minimize: If True, stop when fitness <= target.
            If False, stop when fitness >= target.
    
    Example:
        >>> # For minimisation (default)
        >>> termination = TargetReached(1e-6)
        >>> 
        >>> # For maximisation
        >>> termination = TargetReached(0.99, minimize=False)
    """
    
    def __init__(self, target_fitness: float, minimize: bool = True) -> None:
        super().__init__()
        self.target_fitness = target_fitness
        self.minimize       = minimize
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        best = algorithm.best_fitness
        
        if self.minimize:
            reached = best <= self.target_fitness
        else:
            reached = best >= self.target_fitness
        
        if reached:
            direction = "<=" if self.minimize else ">="
            self._termination_reason = (
                f"Target reached: {best:.6g} {direction} {self.target_fitness:.6g}"
            )
            return True
        
        return False
    
    def __repr__(self) -> str:
        mode = "min" if self.minimize else "max"
        return f"TargetReached({self.target_fitness}, {mode})"


# =============================================================================
# Tolerance Reached (Convergence)
# =============================================================================

class ToleranceReached(Termination):
    """
    Terminate when fitness improvement falls below tolerance.
    
    Monitors the change in best fitness over a window of generations.
    Stops when the relative or absolute change is below the threshold.
    
    Args:
        tol: Tolerance threshold for fitness change.
        n_last: Number of generations to consider for change calculation.
        mode: 'absolute' or 'relative' change measurement.
    
    Example:
        >>> # Stop when absolute change < 1e-8 over last 20 generations
        >>> termination = ToleranceReached(tol=1e-8, n_last=20)
        >>> 
        >>> # Stop when relative change < 0.1% over last 50 generations
        >>> termination = ToleranceReached(tol=0.001, n_last=50, mode='relative')
    """
    
    def __init__(
        self,
        tol: float = 1e-6,
        n_last: int = 20,
        mode: str = "absolute",
    ) -> None:
        super().__init__()
        
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if n_last < 2:
            raise ValueError(f"n_last must be >= 2, got {n_last}")
        if mode not in ("absolute", "relative"):
            raise ValueError(f"mode must be 'absolute' or 'relative', got '{mode}'")
        
        self.tol = tol
        self.n_last = n_last
        self.mode = mode
        
        self._history: List[float] = []
    
    def _reset(self) -> None:
        super()._reset()
        self._history.clear()
    
    def _update(self, algorithm: Algorithm) -> None:
        self._history.append(algorithm.best_fitness)
        
        # Keep only n_last entries
        if len(self._history) > self.n_last:
            self._history.pop(0)
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        # Need enough history
        if len(self._history) < self.n_last:
            return False
        
        old_val = self._history[0]
        new_val = self._history[-1]
        
        if self.mode == "absolute":
            change = abs(new_val - old_val)
        else:  # relative
            if abs(old_val) < 1e-10:
                change = abs(new_val - old_val)
            else:
                change = abs((new_val - old_val) / old_val)
        
        if change < self.tol:
            self._termination_reason = (
                f"Tolerance reached: {self.mode} change {change:.2e} < {self.tol:.2e} "
                f"over {self.n_last} generations"
            )
            return True
        
        return False
    
    def __repr__(self) -> str:
        return f"ToleranceReached(tol={self.tol}, n_last={self.n_last}, mode='{self.mode}')"


# =============================================================================
# Time Limit
# =============================================================================

class TimeLimit(Termination):
    """
    Terminate after time limit is reached.
    
    Args:
        max_seconds: Maximum time in seconds.
    
    Example:
        >>> # Stop after 60 seconds
        >>> termination = TimeLimit(60)
        >>> 
        >>> # Stop after 5 minutes
        >>> termination = TimeLimit(5 * 60)
    """
    
    def __init__(self, max_seconds: float) -> None:
        super().__init__()
        
        if max_seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {max_seconds}")
        
        self.max_seconds = max_seconds
        self._start_time: Optional[float] = None
    
    def _reset(self) -> None:
        super()._reset()
        self._start_time = None
    
    def _update(self, algorithm: Algorithm) -> None:
        if self._start_time is None:
            self._start_time = time.perf_counter()
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        if self._start_time is None:
            return False
        
        elapsed = time.perf_counter() - self._start_time
        
        if elapsed >= self.max_seconds:
            self._termination_reason = (
                f"Time limit reached: {elapsed:.1f}s >= {self.max_seconds:.1f}s"
            )
            return True
        
        return False
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time
    
    def progress(self, algorithm: Algorithm) -> float:
        """Return progress as fraction [0, 1]."""
        return min(1.0, self.elapsed / self.max_seconds)
    
    def __repr__(self) -> str:
        return f"TimeLimit({self.max_seconds}s)"


# =============================================================================
# No Termination
# =============================================================================

class NoTermination(Termination):
    """
    Never terminates. Use with caution!
    
    Useful when termination is handled externally or for testing.
    
    Example:
        >>> termination = NoTermination()
    """
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        return False
    
    def __repr__(self) -> str:
        return "NoTermination()"


# =============================================================================
# Termination Collection (Combined Criteria)
# =============================================================================

class TerminationCollection(Termination):
    """
    Combine multiple termination criteria.
    
    Args:
        criteria: List of termination criteria.
        mode: 'or' (stop when any is met) or 'and' (stop when all are met).
    
    Example:
        >>> # Stop when either max evals OR target reached
        >>> combined = TerminationCollection(
        ...     [MaxEvaluations(10000), TargetReached(1e-6)],
        ...     mode='or'
        ... )
        >>> 
        >>> # Or use operators
        >>> combined = MaxEvaluations(10000) | TargetReached(1e-6)
    """
    
    def __init__(
        self,
        criteria: List[Termination],
        mode: str = "or",
    ) -> None:
        super().__init__()
        
        if mode not in ("or", "and"):
            raise ValueError(f"mode must be 'or' or 'and', got '{mode}'")
        
        self.criteria = list(criteria)
        self.mode = mode
    
    def _reset(self) -> None:
        super()._reset()
        for criterion in self.criteria:
            criterion.reset()
    
    def _update(self, algorithm: Algorithm) -> None:
        for criterion in self.criteria:
            criterion._update(algorithm)
    
    def _should_terminate(self, algorithm: Algorithm) -> bool:
        results = [c._should_terminate(algorithm) for c in self.criteria]
        
        if self.mode == "or":
            # Terminate if ANY criterion is met
            if any(results):
                # Find which criterion triggered
                reasons = [
                    c.reason for c, r in zip(self.criteria, results)
                    if r and c.reason
                ]
                self._termination_reason = " OR ".join(reasons)
                return True
            return False
        else:
            # Terminate only if ALL criteria are met
            if all(results):
                reasons = [c.reason for c in self.criteria if c.reason]
                self._termination_reason = " AND ".join(reasons)
                return True
            return False
    
    def __or__(self, other: Termination) -> "TerminationCollection":
        """Add criterion with OR logic."""
        if self.mode == "or":
            return TerminationCollection(self.criteria + [other], mode="or")
        return TerminationCollection([self, other], mode="or")
    
    def __and__(self, other: Termination) -> "TerminationCollection":
        """Add criterion with AND logic."""
        if self.mode == "and":
            return TerminationCollection(self.criteria + [other], mode="and")
        return TerminationCollection([self, other], mode="and")
    
    def __repr__(self) -> str:
        op = " | " if self.mode == "or" else " & "
        criteria_str = op.join(repr(c) for c in self.criteria)
        return f"({criteria_str})"


# =============================================================================
# Convenience Functions
# =============================================================================

def default_termination(
    max_evals: Optional[int] = None,
    max_gen: Optional[int] = None,
    target: Optional[float] = None,
    tol: Optional[float] = None,
    time_limit: Optional[float] = None,
) -> Termination:
    """
    Create a termination criterion from common parameters.
    
    Multiple criteria are combined with OR (stop when any is met).
    
    Args:
        max_evals: Maximum fitness evaluations.
        max_gen: Maximum generations.
        target: Target fitness value (for minimisation).
        tol: Convergence tolerance.
        time_limit: Time limit in seconds.
    
    Returns:
        Termination criterion (single or combined).
    
    Example:
        >>> # Stop at 10000 evals or when target reached
        >>> termination = default_termination(max_evals=10000, target=1e-6)
    """
    criteria = []
    
    if max_evals is not None:
        criteria.append(MaxEvaluations(max_evals))
    
    if max_gen is not None:
        criteria.append(MaxGenerations(max_gen))
    
    if target is not None:
        criteria.append(TargetReached(target))
    
    if tol is not None:
        criteria.append(ToleranceReached(tol))
    
    if time_limit is not None:
        criteria.append(TimeLimit(time_limit))
    
    if not criteria:
        raise ValueError("At least one termination criterion must be specified")
    
    if len(criteria) == 1:
        return criteria[0]
    
    return TerminationCollection(criteria, mode="or")
