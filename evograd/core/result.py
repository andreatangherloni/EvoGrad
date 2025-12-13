"""
Result container for EvoGrad optimisation.

This module provides a clean container for storing and accessing
optimisation results returned by minimize/maximize functions.

Example:
    >>> from evograd.core import minimize
    >>> 
    >>> result = minimize(algorithm, problem, max_evals=10000)
    >>> 
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> print(f"Best solution: {result.best_solution}")
    >>> print(f"Evaluations: {result.n_evals}")
    >>> print(f"Generations: {result.n_gen}")
    >>> print(f"Success: {result.success}")
    >>> 
    >>> # Access convergence history (if tracked by callback)
    >>> if result.history:
    ...     plt.plot(result.history['best_fitness'])
    >>> 
    >>> # Save/load results
    >>> result.save('result.pt')
    >>> loaded = Result.load('result.pt')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

__all__ = [
    "Result",
]


@dataclass
class Result:
    """
    Container for optimisation results.
    
    Attributes:
        best_solution: Best solution found (shape: [n_var]).
        best_fitness: Best fitness value (scalar).
        population: Final population (shape: [pop_size, n_var]).
        fitness: Final fitness values (shape: [pop_size]).
        n_evals: Total number of fitness evaluations.
        n_gen: Total number of generations.
        success: Whether optimisation succeeded (target reached, etc.).
        termination_reason: Why optimisation stopped.
        history: Convergence history (from callbacks).
        hyperparams: Final algorithm hyperparameters.
        algorithm_state: Full algorithm state for checkpointing.
        problem_name: Name of the problem.
        algorithm_name: Name of the algorithm.
        start_time: When optimisation started.
        end_time: When optimisation ended.
        elapsed_time: Total time in seconds.
        device: Device used for computation.
        extra: Additional user-defined data.
    
    Example:
        >>> result = minimize(ga, problem, max_evals=10000)
        >>> 
        >>> # Access results
        >>> print(result.best_fitness)
        >>> print(result.best_solution)
        >>> 
        >>> # Check success
        >>> if result.success:
        ...     print("Target reached!")
        >>> 
        >>> # Plot convergence
        >>> if 'best_fitness' in result.history:
        ...     plt.plot(result.history['best_fitness'])
    """
    
    # Core results
    best_solution: Tensor
    best_fitness: float
    
    # Final population state
    population: Optional[Tensor] = None
    fitness: Optional[Tensor] = None
    
    # Counts
    n_evals: int = 0
    n_gen: int = 0
    
    # Termination info
    success: bool = False
    termination_reason: Optional[str] = None
    
    # History and state
    history: Dict[str, List[Any]] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    algorithm_state: Optional[Dict[str, Any]] = None
    
    # Metadata
    problem_name: Optional[str] = None
    algorithm_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    elapsed_time: Optional[float] = None
    device: Optional[str] = None
    
    # User data
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def x(self) -> Tensor:
        """Alias for best_solution."""
        return self.best_solution
    
    @property
    def f(self) -> float:
        """Alias for best_fitness."""
        return self.best_fitness
    
    @property
    def X(self) -> Optional[Tensor]:
        """Alias for population."""
        return self.population
    
    @property
    def F(self) -> Optional[Tensor]:
        """Alias for fitness."""
        return self.fitness
    
    @property
    def n_var(self) -> int:
        """Number of variables (dimensions)."""
        return self.best_solution.shape[-1]
    
    @property
    def pop_size(self) -> Optional[int]:
        """Population size, if available."""
        if self.population is not None:
            return self.population.shape[0]
        return None
    
    # -------------------------------------------------------------------------
    # History access
    # -------------------------------------------------------------------------
    
    def get_history(self, key: str) -> Optional[List[Any]]:
        """
        Get history for a specific key.
        
        Args:
            key: History key (e.g., 'best_fitness', 'population').
        
        Returns:
            List of values or None if key not found.
        """
        return self.history.get(key)
    
    @property
    def best_fitness_history(self) -> Optional[List[float]]:
        """Convenience accessor for best fitness history."""
        return self.history.get("best_fitness")
    
    @property
    def convergence(self) -> Optional[List[float]]:
        """Alias for best_fitness_history."""
        return self.best_fitness_history
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def to_numpy(self) -> "Result":
        """
        Convert tensors to numpy arrays.
        
        Returns:
            New Result with numpy arrays instead of tensors.
        """
        import numpy as np
        
        def to_np(x: Any) -> Any:
            if isinstance(x, Tensor):
                return x.detach().cpu().numpy()
            return x
        
        return Result(
            best_solution=to_np(self.best_solution),
            best_fitness=self.best_fitness,
            population=to_np(self.population),
            fitness=to_np(self.fitness),
            n_evals=self.n_evals,
            n_gen=self.n_gen,
            success=self.success,
            termination_reason=self.termination_reason,
            history={k: [to_np(v) for v in vals] for k, vals in self.history.items()},
            hyperparams=self.hyperparams,
            algorithm_state=None,  # State may contain non-serializable items
            problem_name=self.problem_name,
            algorithm_name=self.algorithm_name,
            start_time=self.start_time,
            end_time=self.end_time,
            elapsed_time=self.elapsed_time,
            device=self.device,
            extra=self.extra,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of results.
        """
        def tensor_to_list(x: Any) -> Any:
            if isinstance(x, Tensor):
                return x.detach().cpu().tolist()
            return x
        
        return {
            "best_solution": tensor_to_list(self.best_solution),
            "best_fitness": self.best_fitness,
            "population": tensor_to_list(self.population),
            "fitness": tensor_to_list(self.fitness),
            "n_evals": self.n_evals,
            "n_gen": self.n_gen,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "history": {
                k: [tensor_to_list(v) for v in vals]
                for k, vals in self.history.items()
            },
            "hyperparams": self.hyperparams,
            "problem_name": self.problem_name,
            "algorithm_name": self.algorithm_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_time": self.elapsed_time,
            "device": self.device,
            "extra": self.extra,
        }
    
    # -------------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------------
    
    def save(
        self,
        path: Union[str, Path],
        include_state: bool = True,
        include_history: bool = True,
    ) -> None:
        """
        Save result to file.
        
        Args:
            path: File path (will use torch.save).
            include_state: Whether to include algorithm state.
            include_history: Whether to include convergence history.
        
        Example:
            >>> result.save('optimization_result.pt')
        """
        path = Path(path)
        
        data = {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "population": self.population,
            "fitness": self.fitness,
            "n_evals": self.n_evals,
            "n_gen": self.n_gen,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "hyperparams": self.hyperparams,
            "problem_name": self.problem_name,
            "algorithm_name": self.algorithm_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.elapsed_time,
            "device": self.device,
            "extra": self.extra,
        }
        
        if include_history:
            data["history"] = self.history
        
        if include_state:
            data["algorithm_state"] = self.algorithm_state
        
        torch.save(data, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Result":
        """
        Load result from file.
        
        Args:
            path: File path to load from.
        
        Returns:
            Loaded Result instance.
        
        Example:
            >>> result = Result.load('optimization_result.pt')
        """
        path = Path(path)
        data = torch.load(path, weights_only=False)
        
        return cls(
            best_solution=data["best_solution"],
            best_fitness=data["best_fitness"],
            population=data.get("population"),
            fitness=data.get("fitness"),
            n_evals=data.get("n_evals", 0),
            n_gen=data.get("n_gen", 0),
            success=data.get("success", False),
            termination_reason=data.get("termination_reason"),
            history=data.get("history", {}),
            hyperparams=data.get("hyperparams", {}),
            algorithm_state=data.get("algorithm_state"),
            problem_name=data.get("problem_name"),
            algorithm_name=data.get("algorithm_name"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            elapsed_time=data.get("elapsed_time"),
            device=data.get("device"),
            extra=data.get("extra", {}),
        )
    
    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (
            f"Result(\n"
            f"  best_fitness={self.best_fitness:.6g},\n"
            f"  n_var={self.n_var},\n"
            f"  n_evals={self.n_evals},\n"
            f"  n_gen={self.n_gen},\n"
            f"  success={self.success}\n"
            f")"
        )
    
    def summary(self) -> str:
        """
        Generate detailed summary string.
        
        Returns:
            Multi-line summary of optimisation results.
        """
        lines = [
            "=" * 60,
            "OPTIMISATION RESULT",
            "=" * 60,
        ]
        
        # Problem/Algorithm info
        if self.problem_name or self.algorithm_name:
            lines.append("")
            if self.problem_name:
                lines.append(f"Problem:    {self.problem_name}")
            if self.algorithm_name:
                lines.append(f"Algorithm:  {self.algorithm_name}")
        
        # Best result
        lines.extend([
            "",
            "Best Solution:",
            f"  Fitness:  {self.best_fitness:.10g}",
            f"  n_var:    {self.n_var}",
        ])
        
        # Show solution if small
        if self.n_var <= 10:
            sol_str = ", ".join(f"{x:.4g}" for x in self.best_solution.tolist())
            lines.append(f"  x:        [{sol_str}]")
        
        # Counts
        lines.extend([
            "",
            "Statistics:",
            f"  Evaluations:  {self.n_evals:,}",
            f"  Generations:  {self.n_gen:,}",
        ])
        
        if self.pop_size:
            lines.append(f"  Pop size:     {self.pop_size}")
        
        # Timing
        if self.elapsed_time is not None:
            lines.append(f"  Time:         {self.elapsed_time:.2f}s")
            if self.n_evals > 0:
                evals_per_sec = self.n_evals / self.elapsed_time
                lines.append(f"  Evals/sec:    {evals_per_sec:,.0f}")
        
        # Termination
        lines.extend([
            "",
            "Termination:",
            f"  Success:  {self.success}",
        ])
        if self.termination_reason:
            lines.append(f"  Reason:   {self.termination_reason}")
        
        # Hyperparameters
        if self.hyperparams:
            lines.extend(["", "Final Hyperparameters:"])
            for key, value in self.hyperparams.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6g}")
                else:
                    lines.append(f"  {key}: {value}")
        
        # History
        if self.history:
            lines.extend(["", "History Keys:"])
            for key, values in self.history.items():
                lines.append(f"  {key}: {len(values)} entries")
        
        # Device
        if self.device:
            lines.extend(["", f"Device: {self.device}"])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print detailed summary to stdout."""
        print(self.summary())


# =============================================================================
# Result Builder (for internal use by minimize/maximize)
# =============================================================================

class ResultBuilder:
    """
    Builder for constructing Result objects.
    
    Used internally by minimize/maximize functions to accumulate
    results during optimisation.
    
    Example:
        >>> builder = ResultBuilder()
        >>> builder.set_problem(problem)
        >>> builder.set_algorithm(algorithm)
        >>> builder.start()
        >>> # ... run optimisation ...
        >>> builder.finish(algorithm, termination)
        >>> result = builder.build()
    """
    
    def __init__(self) -> None:
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._problem_name: Optional[str] = None
        self._algorithm_name: Optional[str] = None
        self._device: Optional[str] = None
        self._history: Dict[str, List[Any]] = {}
        self._extra: Dict[str, Any] = {}
    
    def set_problem(self, problem: Any) -> "ResultBuilder":
        """Set problem information."""
        self._problem_name = getattr(problem, "name", None)
        return self
    
    def set_algorithm(self, algorithm: Any) -> "ResultBuilder":
        """Set algorithm information."""
        self._algorithm_name = algorithm.__class__.__name__
        self._device = str(getattr(algorithm, "device", "unknown"))
        return self
    
    def start(self) -> "ResultBuilder":
        """Mark start of optimisation."""
        self._start_time = datetime.now()
        return self
    
    def finish(
        self,
        algorithm: Any,
        termination: Optional[Any] = None,
        success: bool = False,
    ) -> "ResultBuilder":
        """
        Mark end of optimisation and capture final state.
        
        Args:
            algorithm: The algorithm instance.
            termination: The termination criterion.
            success: Whether target was reached.
        """
        self._end_time = datetime.now()
        self._algorithm = algorithm
        self._termination = termination
        self._success = success
        return self
    
    def set_history(self, history: Dict[str, List[Any]]) -> "ResultBuilder":
        """Set convergence history."""
        self._history = history
        return self
    
    def add_extra(self, key: str, value: Any) -> "ResultBuilder":
        """Add extra user data."""
        self._extra[key] = value
        return self
    
    def build(self, include_state: bool = True) -> Result:
        """
        Build the final Result object.
        
        Args:
            include_state: Whether to include full algorithm state.
        
        Returns:
            Constructed Result instance.
        """
        algorithm = self._algorithm
        termination = self._termination
        
        # Calculate elapsed time
        elapsed = None
        if self._start_time and self._end_time:
            elapsed = (self._end_time - self._start_time).total_seconds()
        
        # Get termination reason
        reason = None
        if termination is not None:
            reason = getattr(termination, "reason", None)
        
        # Get hyperparameters
        hyperparams = {}
        if hasattr(algorithm, "_get_hyperparams"):
            hyperparams = algorithm._get_hyperparams()
        
        # Get algorithm state
        state = None
        if include_state and hasattr(algorithm, "state_dict"):
            state = algorithm.state_dict()
        
        return Result(
            best_solution=algorithm.best_solution.clone(),
            best_fitness=float(algorithm.best_fitness),
            population=algorithm.population.clone() if algorithm.population is not None else None,
            fitness=algorithm.fitness.clone() if algorithm.fitness is not None else None,
            n_evals=algorithm.n_evals,
            n_gen=algorithm.generation,
            success=self._success,
            termination_reason=reason,
            history=self._history,
            hyperparams=hyperparams,
            algorithm_state=state,
            problem_name=self._problem_name,
            algorithm_name=self._algorithm_name,
            start_time=self._start_time,
            end_time=self._end_time,
            elapsed_time=elapsed,
            device=self._device,
            extra=self._extra,
        )
