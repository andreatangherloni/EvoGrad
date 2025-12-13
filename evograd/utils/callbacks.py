"""
Callbacks for monitoring and controlling the optimisation process.

This module provides a callback system inspired by Keras/PyTorch Lightning,
allowing users to hook into the optimisation loop at various points.

Available callbacks:
    - HistoryCallback: Track fitness and hyperparameter history
    - EarlyStoppingCallback: Stop when no improvement is detected
    - ConvergenceCallback: Stop when fitness change falls below threshold
    - PrintCallback: Print progress during optimisation
    - CheckpointCallback: Save algorithm state periodically
    - CompositeCallback: Combine multiple callbacks

Example:
    >>> from evograd.utils.callbacks import HistoryCallback, EarlyStoppingCallback
    >>> 
    >>> history = HistoryCallback(track_population=False)
    >>> early_stop = EarlyStoppingCallback(patience=50, min_delta=1e-6)
    >>> 
    >>> # Pass to minimize function
    >>> result = minimize(algorithm, callbacks=[history, early_stop])
    >>> 
    >>> # Access history after optimisation
    >>> print(history.best_fitness)  # List of best fitness per generation
    >>> print(history.to_dataframe())  # Pandas DataFrame if available
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "CallbackEvent",
    "Callback",
    "HistoryCallback",
    "EarlyStoppingCallback", 
    "ConvergenceCallback",
    "PrintCallback",
    "CheckpointCallback",
    "CompositeCallback",
    "CallbackList",
]


# =============================================================================
# Callback Events
# =============================================================================

class CallbackEvent(Enum):
    """Events that trigger callback methods."""
    
    OPTIMISATION_START = auto()
    OPTIMISATION_END = auto()
    GENERATION_START = auto()
    GENERATION_END = auto()
    EVALUATION_START = auto()
    EVALUATION_END = auto()
    

# =============================================================================
# Callback State Container
# =============================================================================

@dataclass
class CallbackState:
    """
    State object passed to callbacks containing current optimisation info.
    
    Attributes:
        generation: Current generation number (0-indexed).
        n_evals: Total number of fitness evaluations so far.
        max_evals: Maximum allowed fitness evaluations.
        max_generations: Maximum allowed generations (if set).
        best_fitness: Best fitness value found so far.
        best_solution: Best solution found so far.
        current_fitness: Fitness values of current population.
        current_population: Current population tensor.
        algorithm: Reference to the algorithm instance.
        hyperparams: Dictionary of current hyperparameter values.
        elapsed_time: Time elapsed since optimisation start.
        stop_optimisation: Flag to signal early stopping.
        extra: Dictionary for custom data from algorithms.
    """
    
    generation: int = 0
    n_evals: int = 0
    max_evals: Optional[int] = None
    max_generations: Optional[int] = None
    best_fitness: float = float('inf')
    best_solution: Optional[Tensor] = None
    current_fitness: Optional[Tensor] = None
    current_population: Optional[Tensor] = None
    algorithm: Optional[Any] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0
    stop_optimisation: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def request_stop(self, reason: str = "") -> None:
        """Request the optimisation loop to stop."""
        self.stop_optimisation = True
        self.extra["stop_reason"] = reason


# =============================================================================
# Base Callback
# =============================================================================

class Callback(ABC):
    """
    Abstract base class for all callbacks.
    
    Callbacks can hook into various points of the optimisation loop:
        - on_optimisation_start: Called once before optimisation begins
        - on_optimisation_end: Called once after optimisation completes
        - on_generation_start: Called at the start of each generation
        - on_generation_end: Called at the end of each generation
        - on_evaluation_start: Called before fitness evaluation
        - on_evaluation_end: Called after fitness evaluation
    
    To create a custom callback, subclass this and override the desired methods.
    
    Example:
        >>> class MyCallback(Callback):
        ...     def on_generation_end(self, state: CallbackState) -> None:
        ...         if state.generation % 100 == 0:
        ...             print(f"Gen {state.generation}: {state.best_fitness:.6f}")
    """
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Called when optimisation begins."""
        pass
    
    def on_optimisation_end(self, state: CallbackState) -> None:
        """Called when optimisation ends."""
        pass
    
    def on_generation_start(self, state: CallbackState) -> None:
        """Called at the start of each generation."""
        pass
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Called at the end of each generation."""
        pass
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """Called before fitness evaluation."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """Called after fitness evaluation."""
        pass


# =============================================================================
# History Callback
# =============================================================================

class HistoryCallback(Callback):
    """
    Track optimisation history including fitness values and hyperparameters.
    
    This callback records various metrics at each generation, providing
    a complete picture of the optimisation trajectory.
    
    Args:
        track_population: Whether to store full population at each generation.
            Warning: This can consume significant memory for large populations.
        track_hyperparams: Whether to track hyperparameter changes.
        track_diversity: Whether to compute population diversity metrics.
        track_fitness_stats: Whether to track min/max/mean/std of fitness.
    
    Attributes:
        best_fitness: List of best fitness at each generation.
        best_solution: List of best solutions at each generation.
        mean_fitness: List of mean fitness at each generation.
        std_fitness: List of fitness std at each generation.
        min_fitness: List of min fitness at each generation.
        max_fitness: List of max fitness at each generation.
        n_evals: List of cumulative evaluations at each generation.
        elapsed_time: List of elapsed time at each generation.
        hyperparams: Dict mapping hyperparam names to lists of values.
        populations: List of population tensors (if track_population=True).
        diversity: List of diversity metrics (if track_diversity=True).
    
    Example:
        >>> history = HistoryCallback(track_hyperparams=True)
        >>> result = minimize(algorithm, callbacks=[history])
        >>> 
        >>> # Plot convergence
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history.best_fitness)
        >>> plt.xlabel('Generation')
        >>> plt.ylabel('Best Fitness')
    """
    
    def __init__(
        self,
        track_population: bool = False,
        track_hyperparams: bool = True,
        track_diversity: bool = False,
        track_fitness_stats: bool = True,
    ) -> None:
        self.track_population = track_population
        self.track_hyperparams = track_hyperparams
        self.track_diversity = track_diversity
        self.track_fitness_stats = track_fitness_stats
        
        # Core tracking
        self.best_fitness: List[float] = []
        self.best_solution: List[Tensor] = []
        self.n_evals: List[int] = []
        self.elapsed_time: List[float] = []
        self.generations: List[int] = []
        
        # Fitness statistics
        self.mean_fitness: List[float] = []
        self.std_fitness: List[float] = []
        self.min_fitness: List[float] = []
        self.max_fitness: List[float] = []
        
        # Hyperparameters
        self.hyperparams: Dict[str, List[Any]] = {}
        
        # Population (optional)
        self.populations: List[Tensor] = []
        
        # Diversity (optional)
        self.diversity: List[float] = []
        
        # Timing
        self._start_time: Optional[float] = None
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Reset history and start timer."""
        self._start_time = time.perf_counter()
        
        # Clear all lists
        self.best_fitness.clear()
        self.best_solution.clear()
        self.n_evals.clear()
        self.elapsed_time.clear()
        self.generations.clear()
        self.mean_fitness.clear()
        self.std_fitness.clear()
        self.min_fitness.clear()
        self.max_fitness.clear()
        self.hyperparams.clear()
        self.populations.clear()
        self.diversity.clear()
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Record metrics at end of generation."""
        # Core metrics
        self.generations.append(state.generation)
        self.best_fitness.append(float(state.best_fitness))
        self.n_evals.append(state.n_evals)
        
        if self._start_time is not None:
            self.elapsed_time.append(time.perf_counter() - self._start_time)
        
        # Best solution (detached clone)
        if state.best_solution is not None:
            self.best_solution.append(state.best_solution.detach().clone())
        
        # Fitness statistics
        if self.track_fitness_stats and state.current_fitness is not None:
            fitness = state.current_fitness
            self.mean_fitness.append(float(fitness.mean()))
            self.std_fitness.append(float(fitness.std()))
            self.min_fitness.append(float(fitness.min()))
            self.max_fitness.append(float(fitness.max()))
        
        # Hyperparameters
        if self.track_hyperparams and state.hyperparams:
            for name, value in state.hyperparams.items():
                if name not in self.hyperparams:
                    self.hyperparams[name] = []
                # Convert tensor to scalar if needed
                if isinstance(value, torch.Tensor):
                    value = float(value.mean()) if value.numel() > 1 else float(value)
                self.hyperparams[name].append(value)
        
        # Population snapshot
        if self.track_population and state.current_population is not None:
            self.populations.append(state.current_population.detach().clone())
        
        # Diversity
        if self.track_diversity and state.current_population is not None:
            div = self._compute_diversity(state.current_population)
            self.diversity.append(div)
    
    def _compute_diversity(self, population: Tensor) -> float:
        """
        Compute population diversity as mean pairwise distance.
        
        Uses L2 norm between individuals.
        """
        if population.shape[0] < 2:
            return 0.0
        
        # Compute pairwise distances
        dists = torch.cdist(population, population, p=2)
        
        # Mean of upper triangle (excluding diagonal)
        n = population.shape[0]
        mask = torch.triu(torch.ones(n, n, device=population.device), diagonal=1).bool()
        mean_dist = dists[mask].mean()
        
        return float(mean_dist)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary."""
        result = {
            "generation": self.generations,
            "best_fitness": self.best_fitness,
            "n_evals": self.n_evals,
            "elapsed_time": self.elapsed_time,
        }
        
        if self.track_fitness_stats:
            result["mean_fitness"] = self.mean_fitness
            result["std_fitness"] = self.std_fitness
            result["min_fitness"] = self.min_fitness
            result["max_fitness"] = self.max_fitness
        
        if self.track_hyperparams:
            for name, values in self.hyperparams.items():
                result[f"hp_{name}"] = values
        
        if self.track_diversity:
            result["diversity"] = self.diversity
        
        return result
    
    def to_dataframe(self):
        """
        Convert history to pandas DataFrame.
        
        Returns:
            pandas.DataFrame if pandas is available, else raises ImportError.
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.to_dict())
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )
    
    def __len__(self) -> int:
        """Number of recorded generations."""
        return len(self.generations)
    
    def __repr__(self) -> str:
        n_gens = len(self.generations)
        best = self.best_fitness[-1] if self.best_fitness else float('inf')
        return f"HistoryCallback(generations={n_gens}, best_fitness={best:.6g})"


# =============================================================================
# Early Stopping Callback
# =============================================================================

class EarlyStoppingCallback(Callback):
    """
    Stop optimisation when fitness stops improving.
    
    Monitors the best fitness and stops if no improvement is seen
    for a specified number of generations (patience).
    
    Args:
        patience: Number of generations to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
            Improvement is defined as: new_best < best - min_delta
        baseline: Initial baseline value. If None, uses first generation's best.
        restore_best: Whether to restore best solution when stopping.
        verbose: Whether to print when stopping.
    
    Attributes:
        best_fitness: Best fitness seen so far.
        best_generation: Generation where best fitness was found.
        wait: Current number of generations without improvement.
        stopped_generation: Generation where stopping was triggered.
    
    Example:
        >>> early_stop = EarlyStoppingCallback(patience=100, min_delta=1e-8)
        >>> result = minimize(algorithm, callbacks=[early_stop])
        >>> 
        >>> if early_stop.stopped_generation is not None:
        ...     print(f"Stopped at generation {early_stop.stopped_generation}")
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.0,
        baseline: Optional[float] = None,
        restore_best: bool = True,
        verbose: bool = False,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.baseline = baseline
        self.restore_best = restore_best
        self.verbose = verbose
        
        # State
        self.best_fitness: float = float('inf')
        self.best_generation: int = 0
        self.best_solution: Optional[Tensor] = None
        self.wait: int = 0
        self.stopped_generation: Optional[int] = None
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Reset state at start of optimisation."""
        self.best_fitness = self.baseline if self.baseline is not None else float('inf')
        self.best_generation = 0
        self.best_solution = None
        self.wait = 0
        self.stopped_generation = None
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Check for improvement and possibly stop."""
        current = state.best_fitness
        
        # Check for improvement
        if current < self.best_fitness - self.min_delta:
            self.best_fitness = current
            self.best_generation = state.generation
            if state.best_solution is not None:
                self.best_solution = state.best_solution.detach().clone()
            self.wait = 0
        else:
            self.wait += 1
        
        # Check patience
        if self.wait >= self.patience:
            self.stopped_generation = state.generation
            state.request_stop(
                f"EarlyStopping: No improvement for {self.patience} generations. "
                f"Best: {self.best_fitness:.6g} at generation {self.best_generation}"
            )
            
            if self.verbose:
                print(
                    f"Early stopping at generation {state.generation}. "
                    f"Best fitness: {self.best_fitness:.6g} "
                    f"(generation {self.best_generation})"
                )
    
    def on_optimisation_end(self, state: CallbackState) -> None:
        """Optionally restore best solution."""
        if self.restore_best and self.best_solution is not None:
            # Store in extra for minimize to pick up
            state.extra["restored_solution"] = self.best_solution
            state.extra["restored_fitness"] = self.best_fitness
    
    def __repr__(self) -> str:
        status = "active" if self.stopped_generation is None else f"stopped@{self.stopped_generation}"
        return (
            f"EarlyStoppingCallback(patience={self.patience}, "
            f"min_delta={self.min_delta}, status={status})"
        )


# =============================================================================
# Convergence Callback
# =============================================================================

class ConvergenceCallback(Callback):
    """
    Stop optimisation when fitness change falls below threshold.
    
    Monitors the relative or absolute change in best fitness over a window
    of generations and stops when change is consistently small.
    
    Args:
        threshold: Convergence threshold for fitness change.
        window: Number of generations to average change over.
        mode: 'absolute' or 'relative' change measurement.
        min_generations: Minimum generations before convergence check starts.
        verbose: Whether to print when stopping.
    
    Example:
        >>> conv = ConvergenceCallback(threshold=1e-6, window=20, mode='relative')
        >>> result = minimize(algorithm, callbacks=[conv])
    """
    
    def __init__(
        self,
        threshold: float = 1e-6,
        window: int = 10,
        mode: str = "absolute",
        min_generations: int = 100,
        verbose: bool = False,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if mode not in ("absolute", "relative"):
            raise ValueError(f"mode must be 'absolute' or 'relative', got '{mode}'")
        
        self.threshold = threshold
        self.window = window
        self.mode = mode
        self.min_generations = min_generations
        self.verbose = verbose
        
        # State
        self.fitness_history: List[float] = []
        self.stopped_generation: Optional[int] = None
        self.final_change: Optional[float] = None
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Reset state."""
        self.fitness_history.clear()
        self.stopped_generation = None
        self.final_change = None
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Check convergence."""
        self.fitness_history.append(float(state.best_fitness))
        
        # Skip if not enough history
        if len(self.fitness_history) < self.window:
            return
        
        # Skip if minimum generations not reached
        if state.generation < self.min_generations:
            return
        
        # Compute change over window
        recent = self.fitness_history[-self.window:]
        old_val = recent[0]
        new_val = recent[-1]
        
        if self.mode == "absolute":
            change = abs(new_val - old_val)
        else:  # relative
            if abs(old_val) < 1e-10:
                change = abs(new_val - old_val)
            else:
                change = abs((new_val - old_val) / old_val)
        
        self.final_change = change
        
        # Check threshold
        if change < self.threshold:
            self.stopped_generation = state.generation
            state.request_stop(
                f"Convergence: {self.mode} change {change:.2e} < {self.threshold:.2e} "
                f"over {self.window} generations"
            )
            
            if self.verbose:
                print(
                    f"Converged at generation {state.generation}. "
                    f"{self.mode.capitalize()} change: {change:.2e}"
                )
    
    def __repr__(self) -> str:
        status = "active" if self.stopped_generation is None else f"stopped@{self.stopped_generation}"
        return (
            f"ConvergenceCallback(threshold={self.threshold}, "
            f"window={self.window}, mode='{self.mode}', status={status})"
        )


# =============================================================================
# Print Callback
# =============================================================================

class PrintCallback(Callback):
    """
    Print progress during optimisation.
    
    Args:
        every: Print every N generations.
        show_hyperparams: Whether to show hyperparameter values.
        show_evals: Whether to show evaluation count.
        show_time: Whether to show elapsed time.
        format_spec: Format specification for fitness (e.g., '.6f', '.4e').
    
    Example:
        >>> printer = PrintCallback(every=50, show_time=True)
        >>> result = minimize(algorithm, callbacks=[printer])
        
        # Output:
        # Gen    50 | Best: 1.234567e+02 | Evals:   5000 | Time: 1.23s
        # Gen   100 | Best: 4.567890e+01 | Evals:  10000 | Time: 2.45s
    """
    
    def __init__(
        self,
        every: int = 1,
        show_hyperparams: bool = False,
        show_evals: bool = True,
        show_time: bool = True,
        format_spec: str = ".6e",
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        
        self.every = every
        self.show_hyperparams = show_hyperparams
        self.show_evals = show_evals
        self.show_time = show_time
        self.format_spec = format_spec
        
        self._start_time: Optional[float] = None
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Record start time and print header."""
        self._start_time = time.perf_counter()
        
        # Print header
        header = "Gen"
        header += " | Best Fitness"
        if self.show_evals:
            header += " | Evals"
        if self.show_time:
            header += " | Time"
        
        print("-" * len(header))
        print(header)
        print("-" * len(header))
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Print progress if at print interval."""
        if state.generation % self.every != 0:
            return
        
        # Build output string
        parts = [f"Gen {state.generation:5d}"]
        parts.append(f"Best: {state.best_fitness:{self.format_spec}}")
        
        if self.show_evals:
            parts.append(f"Evals: {state.n_evals:7d}")
        
        if self.show_time and self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            parts.append(f"Time: {elapsed:.2f}s")
        
        if self.show_hyperparams and state.hyperparams:
            hp_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(state.hyperparams.items())[:3]  # Limit to 3
            )
            parts.append(f"HP: {hp_str}")
        
        print(" | ".join(parts))
    
    def on_optimisation_end(self, state: CallbackState) -> None:
        """Print final summary."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        print("-" * 60)
        print(f"Optimisation complete!")
        print(f"  Final best: {state.best_fitness:{self.format_spec}}")
        print(f"  Generations: {state.generation}")
        print(f"  Evaluations: {state.n_evals}")
        print(f"  Time: {elapsed:.2f}s")
        
        if state.extra.get("stop_reason"):
            print(f"  Stop reason: {state.extra['stop_reason']}")


# =============================================================================
# Checkpoint Callback
# =============================================================================

class CheckpointCallback(Callback):
    """
    Save algorithm state periodically.
    
    Args:
        directory: Directory to save checkpoints.
        every: Save every N generations.
        save_best_only: Only save when best fitness improves.
        max_to_keep: Maximum number of checkpoints to keep (None for all).
        prefix: Filename prefix for checkpoints.
    
    Example:
        >>> ckpt = CheckpointCallback(
        ...     directory="checkpoints",
        ...     every=100,
        ...     save_best_only=True
        ... )
        >>> result = minimize(algorithm, callbacks=[ckpt])
    """
    
    def __init__(
        self,
        directory: Union[str, Path],
        every: int = 100,
        save_best_only: bool = False,
        max_to_keep: Optional[int] = 5,
        prefix: str = "checkpoint",
    ) -> None:
        self.directory = Path(directory)
        self.every = every
        self.save_best_only = save_best_only
        self.max_to_keep = max_to_keep
        self.prefix = prefix
        
        # State
        self.best_fitness: float = float('inf')
        self.saved_paths: List[Path] = []
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        """Create directory and reset state."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_fitness = float('inf')
        self.saved_paths.clear()
    
    def on_generation_end(self, state: CallbackState) -> None:
        """Save checkpoint if conditions are met."""
        should_save = False
        
        if self.save_best_only:
            if state.best_fitness < self.best_fitness:
                self.best_fitness = state.best_fitness
                should_save = True
        else:
            if state.generation % self.every == 0:
                should_save = True
        
        if not should_save:
            return
        
        # Build checkpoint
        checkpoint = {
            "generation": state.generation,
            "n_evals": state.n_evals,
            "best_fitness": float(state.best_fitness),
            "best_solution": state.best_solution.detach().cpu() if state.best_solution is not None else None,
            "hyperparams": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state.hyperparams.items()
            },
        }
        
        # Save algorithm state dict if available
        if state.algorithm is not None and hasattr(state.algorithm, "state_dict"):
            checkpoint["algorithm_state"] = state.algorithm.state_dict()
        
        # Save
        filename = f"{self.prefix}_gen{state.generation:06d}.pt"
        filepath = self.directory / filename
        torch.save(checkpoint, filepath)
        self.saved_paths.append(filepath)
        
        # Clean up old checkpoints
        if self.max_to_keep is not None:
            while len(self.saved_paths) > self.max_to_keep:
                old_path = self.saved_paths.pop(0)
                if old_path.exists():
                    old_path.unlink()
    
    @staticmethod
    def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a checkpoint file."""
        return torch.load(path, weights_only=False)
    
    def __repr__(self) -> str:
        return (
            f"CheckpointCallback(directory='{self.directory}', "
            f"every={self.every}, saved={len(self.saved_paths)})"
        )


# =============================================================================
# Composite Callback
# =============================================================================

class CompositeCallback(Callback):
    """
    Combine multiple callbacks into one.
    
    This is useful for passing a single callback object that internally
    manages multiple callbacks.
    
    Args:
        callbacks: List of callbacks to combine.
    
    Example:
        >>> composite = CompositeCallback([
        ...     HistoryCallback(),
        ...     EarlyStoppingCallback(patience=50),
        ...     PrintCallback(every=10)
        ... ])
        >>> result = minimize(algorithm, callbacks=[composite])
    """
    
    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = list(callbacks)
    
    def add(self, callback: Callback) -> "CompositeCallback":
        """Add a callback."""
        self.callbacks.append(callback)
        return self
    
    def on_optimisation_start(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_optimisation_start(state)
    
    def on_optimisation_end(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_optimisation_end(state)
    
    def on_generation_start(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_generation_start(state)
    
    def on_generation_end(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_generation_end(state)
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_evaluation_start(state)
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        for cb in self.callbacks:
            cb.on_evaluation_end(state)
    
    def __len__(self) -> int:
        return len(self.callbacks)
    
    def __iter__(self):
        return iter(self.callbacks)
    
    def __repr__(self) -> str:
        return f"CompositeCallback({len(self.callbacks)} callbacks)"


# =============================================================================
# Callback List (Convenience Alias)
# =============================================================================

class CallbackList(CompositeCallback):
    """
    Alias for CompositeCallback with list-like interface.
    
    This mimics Keras's CallbackList for familiarity.
    """
    
    def append(self, callback: Callback) -> None:
        """Append a callback."""
        self.callbacks.append(callback)
    
    def extend(self, callbacks: List[Callback]) -> None:
        """Extend with multiple callbacks."""
        self.callbacks.extend(callbacks)
    
    def __getitem__(self, idx: int) -> Callback:
        return self.callbacks[idx]


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_callbacks(
    verbose: bool = True,
    history: bool = True,
    early_stopping: bool = False,
    patience: int = 50,
    print_every: int = 100,
) -> CallbackList:
    """
    Create a default set of callbacks.
    
    Args:
        verbose: Whether to include PrintCallback.
        history: Whether to include HistoryCallback.
        early_stopping: Whether to include EarlyStoppingCallback.
        patience: Patience for early stopping.
        print_every: Print interval.
    
    Returns:
        CallbackList with requested callbacks.
    """
    callbacks = CallbackList([])
    
    if history:
        callbacks.append(HistoryCallback())
    
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(patience=patience))
    
    if verbose:
        callbacks.append(PrintCallback(every=print_every))
    
    return callbacks
