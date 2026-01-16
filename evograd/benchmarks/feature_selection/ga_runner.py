"""
GA runner for feature selection benchmarks.

Wraps EvoGrad's GA implementation with configuration mapping
for different modes: classic, adaptive, differentiable, full.
"""

import numpy as np
from typing import Tuple, List

from .common import import_minimize, import_ga_default, make_termination


def run_ga(
    problem,
    config: str,
    pop: int,
    max_evals: int,
    seed: int,
    device: str,
) -> Tuple[float, List[float], int]:
    """
    Run Genetic Algorithm on a feature selection problem.
    
    Args:
        problem: Feature selection problem instance.
        config: Configuration string:
            - 'classic': No gradients, no adaptive hyperparameters
            - 'adaptive': Learnable hyperparameters, no population gradients
            - 'differentiable': Population gradients, fixed hyperparameters
            - 'full': Both adaptive hyperparameters and population gradients
        pop: Population size.
        max_evals: Maximum fitness evaluations.
        seed: Random seed.
        device: Device string ('cpu', 'cuda', 'mps').
        
    Returns:
        Tuple of (best_fitness, fitness_history, n_evaluations).
    """
    minimize = import_minimize()
    ga_default = import_ga_default()

    # Map config string to flags
    adaptive = config in ("adaptive", "full")
    differentiable = config in ("differentiable", "full")
    
    # Update problem's differentiable flag to match
    if hasattr(problem, 'differentiable'):
        problem.differentiable = differentiable

    algo = ga_default(
        pop_size=pop,
        adaptive=adaptive,
        differentiable=differentiable,
        device=device,
    )
    term = make_termination(max_evals)

    res = minimize(problem, algo, termination=term, seed=seed, verbose=False)
    
    # Extract results with fallbacks
    hist = getattr(res, "best_fitness_history", [])
    best = getattr(res, "best_fitness", hist[-1] if hist else float("nan"))
    n_evals = getattr(res, "n_evals", max_evals)
    
    # Get best solution for feature recovery analysis
    best_solution = getattr(res, "best_solution", None)

    return float(best), list(map(float, hist)), int(n_evals), best_solution
