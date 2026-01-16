"""
Random search baseline runner for feature selection benchmarks.

Samples random masks uniformly in [0, 1]^D and tracks best fitness.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List, Optional


def run_random(
    problem,
    pop: int,
    max_evals: int,
    seed: int,
    device: torch.device,
) -> Tuple[float, List[float], int, Optional[Tensor]]:
    """
    Run random search baseline on feature selection problem.
    
    Samples random masks uniformly in [0, 1]^D and evaluates them.
    Tracks the best solution found across all evaluations.
    
    Args:
        problem: Feature selection problem instance.
        pop: Number of random samples per generation.
        max_evals: Maximum fitness evaluations.
        seed: Random seed.
        device: Torch device.
        
    Returns:
        Tuple of (best_fitness, fitness_history, n_evaluations, best_solution).
    """
    rng = np.random.default_rng(seed)
    
    best = float("inf")
    best_solution = None
    hist = []
    evals = 0

    while evals < max_evals:
        # Sample random masks in [0, 1]
        m = rng.random((pop, problem.n_var)).astype(np.float32)
        m_tensor = torch.tensor(m, device=device)
        
        # Evaluate (classical mode, no gradients needed)
        with torch.no_grad():
            f = problem.evaluate(m_tensor)
        
        # Track best
        min_idx = f.argmin()
        min_val = float(f[min_idx])
        if min_val < best:
            best = min_val
            best_solution = m_tensor[min_idx].clone()
        
        hist.append(best)
        evals += pop

    return best, hist, evals, best_solution
