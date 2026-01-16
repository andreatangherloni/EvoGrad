"""
Adam baseline runner for feature selection benchmarks.

Uses gradient descent with sigmoid-transformed parameters to optimise
the feature mask directly.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List, Optional


def run_adam(
    problem,
    pop: int,
    max_evals: int,
    seed: int,
    device: torch.device,
    lr: float = 0.05,
    b1: float = 0.9,
    b2: float = 0.999,
    wd: float = 0.0,
) -> Tuple[float, List[float], int, Optional[Tensor]]:
    """
    Run Adam optimiser on feature selection problem.
    
    Optimises in unconstrained space (z) and applies sigmoid to get
    mask values in [0, 1]. Uses multiple parallel "individuals" for
    fair comparison with population-based methods.
    
    Args:
        problem: Feature selection problem instance.
        pop: Number of parallel solutions (for fair eval count comparison).
        max_evals: Maximum fitness evaluations.
        seed: Random seed.
        device: Torch device.
        lr: Learning rate.
        b1: Adam beta1 parameter.
        b2: Adam beta2 parameter.
        wd: Weight decay (L2 regularisation).
        
    Returns:
        Tuple of (best_fitness, fitness_history, n_evaluations, best_solution).
    """
    torch.manual_seed(seed)
    
    # Initialise in unconstrained space (sigmoid(0) = 0.5)
    z = torch.zeros(pop, problem.n_var, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr, betas=(b1, b2), weight_decay=wd)

    best = float("inf")
    best_solution = None
    hist = []
    evals = 0

    while evals < max_evals:
        opt.zero_grad()
        
        # Transform to [0, 1] range
        m = torch.sigmoid(z)
        
        # Evaluate (problem should handle differentiable mode internally)
        # Temporarily set differentiable mode for Adam
        old_diff = getattr(problem, 'differentiable', False)
        problem.differentiable = True
        
        f = problem.evaluate(m)
        
        # Restore original mode
        problem.differentiable = old_diff
        
        # Backpropagate mean loss
        loss = f.mean()
        loss.backward()
        opt.step()

        # Track best (without gradient tracking to avoid memory leak)
        with torch.no_grad():
            min_idx = f.argmin()
            min_val = float(f[min_idx])
            if min_val < best:
                best = min_val
                best_solution = m[min_idx].detach().clone()
        
        hist.append(best)
        evals += pop

    return best, hist, evals, best_solution
