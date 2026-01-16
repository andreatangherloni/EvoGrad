"""
Adam baseline runner for feature selection benchmarks.

Uses projected gradient descent to optimise the feature mask directly
in [0, 1] space. After each Adam update, values are clamped to maintain
feasibility.
"""

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
    
    Optimises directly in [0, 1] mask space using projected gradient
    descent. After each Adam update, values are clamped to maintain
    feasibility. Uses multiple parallel "individuals" for fair 
    comparison with population-based methods.
    
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
        
    # Directly optimise mask in [0, 1] - initialise at center
    m = torch.full((pop, problem.n_var), 0.5, device=device, requires_grad=True)
    
    opt = torch.optim.Adam([m], lr=lr, betas=(b1, b2), weight_decay=wd)

    best = float("inf")
    best_solution = None
    hist = []
    evals = 0

    while evals < max_evals:
        opt.zero_grad()
        
        # Evaluate fitness (problem handles internal clamping)
        f = problem.evaluate(m)

        # Backpropagate mean loss across population
        loss = f.mean()
        loss.backward()
        opt.step()
        
        # Project back to feasible region [0, 1]
        with torch.no_grad():
            m.clamp_(0.0, 1.0)

            # Track best solution
            min_idx = f.argmin()
            min_val = float(f[min_idx])
            if min_val < best:
                best = min_val
                best_solution = m[min_idx].detach().clone()

        hist.append(best)
        evals += pop

    return best, hist, evals, best_solution
