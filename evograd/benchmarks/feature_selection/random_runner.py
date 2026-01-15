import numpy as np
import torch


def run_random(problem, pop, max_evals, seed, device):
    rng = np.random.default_rng(seed)
    best = float("inf")
    hist = []
    evals = 0

    while evals < max_evals:
        m = rng.random((pop, problem.n_var)).astype(np.float32)
        f = problem.evaluate(torch.tensor(m, device=device))
        best = min(best, float(f.min()))
        hist.append(best)
        evals += pop

    return best, hist, evals