import numpy as np
import torch


def run_adam(problem, pop, max_evals, seed, device, lr, b1, b2, wd):
    torch.manual_seed(seed)
    z = torch.zeros(pop, problem.n_var, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr, betas=(b1, b2), weight_decay=wd)

    best = float("inf")
    hist = []
    evals = 0

    while evals < max_evals:
        opt.zero_grad()
        m = torch.sigmoid(z)
        f = problem.evaluate(m)
        f.mean().backward()
        opt.step()

        best = min(best, float(f.min()))
        hist.append(best)
        evals += pop

    return best, hist, evals