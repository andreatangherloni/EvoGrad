import numpy as np
from .common import import_minimize, import_ga_default, make_termination


def run_ga(problem, config, pop, max_evals, seed, device):
    minimize = import_minimize()
    ga_default = import_ga_default()

    adaptive = config in ("adaptive", "full")
    differentiable = config in ("differentiable", "full")

    algo = ga_default(pop_size=pop, adaptive=adaptive, differentiable=differentiable, device=device)
    term = make_termination(max_evals)

    res = minimize(problem, algo, termination=term, seed=seed, verbose=False)
    hist = getattr(res, "best_fitness_history", [])
    best = getattr(res, "best_fitness", hist[-1] if hist else float("nan"))
    n_evals = getattr(res, "n_evals", max_evals)

    return float(best), list(map(float, hist)), int(n_evals)