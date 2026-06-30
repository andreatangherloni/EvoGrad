"""Behavioural tests for Differential Evolution — coverage added before the refactor."""
import math
import sys

import torch

from evograd.core.problem import Problem
from evograd.core.minimize import minimize
from evograd.core.termination import MaxEvaluations
from evograd.algorithms.de import DE


def sphere(x):
    return (x ** 2).sum(dim=-1)


def P(n=10, lo=-5.0, hi=5.0):
    return Problem(objective=sphere, n_var=n, xl=lo, xu=hi)


def test_de_converges():
    print("\n1. DE converges on sphere...")
    r = minimize(P(10), DE(pop_size=30), termination=MaxEvaluations(8000), seed=0, verbose=False)
    assert r.best_fitness < 1.0, f"DE best {r.best_fitness} !< 1.0"


def test_de_variants():
    print("2. DE variants run, finite fitness...")
    for v in ("DE/rand/1/bin", "DE/best/1/bin", "DE/current-to-best/1/bin", "DE/rand/2/bin"):
        r = minimize(P(8), DE(pop_size=20, variant=v), termination=MaxEvaluations(1500), seed=1, verbose=False)
        assert math.isfinite(r.best_fitness), f"DE {v} non-finite fitness"


def test_de_four_modes():
    print("3. DE four modes run, finite fitness...")
    for kw in ({}, dict(adaptive=True), dict(differentiable=True), dict(adaptive=True, differentiable=True)):
        r = minimize(P(8), DE(pop_size=20, **kw), termination=MaxEvaluations(800), seed=2, verbose=False)
        assert math.isfinite(r.best_fitness), f"DE {kw} non-finite fitness"


def test_de_respects_bounds():
    print("4. DE respects bounds...")
    r = minimize(P(6, -2.0, 2.0), DE(pop_size=20), termination=MaxEvaluations(1500), seed=3, verbose=False)
    x = r.best_solution
    assert (x >= -2.0 - 1e-3).all() and (x <= 2.0 + 1e-3).all(), "DE violated bounds"


def run_all_tests():
    ok = True
    for t in (test_de_converges, test_de_variants, test_de_four_modes, test_de_respects_bounds):
        try:
            t()
            print("   PASS")
        except Exception:
            ok = False
            import traceback
            traceback.print_exc()
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
