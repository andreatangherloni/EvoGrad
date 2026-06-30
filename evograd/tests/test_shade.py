"""Behavioural tests for SHADE (and LSHADE) — coverage added before the refactor."""
import math
import sys

import torch

from evograd.core.problem import Problem
from evograd.core.minimize import minimize
from evograd.core.termination import MaxEvaluations
from evograd.algorithms.shade import SHADE, LSHADE


def sphere(x):
    return (x ** 2).sum(dim=-1)


def P(n=10, lo=-5.0, hi=5.0):
    return Problem(objective=sphere, n_var=n, xl=lo, xu=hi)


def test_shade_converges():
    print("\n1. SHADE converges on sphere...")
    r = minimize(P(10), SHADE(pop_size=30), termination=MaxEvaluations(3000), seed=0, verbose=False)
    assert r.best_fitness < 1.0, f"SHADE best {r.best_fitness} !< 1.0"


def test_shade_four_modes():
    print("2. SHADE four modes run, finite fitness...")
    for kw in ({}, dict(adaptive=True), dict(differentiable=True), dict(adaptive=True, differentiable=True)):
        r = minimize(P(8), SHADE(pop_size=20, **kw), termination=MaxEvaluations(800), seed=1, verbose=False)
        assert math.isfinite(r.best_fitness), f"SHADE {kw} non-finite fitness"


def test_shade_respects_bounds():
    print("3. SHADE respects bounds...")
    r = minimize(P(6, -2.0, 2.0), SHADE(pop_size=20), termination=MaxEvaluations(1200), seed=2, verbose=False)
    x = r.best_solution
    assert (x >= -2.0 - 1e-3).all() and (x <= 2.0 + 1e-3).all(), "SHADE violated bounds"


def test_lshade_runs():
    print("4. LSHADE runs, finite fitness, respects bounds (fixed)...")
    r = minimize(P(10), LSHADE(pop_size_init=40, pop_size_min=4),
                 termination=MaxEvaluations(2000), seed=3, verbose=False)
    assert math.isfinite(r.best_fitness), "LSHADE non-finite fitness"
    x = r.best_solution
    assert (x >= -5.0 - 1e-3).all() and (x <= 5.0 + 1e-3).all(), "LSHADE violated bounds"


def run_all_tests():
    ok = True
    for t in (test_shade_converges, test_shade_four_modes, test_shade_respects_bounds, test_lshade_runs):
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
