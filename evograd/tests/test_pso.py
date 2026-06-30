"""Behavioural tests for Particle Swarm Optimisation — coverage added before the refactor."""
import math
import sys

import torch

from evograd.core.problem import Problem
from evograd.core.minimize import minimize
from evograd.core.termination import MaxEvaluations
from evograd.algorithms.pso import PSO


def sphere(x):
    return (x ** 2).sum(dim=-1)


def P(n=10, lo=-5.0, hi=5.0):
    return Problem(objective=sphere, n_var=n, xl=lo, xu=hi)


def test_pso_converges():
    print("\n1. PSO converges on sphere...")
    r = minimize(P(10), PSO(pop_size=30), termination=MaxEvaluations(8000), seed=0, verbose=False)
    assert r.best_fitness < 1.0, f"PSO best {r.best_fitness} !< 1.0"


def test_pso_four_modes():
    print("2. PSO four modes run, finite fitness...")
    for kw in ({}, dict(adaptive=True), dict(differentiable=True), dict(adaptive=True, differentiable=True)):
        r = minimize(P(8), PSO(pop_size=20, **kw), termination=MaxEvaluations(800), seed=1, verbose=False)
        assert math.isfinite(r.best_fitness), f"PSO {kw} non-finite fitness"


def test_pso_per_particle_coeffs():
    print("3. PSO per-particle adaptive coeffs run...")
    r = minimize(P(8), PSO(pop_size=20, adaptive=True, per_particle_coeffs=True),
                 termination=MaxEvaluations(800), seed=2, verbose=False)
    assert math.isfinite(r.best_fitness), "PSO per-particle non-finite fitness"


def test_pso_respects_bounds():
    print("4. PSO respects bounds...")
    r = minimize(P(6, -2.0, 2.0), PSO(pop_size=20), termination=MaxEvaluations(1500), seed=3, verbose=False)
    x = r.best_solution
    assert (x >= -2.0 - 1e-3).all() and (x <= 2.0 + 1e-3).all(), "PSO violated bounds"


def run_all_tests():
    ok = True
    for t in (test_pso_converges, test_pso_four_modes, test_pso_per_particle_coeffs, test_pso_respects_bounds):
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
