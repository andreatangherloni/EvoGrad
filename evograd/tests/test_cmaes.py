"""
Test script for EvoGrad CMA-ES implementation.

Tests:
    - Default (auto) population size: lambda = 4 + floor(3*ln(n))
    - Explicit population size
    - Population tensor is allocated at the resolved pop_size
    - End-to-end minimize() runs and converges

Regression:
    With no pop_size, CMA-ES used to allocate the population at a
    placeholder size (10) and only recompute lambda afterwards, causing
    a tensor-size mismatch on the first update for any n_var where
    4 + floor(3*ln(n)) != 10. See _setup_pop_size().

Usage:
    cd evograd && python tests/test_cmaes.py
"""

import sys
import os
import math
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evograd.core.problem import Problem
from evograd.core.minimize import minimize
from evograd.core.termination import MaxEvaluations
from evograd.algorithms.cmaes import CMAES


def sphere(x):
    """Sphere function: sum of squares. Global optimum at origin."""
    return (x ** 2).sum(dim=-1)


def _expected_lambda(n_var):
    """CMA-ES default population size (lambda)."""
    return 4 + int(3 * math.log(n_var))


def make_problem(n_var):
    return Problem(objective=sphere, n_var=n_var, xl=-5.0, xu=5.0)


# =============================================================================
# Tests
# =============================================================================

def test_auto_pop_size():
    """Default pop_size must resolve to lambda and size the population."""
    print("\n1. Testing auto pop_size (no pop_size given)...")

    # Cover sizes where the auto value differs from the old placeholder (10).
    for n_var in (2, 30, 50):
        expected = _expected_lambda(n_var)

        cmaes = CMAES(sigma=0.5)
        cmaes.initialize(make_problem(n_var))

        assert cmaes.pop_size == expected, (
            f"n_var={n_var}: pop_size {cmaes.pop_size} != expected {expected}"
        )
        assert cmaes.n_offsprings == expected, (
            f"n_var={n_var}: n_offsprings {cmaes.n_offsprings} != {expected}"
        )
        assert tuple(cmaes.population.shape) == (expected, n_var), (
            f"n_var={n_var}: population shape {tuple(cmaes.population.shape)} "
            f"!= {(expected, n_var)}"
        )
        print(f"   n_var={n_var:>3}: pop_size={cmaes.pop_size} "
              f"population={tuple(cmaes.population.shape)} ✓")


def test_explicit_pop_size():
    """Explicit pop_size must be honoured exactly."""
    print("\n2. Testing explicit pop_size...")

    cmaes = CMAES(pop_size=50, sigma=0.5)
    cmaes.initialize(make_problem(30))

    assert cmaes.pop_size == 50, f"pop_size {cmaes.pop_size} != 50"
    assert tuple(cmaes.population.shape) == (50, 30)
    print(f"   pop_size={cmaes.pop_size} "
          f"population={tuple(cmaes.population.shape)} ✓")


def test_step_no_crash():
    """The first update must not raise a size mismatch (the regression)."""
    print("\n3. Testing step() with auto pop_size (regression)...")

    cmaes = CMAES(sigma=0.5)
    cmaes.initialize(make_problem(30))
    for _ in range(3):
        cmaes.step()
    print(f"   3 steps ran; generation={cmaes.generation}, "
          f"pop_size={cmaes.pop_size} ✓")


def test_convergence():
    """End-to-end minimize() should converge on the sphere function."""
    print("\n4. Testing convergence via minimize()...")

    problem = make_problem(10)
    cmaes = CMAES(sigma=0.5)  # auto pop_size
    result = minimize(
        problem, cmaes, termination=MaxEvaluations(5000),
        seed=42, verbose=False,
    )
    assert result.best_fitness < 1.0, (
        f"did not converge: best_fitness={result.best_fitness}"
    )
    print(f"   best_fitness={result.best_fitness:.6f} (< 1.0) ✓")


def run_all_tests():
    """Run all CMA-ES tests."""
    print("\n" + "#"*60)
    print("# EvoGrad CMA-ES Tests")
    print("#"*60)

    try:
        test_auto_pop_size()
        test_explicit_pop_size()
        test_step_no_crash()
        test_convergence()

        print("\n" + "="*60)
        print("✓ ALL CMA-ES TESTS PASSED!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
