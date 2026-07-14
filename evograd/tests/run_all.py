"""
Run all EvoGrad tests.

Usage:
    python -m tests.run_all
    # or
    python tests/run_all.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_utils import run_all_tests as test_utils
from tests.test_core import run_all_tests as test_core
from tests.test_operators import run_all_tests as test_operators
from tests.test_cmaes import run_all_tests as test_cmaes
from tests.test_ga import run_all_tests as test_ga
from tests.test_de import run_all_tests as test_de
from tests.test_pso import run_all_tests as test_pso
from tests.test_shade import run_all_tests as test_shade
from tests.test_per_individual import main as test_per_individual
from tests.test_regressions import run_all_tests as test_regressions


def run_all():
    """Run all test suites."""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "        EVOGRAD COMPLETE TEST SUITE".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    results = {}
    
    # Run utils tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING UTILS TESTS")
    print("▶"*60)
    results['utils'] = test_utils()
    
    # Run core tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING CORE TESTS")
    print("▶"*60)
    results['core'] = test_core()
    
    # Run operators tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING OPERATORS TESTS")
    print("▶"*60)
    results['operators'] = test_operators()

    # Run CMA-ES tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING CMA-ES TESTS")
    print("▶"*60)
    results['cmaes'] = test_cmaes()

    # Run GA tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING GA TESTS")
    print("▶"*60)
    results['ga'] = test_ga()

    # Run DE tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING DE TESTS")
    print("▶"*60)
    results['de'] = test_de()

    # Run PSO tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING PSO TESTS")
    print("▶"*60)
    results['pso'] = test_pso()

    # Run SHADE tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING SHADE TESTS")
    print("▶"*60)
    results['shade'] = test_shade()

    # Run per-individual parameter tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING PER-INDIVIDUAL TESTS")
    print("▶"*60)
    results['per_individual'] = test_per_individual()

    # Run audit regression tests
    print("\n\n" + "▶"*60)
    print("▶ RUNNING AUDIT REGRESSION TESTS")
    print("▶"*60)
    results['regressions'] = test_regressions()

    # Summary
    print("\n\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "        TEST SUMMARY".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    all_passed = True
    for module, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"   {module:20s} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("█"*60)
        print("█" + "     ✓ ALL TESTS PASSED!".center(58) + "█")
        print("█"*60)
    else:
        print("█"*60)
        print("█" + "     ✗ SOME TESTS FAILED!".center(58) + "█")
        print("█"*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
