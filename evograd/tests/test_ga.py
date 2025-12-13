"""
Test script for EvoGrad GA (Genetic Algorithm) implementation.

Tests:
    - GA creation with default and custom operators
    - Different survival strategies (plus, comma, replace_worst)
    - Elitism behavior
    - Classical and differentiable modes
    - State persistence (save/load)
    - Convergence on test functions

Usage:
    cd evograd && python tests/test_ga.py
"""

import sys
import os
import tempfile
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evograd.core.problem import Problem
from evograd.algorithms.ga import GA, ga_default, ga_steady_state, ga_comma
from evograd.operators.sampling import UniformSampling
from evograd.operators.selection import TournamentSelection, RouletteSelection
from evograd.operators.crossover import SBXCrossover, BlendCrossover
from evograd.operators.mutation import PolynomialMutation, GaussianMutation
from evograd.operators.repair import ReflectRepair
from evograd.operators.survival import (
    MergeSurvival,
    CommaSurvival,
    ReplaceWorstSurvival,
    FitnessSurvival,
)


# =============================================================================
# Test Functions
# =============================================================================

def sphere(x):
    """Sphere function: sum of squares. Global optimum at origin."""
    return (x ** 2).sum(dim=-1)


def rastrigin(x):
    """Rastrigin function: highly multimodal."""
    A = 10.0
    n = x.shape[-1]
    return A * n + (x**2 - A * torch.cos(2 * torch.pi * x)).sum(dim=-1)


def rosenbrock(x):
    """Rosenbrock function: narrow valley."""
    return (100 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2).sum(dim=-1)


# =============================================================================
# Tests
# =============================================================================

def test_ga_creation():
    """Test GA creation with default operators."""
    print("\n" + "="*60)
    print("Testing GA Creation")
    print("="*60)
    
    # Test with all defaults
    print("\n1. Testing GA with default operators...")
    ga = GA(pop_size=20)
    print(f"   Created: {ga}")
    print(f"   Selection: {ga.selection}")
    print(f"   Crossover: {ga.crossover}")
    print(f"   Mutation: {ga.mutation}")
    print(f"   Survival: {ga.survival}")
    assert ga.pop_size == 20
    assert ga.selection is not None
    assert ga.crossover is not None
    assert ga.mutation is not None
    assert ga.survival is not None
    
    # Test with custom operators
    print("\n2. Testing GA with custom operators...")
    ga = GA(
        pop_size=30,
        selection=RouletteSelection(differentiable=True),
        crossover=BlendCrossover(alpha=0.5, differentiable=True),
        mutation=GaussianMutation(sigma=0.1, differentiable=True),
        survival=MergeSurvival(elitism=True, n_elite=2, differentiable=True),
    )
    print(f"   Created: {ga}")
    print(f"   Survival: {ga.survival}")
    assert ga.survival.elitism == True
    assert ga.survival.n_elite == 2
    
    # Test factory functions
    print("\n3. Testing factory functions...")
    ga1 = ga_default(pop_size=50)
    print(f"   ga_default: {ga1}")
    
    ga2 = ga_steady_state(pop_size=50, n_offsprings=2)
    print(f"   ga_steady_state: {ga2}")
    print(f"   Survival type: {type(ga2.survival).__name__}")
    assert type(ga2.survival).__name__ == 'ReplaceWorstSurvival', \
        f"Expected ReplaceWorstSurvival, got {type(ga2.survival).__name__}"
    
    ga3 = ga_comma(pop_size=30, n_offsprings=60)
    print(f"   ga_comma: {ga3}")
    print(f"   Survival type: {type(ga3.survival).__name__}")
    assert type(ga3.survival).__name__ == 'CommaSurvival', \
        f"Expected CommaSurvival, got {type(ga3.survival).__name__}"
    
    print("\n✓ GA creation tests passed!")


def test_ga_initialization():
    """Test GA initialization with problem."""
    print("\n" + "="*60)
    print("Testing GA Initialization")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=10,
        xl=-5.0,
        xu=5.0,
    )
    
    ga = GA(
        pop_size=20,
        sampling=UniformSampling(seed=42),
        differentiable=True,
    )
    
    print("\n1. Testing initialization...")
    ga.initialize(problem)
    print(f"   Population shape: {ga.population.shape}")
    print(f"   Fitness shape: {ga.fitness.shape}")
    print(f"   Best fitness: {ga.best_fitness:.4f}")
    print(f"   Best solution shape: {ga.best_solution.shape}")
    
    assert ga.population.shape == (20, 10)
    assert ga.fitness.shape == (20,)
    assert ga.generation == 0
    assert ga.n_evals == 20  # Initial population evaluated
    
    print("\n✓ GA initialization tests passed!")


def test_ga_step():
    """Test GA evolution step."""
    print("\n" + "="*60)
    print("Testing GA Step")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=10,
        xl=-5.0,
        xu=5.0,
    )
    
    ga = GA(
        pop_size=30,
        sampling=UniformSampling(seed=42),
        selection=TournamentSelection(tournament_size=3, differentiable=True),
        crossover=SBXCrossover(eta=15, prob=0.9, differentiable=True),
        mutation=PolynomialMutation(eta=20, differentiable=True),
        survival=MergeSurvival(n_survive=30, elitism=True, n_elite=1, differentiable=True),
        differentiable=True,
    )
    ga.initialize(problem)
    
    print("\n1. Testing single step...")
    initial_best = ga.best_fitness
    ga.step()
    print(f"   Generation: {ga.generation}")
    print(f"   Initial best: {initial_best:.4f}")
    print(f"   After 1 step: {ga.best_fitness:.4f}")
    assert ga.generation == 1
    assert ga.n_evals == 30 + 30  # Initial + one generation
    
    print("\n2. Testing multiple steps...")
    for _ in range(9):
        ga.step()
    print(f"   Generation: {ga.generation}")
    print(f"   Best fitness after 10 steps: {ga.best_fitness:.4f}")
    assert ga.generation == 10
    
    # Check improvement (sphere is easy, should improve)
    print(f"   Improvement: {initial_best - ga.best_fitness:.4f}")
    
    print("\n✓ GA step tests passed!")


def test_survival_strategies():
    """Test different survival selection strategies."""
    print("\n" + "="*60)
    print("Testing Survival Strategies")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    # Test (mu + lambda) - MergeSurvival
    print("\n1. Testing MergeSurvival (mu+lambda)...")
    ga_plus = GA(
        pop_size=20,
        survival=MergeSurvival(n_survive=20, elitism=True, n_elite=1),
        seed=42,
    )
    ga_plus.initialize(problem)
    for _ in range(5):
        ga_plus.step()
    print(f"   Best fitness after 5 gens: {ga_plus.best_fitness:.4f}")
    
    # Test (mu, lambda) - CommaSurvival
    print("\n2. Testing CommaSurvival (mu,lambda)...")
    ga_comma_inst = GA(
        pop_size=20,
        n_offsprings=40,  # Must be >= pop_size
        survival=CommaSurvival(n_survive=20, elitism=True, n_elite=1),
        seed=42,
    )
    ga_comma_inst.initialize(problem)
    for _ in range(5):
        ga_comma_inst.step()
    print(f"   Best fitness after 5 gens: {ga_comma_inst.best_fitness:.4f}")
    
    # Test ReplaceWorstSurvival (steady-state)
    print("\n3. Testing ReplaceWorstSurvival (steady-state)...")
    ga_replace = GA(
        pop_size=20,
        n_offsprings=5,
        survival=ReplaceWorstSurvival(n_survive=20, elitism=True, n_elite=1),
        seed=42,
    )
    ga_replace.initialize(problem)
    for _ in range(20):  # More generations since fewer offspring per gen
        ga_replace.step()
    print(f"   Best fitness after 20 gens: {ga_replace.best_fitness:.4f}")
    
    # Test FitnessSurvival (no elitism)
    print("\n4. Testing FitnessSurvival (pure truncation)...")
    ga_fitness = GA(
        pop_size=20,
        survival=FitnessSurvival(n_survive=20),
        seed=42,
    )
    ga_fitness.initialize(problem)
    for _ in range(5):
        ga_fitness.step()
    print(f"   Best fitness after 5 gens: {ga_fitness.best_fitness:.4f}")
    
    print("\n✓ Survival strategy tests passed!")


def test_elitism():
    """Test elitism behavior."""
    print("\n" + "="*60)
    print("Testing Elitism")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    # GA with elitism
    print("\n1. Testing GA with elitism...")
    ga_elite = GA(
        pop_size=20,
        survival=MergeSurvival(n_survive=20, elitism=True, n_elite=1),
        seed=42,
    )
    ga_elite.initialize(problem)
    
    best_values = [ga_elite.best_fitness]
    for _ in range(10):
        ga_elite.step()
        best_values.append(ga_elite.best_fitness)
    
    # With elitism, best should never increase (for minimization)
    for i in range(1, len(best_values)):
        assert best_values[i] <= best_values[i-1] + 1e-8, \
            f"Elitism violated: {best_values[i]} > {best_values[i-1]}"
    print(f"   Best never increased over {len(best_values)} generations ✓")
    
    # GA without elitism (best can worsen)
    print("\n2. Testing GA without elitism...")
    ga_no_elite = GA(
        pop_size=20,
        survival=FitnessSurvival(n_survive=20),  # No elitism
        seed=42,
    )
    ga_no_elite.initialize(problem)
    
    for _ in range(10):
        ga_no_elite.step()
    print(f"   Final best: {ga_no_elite.best_fitness:.4f}")
    
    print("\n✓ Elitism tests passed!")


def test_differentiable_mode():
    """Test differentiable mode with gradient computation."""
    print("\n" + "="*60)
    print("Testing Differentiable Mode")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    ga = GA(
        pop_size=20,
        differentiable=True,
        seed=42,
    )
    ga.initialize(problem)
    
    # Create optimizer for learnable parameters
    optimizer = torch.optim.Adam(ga.parameters(), lr=0.01)
    
    print("\n1. Testing forward pass...")
    loss = ga.forward()
    print(f"   Loss (best fitness): {loss.item():.4f}")
    assert loss.requires_grad, "Loss should require gradients"
    
    print("\n2. Testing backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # Check that some parameters received gradients
    n_grads = 0
    for name, param in ga.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            n_grads += 1
    print(f"   Parameters with gradients: {n_grads}")
    
    print("\n3. Testing optimizer step...")
    optimizer.step()
    ga.update_state()
    
    print(f"   Generation after update: {ga.generation}")
    
    print("\n4. Testing multiple differentiable iterations...")
    for i in range(5):
        optimizer.zero_grad()
        loss = ga.forward()
        loss.backward()
        optimizer.step()
        ga.update_state()
    print(f"   Final best fitness: {ga.best_fitness:.4f}")
    
    print("\n✓ Differentiable mode tests passed!")


def test_state_persistence():
    """Test state save and load."""
    print("\n" + "="*60)
    print("Testing State Persistence")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    # Create and run GA
    ga1 = GA(
        pop_size=20,
        sampling=UniformSampling(seed=42),
        selection=TournamentSelection(tournament_size=3),
        crossover=SBXCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(eta=20),
        seed=42,
    )
    ga1.initialize(problem)
    
    for _ in range(10):
        ga1.step()
    
    print(f"\n1. Original GA state:")
    print(f"   Generation: {ga1.generation}")
    print(f"   Best fitness: {ga1.best_fitness:.4f}")
    print(f"   N evals: {ga1.n_evals}")
    
    # Save state
    state = ga1.state_dict()
    
    # Create new GA with same structure
    ga2 = GA(
        pop_size=20,
        sampling=UniformSampling(seed=42),
        selection=TournamentSelection(tournament_size=3),
        crossover=SBXCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(eta=20),
        seed=0,  # Different seed
    )
    ga2.initialize(problem)
    
    print(f"\n2. New GA before load:")
    print(f"   Generation: {ga2.generation}")
    print(f"   Best fitness: {ga2.best_fitness:.4f}")
    
    # Load state
    ga2.load_state_dict(state)
    
    print(f"\n3. New GA after load:")
    print(f"   Generation: {ga2.generation}")
    print(f"   Best fitness: {ga2.best_fitness:.4f}")
    
    assert ga2.generation == ga1.generation
    assert abs(ga2.best_fitness - ga1.best_fitness) < 1e-6
    
    # Continue evolution
    for _ in range(5):
        ga2.step()
    print(f"\n4. After 5 more steps:")
    print(f"   Generation: {ga2.generation}")
    print(f"   Best fitness: {ga2.best_fitness:.4f}")
    
    # Test save to file
    print("\n5. Testing save/load to file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "ga_state.pt")
        torch.save(ga1.state_dict(), filepath)
        print(f"   Saved to: {filepath}")
        
        ga3 = GA(
            pop_size=20,
            sampling=UniformSampling(seed=42),
            selection=TournamentSelection(tournament_size=3),
            crossover=SBXCrossover(eta=15, prob=0.9),
            mutation=PolynomialMutation(eta=20),
        )
        ga3.initialize(problem)
        ga3.load_state_dict(torch.load(filepath))
        print(f"   Loaded generation: {ga3.generation}")
        assert ga3.generation == ga1.generation
    
    print("\n✓ State persistence tests passed!")


def test_convergence():
    """Test GA convergence on simple function."""
    print("\n" + "="*60)
    print("Testing Convergence")
    print("="*60)
    
    # Easy problem: 5D sphere
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    ga = GA(
        pop_size=50,
        selection=TournamentSelection(tournament_size=3),
        crossover=SBXCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(eta=20),
        survival=MergeSurvival(n_survive=50, elitism=True, n_elite=1),
        seed=42,
    )
    ga.initialize(problem)
    
    print(f"\n1. Running GA for 100 generations...")
    print(f"   Initial best: {ga.best_fitness:.4f}")
    
    for gen in range(100):
        ga.step()
        if (gen + 1) % 25 == 0:
            print(f"   Gen {gen+1:3d}: best = {ga.best_fitness:.6f}")
    
    print(f"\n2. Final results:")
    print(f"   Best fitness: {ga.best_fitness:.6f}")
    print(f"   Best solution: {ga.best_solution.tolist()}")
    print(f"   Distance to origin: {ga.best_solution.norm().item():.6f}")
    
    # Should get reasonably close to optimum (0)
    assert ga.best_fitness < 1.0, f"GA should converge better, got {ga.best_fitness}"
    print("\n✓ Convergence tests passed!")


def test_hyperparams():
    """Test hyperparameter tracking."""
    print("\n" + "="*60)
    print("Testing Hyperparameter Tracking")
    print("="*60)
    
    problem = Problem(
        objective=sphere,
        n_var=5,
        xl=-5.0,
        xu=5.0,
    )
    
    ga = GA(
        pop_size=20,
        selection=TournamentSelection(tournament_size=3, differentiable=True),
        crossover=SBXCrossover(eta=15, prob=0.9, differentiable=True),
        mutation=PolynomialMutation(eta=20, differentiable=True),
        survival=MergeSurvival(n_survive=20, elitism=True, n_elite=2, differentiable=True),
    )
    ga.initialize(problem)
    ga.step()
    
    params = ga._get_hyperparams()
    print(f"\n1. Hyperparameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    assert 'pop_size' in params
    assert params['pop_size'] == 20
    # Elitism is now tracked via the survival operator
    assert 'elitism' in params
    assert params['elitism'] == True
    
    print("\n✓ Hyperparameter tracking tests passed!")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all GA tests."""
    print("\n" + "#"*60)
    print("# EvoGrad GA Algorithm Tests")
    print("#"*60)
    
    try:
        test_ga_creation()
        test_ga_initialization()
        test_ga_step()
        test_survival_strategies()
        test_elitism()
        test_differentiable_mode()
        test_state_persistence()
        test_convergence()
        test_hyperparams()
        
        print("\n" + "="*60)
        print("✓ ALL GA TESTS PASSED!")
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
