"""
Test script for EvoGrad core module.

Tests:
    - problem.py: Problem definition and evaluation
    - termination.py: Termination criteria
    - result.py: Result container
    - algorithm.py: Algorithm base class

Usage:
    cd evograd && python tests/test_core.py
"""

import sys
import torch
import torch.nn as nn
import tempfile
import os

# Add parent directory to path for imports (works when running from evograd/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import Problem
from core.termination import (
    Termination,
    MaxEvaluations,
    MaxGenerations,
    TargetReached,
    ToleranceReached,
    TimeLimit,
    TerminationCollection,
)
from core.result import Result
from core.algorithm import Algorithm, AlgorithmState


def test_problem():
    """Test Problem class."""
    print("\n" + "="*60)
    print("Testing problem.py")
    print("="*60)
    
    # Test basic Problem creation
    print("\n1. Testing Problem creation...")
    
    def sphere_func(x):
        return (x ** 2).sum(dim=-1)
    
    problem = Problem(
        n_var=10,
        n_obj=1,
        xl=-5.0,
        xu=5.0,
        objective=sphere_func,
    )
    
    print(f"   n_var: {problem.n_var}")
    print(f"   n_obj: {problem.n_obj}")
    print(f"   xl shape: {problem.xl.shape}")
    print(f"   xu shape: {problem.xu.shape}")
    assert problem.n_var == 10
    assert problem.xl.shape == (10,)
    
    # Test evaluation
    print("\n2. Testing Problem evaluation...")
    x = torch.randn(20, 10)  # 20 solutions, 10 variables
    f = problem.evaluate(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {f.shape}")
    print(f"   Sample fitness values: {f[:3].tolist()}")
    assert f.shape == (20,)
    
    # Test bounds as lists
    print("\n3. Testing Problem with list bounds...")
    problem2 = Problem(
        n_var=3,
        xl=[-1.0, -2.0, -3.0],
        xu=[1.0, 2.0, 3.0],
        objective=sphere_func,
    )
    print(f"   xl: {problem2.xl.tolist()}")
    print(f"   xu: {problem2.xu.tolist()}")
    assert problem2.xl[1] == -2.0
    assert problem2.xu[2] == 3.0
    
    # Test Problem with different objective function
    print("\n4. Testing Problem with Rastrigin function...")
    
    def rastrigin(x):
        A = 10.0
        return A * x.shape[-1] + (x**2 - A * torch.cos(2 * 3.14159 * x)).sum(dim=-1)
    
    rastrigin_problem = Problem(
        objective=rastrigin,
        n_var=5,
        xl=-5.12,
        xu=5.12,
    )
    
    x = torch.zeros(1, 5)  # Global optimum
    f = rastrigin_problem(x)
    print(f"   Rastrigin at origin: {f.item():.6f}")
    assert abs(f.item()) < 1e-5, "Rastrigin should be 0 at origin"
    
    # Test constraints
    print("\n5. Testing Problem with constraints...")
    
    def constraint_func(x):
        # g(x) <= 0 format: x[0] + x[1] <= 1
        return x[:, 0] + x[:, 1] - 1.0
    
    constrained_problem = Problem(
        n_var=2,
        xl=0.0,
        xu=2.0,
        objective=sphere_func,
        constraints=[(constraint_func, 'ineq')],
    )
    
    # Feasible point
    x_feasible = torch.tensor([[0.3, 0.3]])
    g = constrained_problem.evaluate_constraints(x_feasible)
    ineq_val = g['ineq'][0, 0].item()
    print(f"   Constraint at [0.3, 0.3]: {ineq_val:.2f} (should be < 0)")
    assert ineq_val < 0, "Point should be feasible"
    
    # Infeasible point
    x_infeasible = torch.tensor([[0.8, 0.8]])
    g = constrained_problem.evaluate_constraints(x_infeasible)
    ineq_val = g['ineq'][0, 0].item()
    print(f"   Constraint at [0.8, 0.8]: {ineq_val:.2f} (should be > 0)")
    assert ineq_val > 0, "Point should be infeasible"
    
    # Test is_feasible
    feasibility = constrained_problem.is_feasible(
        torch.cat([x_feasible, x_infeasible])
    )
    print(f"   Feasibility: {feasibility.tolist()}")
    assert feasibility[0] == True
    assert feasibility[1] == False
    
    print("\n✓ problem.py tests passed!")


def test_termination():
    """Test termination criteria."""
    print("\n" + "="*60)
    print("Testing termination.py")
    print("="*60)
    
    # Create mock algorithm state
    class MockAlgorithm:
        def __init__(self):
            self.n_evals = 0
            self.generation = 0
            self.best_fitness = float('inf')
            self.fitness = torch.tensor([100.0])
            self._prev_best = None
    
    # Test MaxEvaluations
    print("\n1. Testing MaxEvaluations...")
    max_evals = MaxEvaluations(max_evals=100)  # keyword arg as in actual API
    alg = MockAlgorithm()
    
    alg.n_evals = 50
    assert not max_evals.should_terminate(alg), "Should not terminate at 50 evals"
    print(f"   Progress at 50 evals: {max_evals.progress(alg):.1%}")
    
    alg.n_evals = 100
    assert max_evals.should_terminate(alg), "Should terminate at 100 evals"
    print(f"   Terminated at 100 evals: ✓")
    
    # Test MaxGenerations
    print("\n2. Testing MaxGenerations...")
    max_gen = MaxGenerations(max_gens=50)  # keyword arg as in actual API
    alg = MockAlgorithm()
    
    alg.generation = 25
    assert not max_gen.should_terminate(alg)
    print(f"   Progress at gen 25: {max_gen.progress(alg):.1%}")
    
    alg.generation = 50
    assert max_gen.should_terminate(alg)
    print(f"   Terminated at gen 50: ✓")
    
    # Test TargetReached (no tolerance param - just target_fitness and minimize)
    print("\n3. Testing TargetReached...")
    target = TargetReached(target_fitness=1.0, minimize=True)
    alg = MockAlgorithm()
    
    alg.best_fitness = 10.0
    assert not target.should_terminate(alg)
    print(f"   At fitness 10.0: not terminated")
    
    alg.best_fitness = 0.5  # Below target (for minimization)
    assert target.should_terminate(alg)
    print(f"   At fitness 0.5 (below target): terminated ✓")
    
    # Test ToleranceReached
    print("\n4. Testing ToleranceReached...")
    tol = ToleranceReached(tol=0.001, n_last=3, mode='absolute')
    alg = MockAlgorithm()
    
    # Simulate improving then stagnating
    fitness_history = [100.0, 50.0, 25.0, 24.9999, 24.9998, 24.9997]
    terminated_at = None
    
    for gen, fit in enumerate(fitness_history):
        alg.generation = gen
        alg.best_fitness = fit
        if tol.should_terminate(alg):
            terminated_at = gen
            break
    
    print(f"   Fitness history: {fitness_history}")
    print(f"   Terminated at generation: {terminated_at}")
    assert terminated_at is not None, "Should have terminated due to stagnation"
    
    # Test TimeLimit
    print("\n5. Testing TimeLimit...")
    time_limit = TimeLimit(max_seconds=0.1)  # 100ms, keyword arg as in actual API
    alg = MockAlgorithm()
    
    assert not time_limit.should_terminate(alg)
    print(f"   Initial progress: {time_limit.progress(alg):.1%}")
    
    import time
    time.sleep(0.15)  # Wait 150ms
    assert time_limit.should_terminate(alg)
    print(f"   After 150ms: terminated ✓")
    
    # Test TerminationCollection (any)
    print("\n6. Testing TerminationCollection (mode='or')...")
    combined = TerminationCollection(
        criteria=[
            MaxEvaluations(1000),
            MaxGenerations(100),
            TargetReached(0.0),
        ],
        mode='or',
    )
    
    alg = MockAlgorithm()
    alg.n_evals = 50
    alg.generation = 10
    alg.best_fitness = 0.0  # Target reached!
    
    assert combined.should_terminate(alg)
    print(f"   Terminated because target reached (any mode): ✓")
    
    # Test TerminationCollection (all)
    print("\n7. Testing TerminationCollection (mode='and')...")
    combined_all = TerminationCollection(
        criteria=[
            MaxEvaluations(100),
            MaxGenerations(10),
        ],
        mode='and',
    )
    
    alg = MockAlgorithm()
    alg.n_evals = 100  # Met
    alg.generation = 5  # Not met
    assert not combined_all.should_terminate(alg)
    print(f"   100 evals, 5 gens: not terminated (all mode)")
    
    alg.generation = 10  # Both met
    assert combined_all.should_terminate(alg)
    print(f"   100 evals, 10 gens: terminated (all mode) ✓")
    
    print("\n✓ termination.py tests passed!")


def test_result():
    """Test Result class."""
    print("\n" + "="*60)
    print("Testing result.py")
    print("="*60)
    
    # Create a result
    print("\n1. Testing Result creation...")
    result = Result(
        best_solution=torch.tensor([1.0, 2.0, 3.0]),
        best_fitness=0.5,
        n_evals=1000,
        n_gen=50,
        elapsed_time=10.5,
        success=True,
        termination_reason="Optimization completed",
    )
    
    print(f"   Best solution: {result.best_solution.tolist()}")
    print(f"   Best fitness: {result.best_fitness}")
    print(f"   Evaluations: {result.n_evals}")
    print(f"   Generations: {result.n_gen}")
    print(f"   Time: {result.elapsed_time:.1f}s")
    
    # Test with history
    print("\n2. Testing Result with history...")
    result.history = {
        'best_fitness': [100, 50, 25, 10, 5, 1, 0.5],
        'generation': list(range(7)),
    }
    
    print(f"   History keys: {list(result.history.keys())}")
    print(f"   Best fitness history: {result.history['best_fitness']}")
    
    # Test population storage
    print("\n3. Testing Result with population...")
    result.population = torch.randn(20, 3)
    result.fitness = torch.randn(20)
    print(f"   Population shape: {result.population.shape}")
    print(f"   Fitness shape: {result.fitness.shape}")
    
    # Test save and load
    print("\n4. Testing save and load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "result.pt")
        result.save(filepath)
        print(f"   Saved to: {filepath}")
        
        loaded = Result.load(filepath)
        print(f"   Loaded best_solution: {loaded.best_solution.tolist()}")
        print(f"   Loaded best_fitness: {loaded.best_fitness}")
        print(f"   Loaded population shape: {loaded.population.shape}")
        print(f"   Loaded fitness shape: {loaded.fitness.shape}")
        assert torch.allclose(result.best_solution, loaded.best_solution)
        assert result.best_fitness == loaded.best_fitness
        print("   Save/load verified ✓")
    
    # Test string representation
    print("\n5. Testing string representation...")
    print(result)
    
    # Test to_dict
    print("\n6. Testing to_dict...")
    result_dict = result.to_dict()
    print(f"   Dict keys: {list(result_dict.keys())}")
    assert 'best_solution' in result_dict
    assert 'best_fitness' in result_dict
    assert 'population' in result_dict
    assert 'fitness' in result_dict
    
    print("\n✓ result.py tests passed!")


def test_algorithm():
    """Test Algorithm base class."""
    print("\n" + "="*60)
    print("Testing algorithm.py")
    print("="*60)
    
    # Import operators for a concrete implementation
    from operators.sampling import UniformSampling
    from operators.selection import TournamentSelection
    from operators.crossover import SBXCrossover
    from operators.mutation import PolynomialMutation
    from operators.repair import ReflectRepair
    
    # Create a simple concrete algorithm
    print("\n1. Creating concrete Algorithm subclass...")
    
    class SimpleGA(Algorithm):
        """Minimal GA for testing."""
        
        def __init__(self, pop_size=20, **kwargs):
            super().__init__(pop_size=pop_size, **kwargs)
        
        def _setup(self):
            """Initialise population (called by initialize)."""
            # Population is already created by parent class
            pass
        
        def _infill(self):
            """Generate offspring."""
            # Select parents
            parents = self.selection(
                self.population, self.fitness, self.pop_size
            )
            
            # Crossover (pair consecutive)
            n_pairs = self.pop_size // 2
            p1 = parents[:n_pairs]
            p2 = parents[n_pairs:2*n_pairs]
            offspring = self.crossover(p1, p2)
            
            # Mutation
            offspring = self.mutation(offspring, self.problem.xl, self.problem.xu)
            
            # Repair bounds
            offspring = self.repair(offspring, self.problem.xl, self.problem.xu)
            
            return offspring
        
        def _advance(self, offspring, offspring_fitness):
            """Update population."""
            # Combine and select best
            combined_pop = torch.cat([self.population, offspring], dim=0)
            combined_fit = torch.cat([self.fitness, offspring_fitness], dim=0)
            
            # Select top pop_size
            indices = torch.argsort(combined_fit)[:self.pop_size]
            
            # Update population (stored as nn.Parameter)
            with torch.no_grad():
                self._population.copy_(combined_pop[indices])
            self.state.fitness = combined_fit[indices]
            
            # Update best
            self.state.update_best(self.population, self.state.fitness)
    
    # Create problem
    def sphere(x):
        return (x ** 2).sum(dim=-1)
    
    problem = Problem(
        n_var=5,
        xl=-5.0,
        xu=5.0,
        objective=sphere,
    )
    
    # Create algorithm with operators
    print("\n2. Testing Algorithm with dependency injection...")
    ga = SimpleGA(
        pop_size=20,
        sampling=UniformSampling(seed=42),
        selection=TournamentSelection(tournament_size=3),
        crossover=SBXCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(eta=20),
        repair=ReflectRepair(),
    )
    
    print(f"   Algorithm: {ga}")
    print(f"   Pop size: {ga.pop_size}")
    
    # Test initialize (not setup)
    print("\n3. Testing initialize...")
    ga.initialize(problem)
    print(f"   Population shape: {ga.population.shape}")
    print(f"   Initial best fitness: {ga.best_fitness:.4f}")
    
    # Test step
    print("\n4. Testing step() method...")
    initial_best = ga.best_fitness
    for i in range(10):
        ga.step()
    print(f"   Generation: {ga.generation}")
    print(f"   Best fitness after 10 steps: {ga.best_fitness:.4f}")
    assert ga.generation == 10
    
    # Test state dict
    print("\n5. Testing state_dict and load_state_dict...")
    state = ga.state_dict()
    print(f"   State keys: {list(state.keys())}")
    
    # Create new algorithm with SAME operators and load state
    ga2 = SimpleGA(
        pop_size=20,
        sampling=UniformSampling(seed=42),
        selection=TournamentSelection(tournament_size=3),
        crossover=SBXCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(eta=20),
        repair=ReflectRepair(),
    )
    ga2.initialize(problem)
    ga2.load_state_dict(state)
    print(f"   Loaded generation: {ga2.generation}")
    print(f"   Loaded best_fitness: {ga2.best_fitness:.4f}")
    assert ga2.generation == ga.generation
    assert abs(ga2.best_fitness - ga.best_fitness) < 1e-6
    
    # Test save and load using torch
    print("\n6. Testing save and load with torch...")
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "algorithm.pt")
        torch.save(ga.state_dict(), filepath)
        print(f"   Saved to: {filepath}")
        
        ga3 = SimpleGA(
            pop_size=20,
            sampling=UniformSampling(seed=42),
            selection=TournamentSelection(tournament_size=3),
            crossover=SBXCrossover(eta=15, prob=0.9),
            mutation=PolynomialMutation(eta=20),
            repair=ReflectRepair(),
        )
        ga3.initialize(problem)
        ga3.load_state_dict(torch.load(filepath))
        print(f"   Loaded generation: {ga3.generation}")
        assert ga3.generation == ga.generation
    
    # Test AlgorithmState container
    print("\n7. Testing AlgorithmState container...")
    print(f"   state.generation: {ga.state.generation}")
    print(f"   state.n_evals: {ga.state.n_evals}")
    print(f"   state.best_fitness: {ga.state.best_fitness:.4f}")
    assert ga.state.generation == ga.generation
    
    print("\n✓ algorithm.py tests passed!")


def run_all_tests():
    """Run all core tests."""
    print("\n" + "#"*60)
    print("# EvoGrad Core Module Tests")
    print("#"*60)
    
    try:
        test_problem()
        test_termination()
        test_result()
        test_algorithm()
        
        print("\n" + "="*60)
        print("✓ ALL CORE TESTS PASSED!")
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
