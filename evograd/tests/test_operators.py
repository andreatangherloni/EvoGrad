"""
Test script for EvoGrad operators module.

Tests:
    - sampling.py: Population initialisation
    - selection.py: Parent selection
    - crossover.py: Recombination operators
    - mutation.py: Mutation operators
    - repair.py: Bounds handling
"""

import sys
import torch
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import Problem

from operators.sampling import (
    UniformSampling,
    LatinHypercubeSampling,
    NormalSampling,
    LogUniformSampling,
    HaltonSampling,
)

from operators.selection import (
    TournamentSelection,
    RouletteSelection,
    RankSelection,
    RandomSelection,
    TruncationSelection,
    StochasticUniversalSampling,
)

from operators.crossover import (
    SBXCrossover,
    BlendCrossover,
    BinomialCrossover,
    ExponentialCrossover,
    UniformCrossover,
    ArithmeticCrossover,
    NPointCrossover,
)

from operators.mutation import (
    PolynomialMutation,
    GaussianMutation,
    UniformMutation,
    NonUniformMutation,
    BoundaryMutation,
    NoMutation,
    CombinedMutation,
)

from operators.repair import (
    ClipRepair,
    ReflectRepair,
    WrapRepair,
    RandomRepair,
    BoundsRepair,
    SoftClipRepair,
    PenaltyRepair,
    NoRepair,
)


# Helper to create a test problem
def create_test_problem(n_var=10):
    def sphere(x):
        return (x ** 2).sum(dim=-1)
    return Problem(n_var=n_var, xl=-5.0, xu=5.0, objective=sphere)


def test_sampling():
    """Test sampling operators."""
    print("\n" + "="*60)
    print("Testing sampling.py")
    print("="*60)
    
    problem = create_test_problem(n_var=5)
    n_samples = 50
    
    # Test UniformSampling
    print("\n1. Testing UniformSampling...")
    sampler = UniformSampling(seed=42)
    pop = sampler(n_samples, problem)
    print(f"   Shape: {pop.shape}")
    assert pop.shape == (n_samples, 5)
    assert (pop >= problem.xl).all() and (pop <= problem.xu).all()
    print(f"   All within bounds: ✓")
    
    # Test reproducibility
    sampler2 = UniformSampling(seed=42)
    pop2 = sampler2(n_samples, problem)
    assert torch.allclose(pop, pop2)
    print(f"   Reproducibility with seed: ✓")
    
    # Test LatinHypercubeSampling
    print("\n2. Testing LatinHypercubeSampling...")
    lhs = LatinHypercubeSampling(smooth=True, seed=42)
    pop_lhs = lhs(n_samples, problem)
    print(f"   Shape: {pop_lhs.shape}")
    assert pop_lhs.shape == (n_samples, 5)
    assert (pop_lhs >= problem.xl).all() and (pop_lhs <= problem.xu).all()
    print(f"   All within bounds: ✓")
    
    # Test NormalSampling
    print("\n3. Testing NormalSampling...")
    normal = NormalSampling(sigma_factor=0.2, seed=42)
    pop_normal = normal(n_samples, problem)
    print(f"   Shape: {pop_normal.shape}")
    # Most should be within bounds (3-sigma)
    within = ((pop_normal >= problem.xl) & (pop_normal <= problem.xu)).float().mean()
    print(f"   Fraction within bounds: {within:.1%}")
    
    # Test LogUniformSampling
    print("\n4. Testing LogUniformSampling...")
    # Use positive bounds for log sampling
    log_problem = Problem(n_var=5, xl=0.001, xu=1000.0, objective=lambda x: x.sum(-1))
    log_sampler = LogUniformSampling(base=10, seed=42)
    pop_log = log_sampler(n_samples, log_problem)
    print(f"   Shape: {pop_log.shape}")
    print(f"   Min: {pop_log.min().item():.4f}, Max: {pop_log.max().item():.4f}")
    assert (pop_log > 0).all()
    
    # Test HaltonSampling
    print("\n5. Testing HaltonSampling...")
    halton = HaltonSampling(scramble=True, seed=42)
    pop_halton = halton(n_samples, problem)
    print(f"   Shape: {pop_halton.shape}")
    assert pop_halton.shape == (n_samples, 5)
    assert (pop_halton >= problem.xl).all() and (pop_halton <= problem.xu).all()
    print(f"   All within bounds: ✓")
    
    print("\n✓ sampling.py tests passed!")


def test_selection():
    """Test selection operators."""
    print("\n" + "="*60)
    print("Testing selection.py")
    print("="*60)
    
    # Create test population and fitness
    n_pop = 50
    n_var = 5
    population = torch.randn(n_pop, n_var)
    fitness = torch.randn(n_pop)  # Lower is better
    
    n_select = 30
    
    # Test TournamentSelection
    print("\n1. Testing TournamentSelection...")
    tournament = TournamentSelection(tournament_size=3, replacement=True)
    selected = tournament(population, fitness, n_select)
    print(f"   Selected shape: {selected.shape}")
    assert selected.shape == (n_select, n_var)
    
    # Test with indices
    selected, indices = tournament(population, fitness, n_select, return_indices=True)
    print(f"   Indices shape: {indices.shape}")
    assert indices.shape == (n_select,)
    assert (indices >= 0).all() and (indices < n_pop).all()
    print(f"   Index range valid: ✓")
    
    # Test differentiable mode
    print("\n2. Testing TournamentSelection (differentiable)...")
    tournament_diff = TournamentSelection(
        tournament_size=3,
        adaptive=True,
        temperature=1.0,
        learn_temperature=True,
    )
    pop_param = torch.nn.Parameter(population.clone())
    selected_diff = tournament_diff(pop_param, fitness, n_select)
    print(f"   Differentiable selection shape: {selected_diff.shape}")
    
    # Check gradients flow
    loss = selected_diff.sum()
    loss.backward()
    assert pop_param.grad is not None
    print(f"   Gradients flow: ✓")
    
    # Test RouletteSelection
    print("\n3. Testing RouletteSelection...")
    roulette = RouletteSelection()
    selected_roulette = roulette(population, fitness, n_select)
    print(f"   Selected shape: {selected_roulette.shape}")
    assert selected_roulette.shape == (n_select, n_var)
    
    # Test RankSelection
    print("\n4. Testing RankSelection...")
    rank_sel = RankSelection(scheme='linear', selection_pressure=1.5)
    selected_rank = rank_sel(population, fitness, n_select)
    print(f"   Selected shape: {selected_rank.shape}")
    
    rank_exp = RankSelection(scheme='exponential')
    selected_rank_exp = rank_exp(population, fitness, n_select)
    print(f"   Exponential scheme works: ✓")
    
    # Test RandomSelection
    print("\n5. Testing RandomSelection...")
    random_sel = RandomSelection(replacement=True)
    selected_random = random_sel(population, fitness, n_select)
    print(f"   Selected shape: {selected_random.shape}")
    
    # Test TruncationSelection
    print("\n6. Testing TruncationSelection...")
    truncation = TruncationSelection(truncation_ratio=0.5)
    selected_trunc = truncation(population, fitness, n_select)
    print(f"   Selected shape: {selected_trunc.shape}")
    
    # Test StochasticUniversalSampling
    print("\n7. Testing StochasticUniversalSampling...")
    sus = StochasticUniversalSampling()
    selected_sus = sus(population, fitness, n_select)
    print(f"   Selected shape: {selected_sus.shape}")
    
    print("\n✓ selection.py tests passed!")


def test_crossover():
    """Test crossover operators."""
    print("\n" + "="*60)
    print("Testing crossover.py")
    print("="*60)
    
    n_pairs = 25
    n_var = 10
    
    # Create parent pairs
    parent1 = torch.randn(n_pairs, n_var)
    parent2 = torch.randn(n_pairs, n_var)
    
    # Test SBXCrossover
    print("\n1. Testing SBXCrossover...")
    sbx = SBXCrossover(eta=15, prob=0.9)
    offspring = sbx(parent1, parent2)
    print(f"   Offspring shape: {offspring.shape}")
    assert offspring.shape == (n_pairs, n_var)
    
    # Test differentiable mode
    print("\n2. Testing SBXCrossover (differentiable)...")
    sbx_diff = SBXCrossover(
        eta=15,
        prob=0.9,
        adaptive=True,
        learn_eta=True,
        learn_prob=True,
    )
    p1 = torch.nn.Parameter(parent1.clone())
    p2 = torch.nn.Parameter(parent2.clone())
    offspring_diff = sbx_diff(p1, p2)
    
    loss = offspring_diff.sum()
    loss.backward()
    assert p1.grad is not None
    print(f"   Gradients flow through SBX: ✓")
    print(f"   Learnable eta: {sbx_diff._log_eta.item():.4f}")
    
    # Test BlendCrossover
    print("\n3. Testing BlendCrossover...")
    blend = BlendCrossover(alpha=0.5)
    offspring_blend = blend(parent1, parent2)
    print(f"   Offspring shape: {offspring_blend.shape}")
    
    # Test BinomialCrossover (DE-style)
    print("\n4. Testing BinomialCrossover...")
    binomial = BinomialCrossover(cr=0.9)
    offspring_bin = binomial(parent1, parent2)  # parent1=target, parent2=donor
    print(f"   Offspring shape: {offspring_bin.shape}")
    
    # Test differentiable binomial
    binomial_diff = BinomialCrossover(cr=0.9, adaptive=True, learn_cr=True)
    p1 = torch.nn.Parameter(parent1.clone())
    offspring_bin_diff = binomial_diff(p1, parent2)
    loss = offspring_bin_diff.sum()
    loss.backward()
    assert p1.grad is not None
    print(f"   Gradients flow through binomial: ✓")
    
    # Test ExponentialCrossover
    print("\n5. Testing ExponentialCrossover...")
    exponential = ExponentialCrossover(cr=0.9)
    offspring_exp = exponential(parent1, parent2)
    print(f"   Offspring shape: {offspring_exp.shape}")
    
    # Test UniformCrossover
    print("\n6. Testing UniformCrossover...")
    uniform = UniformCrossover(prob=0.9)
    offspring_unif = uniform(parent1, parent2)
    print(f"   Offspring shape: {offspring_unif.shape}")
    
    # Test ArithmeticCrossover
    print("\n7. Testing ArithmeticCrossover...")
    arithmetic = ArithmeticCrossover(alpha=0.5, whole=True)
    offspring_arith = arithmetic(parent1, parent2)
    print(f"   Offspring shape: {offspring_arith.shape}")
    
    # Verify whole arithmetic is weighted average
    expected = 0.5 * parent1 + 0.5 * parent2
    assert torch.allclose(offspring_arith, expected)
    print(f"   Weighted average verified: ✓")
    
    # Test NPointCrossover
    print("\n8. Testing NPointCrossover...")
    npoint = NPointCrossover(n_points=2)
    offspring_npoint = npoint(parent1, parent2)
    print(f"   Offspring shape: {offspring_npoint.shape}")
    
    print("\n✓ crossover.py tests passed!")


def test_mutation():
    """Test mutation operators."""
    print("\n" + "="*60)
    print("Testing mutation.py")
    print("="*60)
    
    n_pop = 50
    n_var = 10
    
    population = torch.randn(n_pop, n_var)
    xl = torch.full((n_var,), -5.0)
    xu = torch.full((n_var,), 5.0)
    
    # Test PolynomialMutation
    print("\n1. Testing PolynomialMutation...")
    poly_mut = PolynomialMutation(eta=20, prob=0.1)
    mutated = poly_mut(population, xl, xu)
    print(f"   Mutated shape: {mutated.shape}")
    assert mutated.shape == population.shape
    
    # Check some genes changed
    changed = (mutated != population).any(dim=0).sum()
    print(f"   Genes with changes: {changed.item()}/{n_var}")
    
    # Test differentiable mode
    print("\n2. Testing PolynomialMutation (differentiable)...")
    poly_diff = PolynomialMutation(
        eta=20,
        prob=0.1,
        adaptive=True,
        learn_eta=True,
        learn_prob=True,
    )
    pop_param = torch.nn.Parameter(population.clone())
    mutated_diff = poly_diff(pop_param, xl, xu)
    
    loss = mutated_diff.sum()
    loss.backward()
    assert pop_param.grad is not None
    print(f"   Gradients flow through polynomial mutation: ✓")
    
    # Test GaussianMutation
    print("\n3. Testing GaussianMutation...")
    gauss_mut = GaussianMutation(sigma=0.1, prob=0.2)
    mutated_gauss = gauss_mut(population, xl, xu)
    print(f"   Mutated shape: {mutated_gauss.shape}")
    
    # Test with sigma_frac
    gauss_frac = GaussianMutation(sigma_frac=0.05)
    mutated_frac = gauss_frac(population, xl, xu)
    print(f"   With sigma_frac: ✓")
    
    # Test differentiable Gaussian
    gauss_diff = GaussianMutation(sigma=0.1, adaptive=True, learn_sigma=True)
    pop_param = torch.nn.Parameter(population.clone())
    mutated_gauss_diff = gauss_diff(pop_param, xl, xu)
    loss = mutated_gauss_diff.sum()
    loss.backward()
    assert pop_param.grad is not None
    print(f"   Gradients flow through Gaussian mutation: ✓")
    
    # Test UniformMutation
    print("\n4. Testing UniformMutation...")
    unif_mut = UniformMutation(prob=0.1)
    mutated_unif = unif_mut(population, xl, xu)
    print(f"   Mutated shape: {mutated_unif.shape}")
    
    # Test NonUniformMutation
    print("\n5. Testing NonUniformMutation...")
    nonunif_mut = NonUniformMutation(max_generations=100, b=5.0)
    nonunif_mut.set_generation(50)
    mutated_nonunif = nonunif_mut(population, xl, xu)
    print(f"   Mutated shape: {mutated_nonunif.shape}")
    print(f"   Current generation: {nonunif_mut.generation}")
    
    # Test BoundaryMutation
    print("\n6. Testing BoundaryMutation...")
    boundary_mut = BoundaryMutation(prob=0.1)
    mutated_boundary = boundary_mut(population, xl, xu)
    print(f"   Mutated shape: {mutated_boundary.shape}")
    
    # Check boundary values
    at_lower = (mutated_boundary == xl).any()
    at_upper = (mutated_boundary == xu).any()
    print(f"   Has values at lower bound: {at_lower}")
    print(f"   Has values at upper bound: {at_upper}")
    
    # Test NoMutation
    print("\n7. Testing NoMutation...")
    no_mut = NoMutation()
    mutated_none = no_mut(population, xl, xu)
    assert torch.allclose(mutated_none, population)
    print(f"   NoMutation returns unchanged: ✓")
    
    # Test CombinedMutation
    print("\n8. Testing CombinedMutation...")
    combined = CombinedMutation([
        GaussianMutation(sigma=0.05, prob=0.5),
        PolynomialMutation(eta=20, prob=0.1),
    ])
    mutated_combined = combined(population, xl, xu)
    print(f"   Combined mutation shape: {mutated_combined.shape}")
    
    print("\n✓ mutation.py tests passed!")


def test_repair():
    """Test repair operators."""
    print("\n" + "="*60)
    print("Testing repair.py")
    print("="*60)
    
    n_pop = 50
    n_var = 5
    
    xl = torch.zeros(n_var)
    xu = torch.ones(n_var)
    
    # Create population with violations
    population = torch.randn(n_pop, n_var) * 2  # Some will be outside [0, 1]
    
    n_violations_before = ((population < xl) | (population > xu)).sum().item()
    print(f"\n   Violations before repair: {n_violations_before}")
    
    # Test ClipRepair
    print("\n1. Testing ClipRepair...")
    clip = ClipRepair()
    repaired_clip = clip(population, xl, xu)
    violations_clip = ((repaired_clip < xl) | (repaired_clip > xu)).sum().item()
    print(f"   Violations after clip: {violations_clip}")
    assert violations_clip == 0
    assert (repaired_clip >= xl).all() and (repaired_clip <= xu).all()
    print(f"   All within bounds: ✓")
    
    # Test ReflectRepair
    print("\n2. Testing ReflectRepair...")
    reflect = ReflectRepair()
    repaired_reflect = reflect(population, xl, xu)
    violations_reflect = ((repaired_reflect < xl) | (repaired_reflect > xu)).sum().item()
    print(f"   Violations after reflect: {violations_reflect}")
    assert violations_reflect == 0
    print(f"   All within bounds: ✓")
    
    # Test that reflection preserves "momentum"
    x_test = torch.tensor([[1.3]])  # 0.3 above upper bound of 1
    xl_test = torch.tensor([0.0])
    xu_test = torch.tensor([1.0])
    reflected = reflect(x_test, xl_test, xu_test)
    print(f"   1.3 reflects to: {reflected.item():.2f} (expected ~0.7)")
    assert abs(reflected.item() - 0.7) < 0.01
    
    # Test WrapRepair
    print("\n3. Testing WrapRepair...")
    wrap = WrapRepair()
    repaired_wrap = wrap(population, xl, xu)
    violations_wrap = ((repaired_wrap < xl) | (repaired_wrap > xu)).sum().item()
    print(f"   Violations after wrap: {violations_wrap}")
    assert violations_wrap == 0
    
    # Test wrapping behaviour
    x_test = torch.tensor([[1.3]])
    wrapped = wrap(x_test, xl_test, xu_test)
    print(f"   1.3 wraps to: {wrapped.item():.2f} (expected ~0.3)")
    assert abs(wrapped.item() - 0.3) < 0.01
    
    # Test RandomRepair
    print("\n4. Testing RandomRepair...")
    random_repair = RandomRepair()
    repaired_random = random_repair(population, xl, xu)
    violations_random = ((repaired_random < xl) | (repaired_random > xu)).sum().item()
    print(f"   Violations after random: {violations_random}")
    assert violations_random == 0
    
    # Test BoundsRepair (configurable)
    print("\n5. Testing BoundsRepair...")
    for method in ['clip', 'reflect', 'wrap', 'random']:
        bounds_repair = BoundsRepair(method=method)
        repaired = bounds_repair(population, xl, xu)
        violations = ((repaired < xl) | (repaired > xu)).sum().item()
        assert violations == 0
        print(f"   Method '{method}': ✓")
    
    # Test SoftClipRepair
    print("\n6. Testing SoftClipRepair...")
    soft_clip = SoftClipRepair(beta=10.0)
    repaired_soft = soft_clip(population, xl, xu)
    print(f"   Soft clip shape: {repaired_soft.shape}")
    
    # Check gradients flow
    pop_param = torch.nn.Parameter(population.clone())
    soft_repaired = soft_clip(pop_param, xl, xu)
    loss = soft_repaired.sum()
    loss.backward()
    assert pop_param.grad is not None
    print(f"   Gradients flow through soft clip: ✓")
    
    # Test PenaltyRepair
    print("\n7. Testing PenaltyRepair...")
    penalty_repair = PenaltyRepair(penalty_weight=100.0, power=2.0)
    
    # Compute penalty
    penalty = penalty_repair.compute_penalty(population, xl, xu)
    print(f"   Penalty shape: {penalty.shape}")
    print(f"   Mean penalty: {penalty.mean().item():.2f}")
    
    # Verify no repair happens
    repaired_penalty = penalty_repair(population, xl, xu)
    assert torch.allclose(repaired_penalty, population)
    print(f"   PenaltyRepair returns unchanged: ✓")
    
    # Test NoRepair
    print("\n8. Testing NoRepair...")
    no_repair = NoRepair()
    repaired_none = no_repair(population, xl, xu)
    assert torch.allclose(repaired_none, population)
    print(f"   NoRepair returns unchanged: ✓")
    
    # Test is_within_bounds helper
    print("\n9. Testing is_within_bounds...")
    feasible = clip.is_within_bounds(repaired_clip, xl, xu)
    assert feasible.all()
    
    infeasible = clip.is_within_bounds(population, xl, xu)
    print(f"   Feasible before repair: {infeasible.sum().item()}/{n_pop}")
    
    print("\n✓ repair.py tests passed!")


def test_operator_integration():
    """Test operators working together."""
    print("\n" + "="*60)
    print("Testing Operator Integration")
    print("="*60)
    
    # Create problem
    problem = create_test_problem(n_var=10)
    
    print("\n1. Testing full GA-style pipeline...")
    
    # Sampling
    sampling = UniformSampling(seed=42)
    population = sampling(50, problem)
    print(f"   Initial population: {population.shape}")
    
    # Evaluate
    fitness = problem.evaluate(population)
    print(f"   Fitness: {fitness.shape}")
    
    # Selection
    selection = TournamentSelection(tournament_size=3)
    parents = selection(population, fitness, 50)
    print(f"   Selected parents: {parents.shape}")
    
    # Crossover
    crossover = SBXCrossover(eta=15, prob=0.9)
    p1, p2 = parents[:25], parents[25:]
    offspring = crossover(p1, p2)
    print(f"   Offspring after crossover: {offspring.shape}")
    
    # Mutation
    mutation = PolynomialMutation(eta=20, prob=0.1)
    offspring = mutation(offspring, problem.xl, problem.xu)
    print(f"   Offspring after mutation: {offspring.shape}")
    
    # Repair
    repair = ReflectRepair()
    offspring = repair(offspring, problem.xl, problem.xu)
    print(f"   Offspring after repair: {offspring.shape}")
    
    # Verify bounds
    assert (offspring >= problem.xl).all() and (offspring <= problem.xu).all()
    print(f"   All offspring within bounds: ✓")
    
    print("\n2. Testing differentiable pipeline...")
    
    # Create differentiable operators
    selection_diff = TournamentSelection(
        tournament_size=3,
        adaptive=True,
        temperature=1.0,
    )
    crossover_diff = SBXCrossover(
        eta=15,
        prob=0.9,
        adaptive=True,
        learn_eta=True,
    )
    mutation_diff = GaussianMutation(
        sigma=0.1,
        adaptive=True,
        learn_sigma=True,
    )
    
    # Run through pipeline with gradient tracking
    pop_param = torch.nn.Parameter(population.clone())
    
    parents_diff = selection_diff(pop_param, fitness, 50)
    p1_diff, p2_diff = parents_diff[:25], parents_diff[25:]
    offspring_diff = crossover_diff(p1_diff, p2_diff)
    offspring_diff = mutation_diff(offspring_diff, problem.xl, problem.xu)
    
    # Compute loss and backprop
    loss = offspring_diff.sum()
    loss.backward()
    
    assert pop_param.grad is not None
    print(f"   Gradients flow through entire pipeline: ✓")
    
    # Check learnable parameters have gradients
    assert crossover_diff._log_eta.grad is not None
    print(f"   Crossover eta gradient: {crossover_diff._log_eta.grad.item():.6f}")
    
    assert mutation_diff._log_sigma.grad is not None
    print(f"   Mutation sigma gradient: {mutation_diff._log_sigma.grad.item():.6f}")
    
    print("\n✓ Operator integration tests passed!")


def run_all_tests():
    """Run all operator tests."""
    print("\n" + "#"*60)
    print("# EvoGrad Operators Module Tests")
    print("#"*60)
    
    try:
        test_sampling()
        test_selection()
        test_crossover()
        test_mutation()
        test_repair()
        test_operator_integration()
        
        print("\n" + "="*60)
        print("✓ ALL OPERATORS TESTS PASSED!")
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
