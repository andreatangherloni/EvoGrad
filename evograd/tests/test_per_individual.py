"""
Test script for per-individual/per-gene parameter support in operators.

This script demonstrates the four parameter configurations:
    1. Fixed (scalar): Same value for all individuals and genes
    2. Per-gene [D]: Different value per gene, same across individuals
    3. Per-individual [N]: Different value per individual, same across genes  
    4. Per-gene + Per-individual [N, D]: Full matrix

Run with:
    python -m tests.test_per_individual
"""

import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evograd.operators.crossover import (
    BinomialCrossover,
    SBXCrossover,
    BlendCrossover,
    ArithmeticCrossover,
)
from evograd.operators.mutation import (
    PolynomialMutation,
    GaussianMutation,
    UniformMutation,
)


def test_crossover_configurations():
    """Test all four parameter configurations for crossover operators."""
    print("\n" + "=" * 70)
    print("Testing Crossover Per-Individual/Per-Gene Configurations")
    print("=" * 70)
    
    N, D = 50, 10  # 50 individuals, 10 variables
    parent1 = torch.randn(N, D)
    parent2 = torch.randn(N, D)
    
    # ==========================================================================
    # BinomialCrossover (DE-style) - Most important for SHADE
    # ==========================================================================
    print("\n1. BinomialCrossover configurations:")
    crossover = BinomialCrossover(cr=0.9)
    
    # Config 1: Fixed (default)
    print("   a) Fixed CR (scalar) - default behavior")
    trial = crossover(parent1, parent2)
    assert trial.shape == (N, D), f"Expected ({N}, {D}), got {trial.shape}"
    print(f"      Shape: {trial.shape} ✓")
    
    # Config 2: Per-gene [D]
    print("   b) Per-gene CR [D]")
    cr_per_gene = torch.linspace(0.5, 1.0, D)  # Different CR per gene
    trial = crossover(parent1, parent2, cr=cr_per_gene)
    assert trial.shape == (N, D)
    print(f"      CR shape: {cr_per_gene.shape} -> Output: {trial.shape} ✓")
    
    # Config 3: Per-individual [N] - SHADE needs this!
    print("   c) Per-individual CR [N] - SHADE-style")
    cr_per_ind = torch.rand(N) * 0.5 + 0.5  # CR in [0.5, 1.0] per individual
    trial = crossover(parent1, parent2, cr=cr_per_ind)
    assert trial.shape == (N, D)
    print(f"      CR shape: {cr_per_ind.shape} -> Output: {trial.shape} ✓")
    
    # Config 4: Full matrix [N, D]
    print("   d) Full matrix CR [N, D]")
    cr_matrix = torch.rand(N, D)
    trial = crossover(parent1, parent2, cr=cr_matrix)
    assert trial.shape == (N, D)
    print(f"      CR shape: {cr_matrix.shape} -> Output: {trial.shape} ✓")
    
    # ==========================================================================
    # SBXCrossover - Test eta and prob overrides
    # ==========================================================================
    print("\n2. SBXCrossover configurations:")
    sbx = SBXCrossover(eta=15, prob=0.9)
    
    # Per-individual eta
    print("   a) Per-individual eta [N]")
    eta_per_ind = torch.rand(N) * 15 + 5  # eta in [5, 20] per individual
    offspring = sbx(parent1, parent2, eta=eta_per_ind)
    assert offspring.shape == (N, D)
    print(f"      Eta shape: {eta_per_ind.shape} -> Output: {offspring.shape} ✓")
    
    # Per-gene prob
    print("   b) Per-gene prob [D]")
    prob_per_gene = torch.linspace(0.5, 1.0, D)
    offspring = sbx(parent1, parent2, prob=prob_per_gene)
    assert offspring.shape == (N, D)
    print(f"      Prob shape: {prob_per_gene.shape} -> Output: {offspring.shape} ✓")
    
    # Both eta and prob as full matrices
    print("   c) Full matrix eta and prob [N, D]")
    eta_matrix = torch.rand(N, D) * 15 + 5
    prob_matrix = torch.rand(N, D) * 0.5 + 0.5
    offspring = sbx(parent1, parent2, eta=eta_matrix, prob=prob_matrix)
    assert offspring.shape == (N, D)
    print(f"      Eta: {eta_matrix.shape}, Prob: {prob_matrix.shape} -> Output: {offspring.shape} ✓")
    
    # ==========================================================================
    # ArithmeticCrossover - Test alpha override
    # ==========================================================================
    print("\n3. ArithmeticCrossover configurations:")
    arith = ArithmeticCrossover(alpha=0.5)
    
    # Per-individual alpha
    print("   a) Per-individual alpha [N]")
    alpha_per_ind = torch.rand(N)
    offspring = arith(parent1, parent2, alpha=alpha_per_ind)
    assert offspring.shape == (N, D)
    print(f"      Alpha shape: {alpha_per_ind.shape} -> Output: {offspring.shape} ✓")
    
    # Verify arithmetic crossover formula
    print("   b) Verify formula: offspring = alpha * p1 + (1-alpha) * p2")
    alpha_test = 0.3
    offspring_test = arith(parent1, parent2, alpha=alpha_test)
    expected = alpha_test * parent1 + (1 - alpha_test) * parent2
    assert torch.allclose(offspring_test, expected, atol=1e-6)
    print(f"      Formula verified ✓")
    
    print("\n✓ All crossover configurations passed!")


def test_mutation_configurations():
    """Test all four parameter configurations for mutation operators."""
    print("\n" + "=" * 70)
    print("Testing Mutation Per-Individual/Per-Gene Configurations")
    print("=" * 70)
    
    N, D = 50, 10  # 50 individuals, 10 variables
    population = torch.randn(N, D)
    xl = torch.zeros(D)
    xu = torch.ones(D)
    
    # ==========================================================================
    # PolynomialMutation - Test eta and prob overrides
    # ==========================================================================
    print("\n1. PolynomialMutation configurations:")
    mutation = PolynomialMutation(eta=20, prob=0.1)
    
    # Config 1: Fixed (default)
    print("   a) Fixed eta and prob (scalar) - default behavior")
    mutated = mutation(population, xl, xu)
    assert mutated.shape == (N, D)
    print(f"      Shape: {mutated.shape} ✓")
    
    # Config 2: Per-gene [D]
    print("   b) Per-gene eta [D]")
    eta_per_gene = torch.linspace(10, 30, D)
    mutated = mutation(population, xl, xu, eta=eta_per_gene)
    assert mutated.shape == (N, D)
    print(f"      Eta shape: {eta_per_gene.shape} -> Output: {mutated.shape} ✓")
    
    # Config 3: Per-individual [N]
    print("   c) Per-individual eta [N] - Self-adaptive GA style")
    eta_per_ind = torch.rand(N) * 20 + 10  # eta in [10, 30] per individual
    mutated = mutation(population, xl, xu, eta=eta_per_ind)
    assert mutated.shape == (N, D)
    print(f"      Eta shape: {eta_per_ind.shape} -> Output: {mutated.shape} ✓")
    
    # Config 4: Full matrix [N, D]
    print("   d) Full matrix eta and prob [N, D]")
    eta_matrix = torch.rand(N, D) * 20 + 10
    prob_matrix = torch.rand(N, D) * 0.2
    mutated = mutation(population, xl, xu, eta=eta_matrix, prob=prob_matrix)
    assert mutated.shape == (N, D)
    print(f"      Eta: {eta_matrix.shape}, Prob: {prob_matrix.shape} -> Output: {mutated.shape} ✓")
    
    # ==========================================================================
    # GaussianMutation - Test sigma override (important for DE/SHADE)
    # ==========================================================================
    print("\n2. GaussianMutation configurations:")
    gauss = GaussianMutation(sigma=0.1)
    
    # Per-individual sigma (like F in DE/SHADE)
    print("   a) Per-individual sigma [N] - SHADE F-style")
    F_per_ind = torch.rand(N) * 0.5 + 0.5  # F in [0.5, 1.0] per individual
    mutated = gauss(population, xl, xu, sigma=F_per_ind)
    assert mutated.shape == (N, D)
    print(f"      Sigma shape: {F_per_ind.shape} -> Output: {mutated.shape} ✓")
    
    # Per-gene sigma
    print("   b) Per-gene sigma [D]")
    sigma_per_gene = torch.linspace(0.05, 0.2, D)
    mutated = gauss(population, xl, xu, sigma=sigma_per_gene)
    assert mutated.shape == (N, D)
    print(f"      Sigma shape: {sigma_per_gene.shape} -> Output: {mutated.shape} ✓")
    
    # ==========================================================================
    # UniformMutation - Test prob override
    # ==========================================================================
    print("\n3. UniformMutation configurations:")
    unif = UniformMutation(prob=0.05)
    
    # Per-individual prob
    print("   a) Per-individual prob [N]")
    prob_per_ind = torch.rand(N) * 0.1
    mutated = unif(population, xl, xu, prob=prob_per_ind)
    assert mutated.shape == (N, D)
    print(f"      Prob shape: {prob_per_ind.shape} -> Output: {mutated.shape} ✓")
    
    print("\n✓ All mutation configurations passed!")


def test_shade_style_usage():
    """Demonstrate SHADE-style usage with per-individual F and CR."""
    print("\n" + "=" * 70)
    print("Demonstrating SHADE-style Usage")
    print("=" * 70)
    
    N, D = 100, 30
    
    # Simulated SHADE setup
    print("\n1. Setting up SHADE-style parameters:")
    
    # Population
    population = torch.rand(N, D)
    xl = torch.zeros(D)
    xu = torch.ones(D)
    
    # Per-individual F and CR (sampled from historical memory in real SHADE)
    F_per_ind = torch.randn(N).abs() * 0.3 + 0.5  # F ~ |N(0.5, 0.3)|
    CR_per_ind = torch.randn(N) * 0.1 + 0.5  # CR ~ N(0.5, 0.1)
    CR_per_ind = CR_per_ind.clamp(0, 1)
    
    print(f"   F per individual: shape={F_per_ind.shape}, range=[{F_per_ind.min():.3f}, {F_per_ind.max():.3f}]")
    print(f"   CR per individual: shape={CR_per_ind.shape}, range=[{CR_per_ind.min():.3f}, {CR_per_ind.max():.3f}]")
    
    # Select random individuals for mutation (DE/rand/1)
    print("\n2. Creating donor vectors (DE/rand/1 style):")
    r1 = torch.randint(0, N, (N,))
    r2 = torch.randint(0, N, (N,))
    r3 = torch.randint(0, N, (N,))
    
    # Mutation: donor = x_r1 + F * (x_r2 - x_r3)
    # Here we use per-individual F
    diff = population[r2] - population[r3]
    donor = population[r1] + F_per_ind.unsqueeze(1) * diff
    print(f"   Donor vectors created: {donor.shape}")
    
    # Binomial crossover with per-individual CR
    print("\n3. Applying binomial crossover with per-individual CR:")
    crossover = BinomialCrossover(cr=0.5)  # Default CR, but we'll override
    trial = crossover(population, donor, cr=CR_per_ind)
    print(f"   Trial vectors created: {trial.shape}")
    
    # Verify different individuals have different number of genes crossed
    genes_from_donor = (trial == donor).float().sum(dim=1)
    print(f"   Genes from donor (per individual): mean={genes_from_donor.mean():.1f}, std={genes_from_donor.std():.1f}")
    
    print("\n✓ SHADE-style demonstration complete!")


def test_differentiable_mode():
    """Test gradient flow with per-individual parameters."""
    print("\n" + "=" * 70)
    print("Testing Gradient Flow with Per-Individual Parameters")
    print("=" * 70)
    
    N, D = 20, 5
    
    # Differentiable operators
    print("\n1. SBXCrossover with gradient flow:")
    sbx = SBXCrossover(eta=15, prob=0.9, adaptive=True, learn_eta=True)
    
    p1 = torch.nn.Parameter(torch.randn(N, D))
    p2 = torch.randn(N, D)
    eta_per_ind = torch.rand(N) * 10 + 5  # Not learnable, just passed in
    
    offspring = sbx(p1, p2, eta=eta_per_ind)
    loss = offspring.sum()
    loss.backward()
    
    assert p1.grad is not None, "Gradients should flow to parent1"
    print(f"   Gradients flow through per-individual eta: ✓")
    print(f"   Parent1 grad norm: {p1.grad.norm():.4f}")
    
    # PolynomialMutation with gradient flow
    print("\n2. PolynomialMutation with gradient flow:")
    mutation = PolynomialMutation(eta=20, prob=0.1, adaptive=True, learn_eta=True)
    
    x = torch.nn.Parameter(torch.randn(N, D))
    xl = torch.zeros(D)
    xu = torch.ones(D)
    eta_per_ind = torch.rand(N) * 20 + 10
    
    mutated = mutation(x, xl, xu, eta=eta_per_ind)
    loss = mutated.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow to input"
    print(f"   Gradients flow through per-individual eta: ✓")
    print(f"   Input grad norm: {x.grad.norm():.4f}")
    
    print("\n✓ All gradient tests passed!")


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# Per-Individual/Per-Gene Parameter Support Tests")
    print("#" * 70)

    try:
        test_crossover_configurations()
        test_mutation_configurations()
        test_shade_style_usage()
        test_differentiable_mode()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)

        print("\nSummary of Four Configurations:")
        print("  1. Fixed (scalar)     - Same value for all")
        print("  2. Per-gene [D]       - Different per variable")
        print("  3. Per-individual [N] - Different per individual (SHADE needs this!)")
        print("  4. Full matrix [N, D] - Maximum flexibility")
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
