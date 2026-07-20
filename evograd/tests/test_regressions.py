"""Regression coverage for defects found during the manuscript/code audit."""

import math
import sys
import warnings

import torch

from evograd.algorithms import CMAES, DE, LSHADE, PSO, cmaes_large, cmaes_small
from evograd.benchmarks.functions import (
    AsymmetricFunction,
    BiasedFunction,
    MultiBasinRosenbrock,
    OscillatedFunction,
    RotatedFunction,
    ScaledFunction,
    ShiftedFunction,
    ShiftedRotatedFunction,
    Sphere,
    get_cec2017_function,
)
from evograd.benchmarks.functions.cec2017 import basic
from evograd.core import (
    MaxEvaluations,
    MaxGenerations,
    Problem,
    TargetReached,
    maximize,
    minimize,
)
from evograd.operators import PolynomialMutation
from evograd.operators.crossover import (
    ExponentialCrossover,
    NPointCrossover,
    UniformCrossover,
)
from evograd.operators.relaxations import expand_param
from evograd.operators.repair import ReflectRepair, WrapRepair


def sphere(x):
    return x.square().sum(dim=-1)


def test_default_lr_is_classical_and_minus_one_learns():
    problem = Problem(sphere, 3, -5.0, 5.0)

    classical = DE(pop_size=8, adaptive=True, differentiable=False)
    name, parameter = next((n, p) for n, p in classical.named_parameters() if p.requires_grad)
    before = parameter.detach().clone()
    minimize(problem, classical, MaxEvaluations(16), seed=1, verbose=False)
    assert torch.equal(parameter.detach(), before), f"{name} changed with lr_hyper=None"
    assert all(p.grad is None for p in classical.parameters())

    learned = DE(pop_size=8, adaptive=True, differentiable=False)
    _, learned_parameter = next((n, p) for n, p in learned.named_parameters() if p.requires_grad)
    before = learned_parameter.detach().clone()
    minimize(
        problem,
        learned,
        MaxEvaluations(16),
        seed=1,
        verbose=False,
        lr_hyper=-1,
    )
    assert not torch.equal(learned_parameter.detach(), before)


def test_best_solution_fitness_pairing():
    problem = Problem(sphere, 3, -5.0, 5.0)
    result = minimize(
        problem,
        PSO(pop_size=8, differentiable=True, adaptive=False),
        MaxEvaluations(48),
        seed=12,
        verbose=False,
        lr_pop=0.05,
    )
    actual = float(problem.evaluate(result.best_solution))
    assert abs(actual - result.best_fitness) < 1e-6


def test_constraints_and_result_feasibility_flag():
    problem = Problem(
        lambda x: x[:, 0],
        1,
        0.0,
        1.0,
        constraints=[(lambda x: 0.8 - x[:, 0], "ineq")],
    )
    result = minimize(
        problem,
        DE(pop_size=10, differentiable=False, adaptive=False),
        MaxEvaluations(200),
        seed=7,
        verbose=False,
    )
    assert bool(problem.is_feasible(result.best_solution))
    assert result.extra["best_feasible"] is True
    assert result.extra["best_constraint_violation"] <= 1e-6

    impossible = Problem(
        lambda x: x[:, 0],
        1,
        0.0,
        1.0,
        constraints=[(lambda x: 2.0 - x[:, 0], "ineq")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        infeasible = minimize(
            impossible,
            DE(pop_size=6, differentiable=False, adaptive=False),
            MaxEvaluations(24),
            seed=7,
            verbose=False,
        )
    assert infeasible.extra["best_feasible"] is False
    assert any("infeasible" in str(item.message) for item in caught)


def test_maximize_nested_target():
    problem = Problem(lambda x: x[:, 0], 1, 0.0, 1.0)
    termination = (
        TargetReached(0.2, minimize=False) | MaxEvaluations(30)
    ) & MaxGenerations(1)
    result = maximize(
        problem,
        DE(pop_size=6, differentiable=False, adaptive=False),
        termination,
        seed=5,
        verbose=False,
    )
    assert result.success
    assert result.best_fitness >= 0.2
    assert ">= 0.2" in result.termination_reason
    # Translation must not mutate the user's criterion tree.
    target = termination.criteria[0].criteria[0]
    assert target.target_fitness == 0.2 and target.minimize is False


def test_lshade_reduces_in_classical_and_differentiable_modes():
    problem = Problem(sphere, 3, -5.0, 5.0)
    for differentiable in (False, True):
        algorithm = LSHADE(
            pop_size_init=10,
            pop_size_min=4,
            differentiable=differentiable,
        )
        result = minimize(
            problem,
            algorithm,
            MaxEvaluations(50),
            seed=2,
            verbose=False,
            lr_pop=-1 if differentiable else None,
        )
        assert result.n_evals == 50
        assert algorithm.pop_size == len(algorithm.population) == 4


def test_bounded_polynomial_mutation_and_fixed_repairs():
    mutation = PolynomialMutation(eta=1.0, prob=1.0, adaptive=True)
    x = torch.full((1000, 3), 0.99, requires_grad=True)
    y = mutation(x, torch.full((3,), -1.0), torch.ones(3))
    assert bool(((y >= -1.0) & (y <= 1.0)).all())
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    fixed_x = torch.tensor([[3.0, -4.0]])
    fixed = torch.tensor([2.0, -1.0])
    for repair in (ReflectRepair(), WrapRepair()):
        assert torch.equal(repair(fixed_x, fixed, fixed), fixed.unsqueeze(0))


def test_adaptive_crossovers_have_live_gradients():
    for cls in (ExponentialCrossover, UniformCrossover, NPointCrossover):
        parent1 = torch.randn(12, 6, requires_grad=True)
        parent2 = torch.randn(12, 6, requires_grad=True)
        operator = cls(adaptive=True)
        operator(parent1, parent2).square().mean().backward()
        grads = [p.grad for p in operator.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), cls.__name__


def test_n_equals_d_parameter_disambiguation_and_shade_path():
    try:
        expand_param(
            torch.arange(4.0),
            0.0,
            4,
            4,
            torch.device("cpu"),
            torch.float32,
        )
    except ValueError as error:
        assert "Ambiguous" in str(error)
    else:
        raise AssertionError("ambiguous [N]==[D] parameter was silently accepted")

    per_individual = expand_param(
        torch.arange(4.0).reshape(4, 1),
        0.0,
        4,
        4,
        torch.device("cpu"),
        torch.float32,
    )
    assert torch.equal(per_individual[:, 0], torch.arange(4.0))
    assert bool((per_individual == per_individual[:, :1]).all())

    # N == D == 1 is unambiguous (both interpretations give the same [1, 1]);
    # it must not raise the disambiguation error.
    singleton = expand_param(
        torch.tensor([0.9]),
        0.0,
        1,
        1,
        torch.device("cpu"),
        torch.float32,
    )
    assert singleton.shape == (1, 1)
    assert abs(float(singleton) - 0.9) < 1e-6


def test_all_transform_wrappers_construct():
    base = Sphere(3)
    wrappers = (
        ShiftedFunction,
        RotatedFunction,
        ShiftedRotatedFunction,
        ScaledFunction,
        AsymmetricFunction,
        OscillatedFunction,
        BiasedFunction,
    )
    for wrapper in wrappers:
        transformed = wrapper(base)
        value = transformed(torch.zeros(2, 3))
        assert value.shape == (2,) and torch.isfinite(value).all(), wrapper.__name__


def test_cec_composition_centres_and_f9_optimum():
    for number in range(21, 31):
        function = get_cec2017_function(number, n_var=10)
        for shift in function.shifts:
            value = function(shift.reshape(1, -1))
            assert torch.isfinite(value).all(), f"F{number} returned {value}"

    for dimension in (10, 30, 100):
        function = get_cec2017_function(9, n_var=dimension)
        assert bool(((function.optimal_x >= -100) & (function.optimal_x <= 100)).all())
        assert abs(float(function(function.optimal_x.reshape(1, -1))) - 900.0) < 1e-4


def test_cec_rosenbrock_noncyclic_and_schaffer_cyclic():
    x = torch.tensor([[0.3, -0.2, 0.7]], dtype=torch.float64)
    scaled = 0.02048 * x + 1.0
    expected_rosen = (
        100 * (scaled[:, 1:] - scaled[:, :-1].square()).square()
        + (scaled[:, :-1] - 1).square()
    ).sum(dim=-1)
    assert torch.allclose(basic.rosenbrock(x), expected_rosen, atol=1e-12, rtol=1e-12)

    rolled = torch.roll(x, -1, dims=-1)
    radius = x.square() + rolled.square()
    expected_schaffer = (
        0.5
        + (torch.sin(torch.sqrt(radius)).square() - 0.5)
        / (1 + 0.001 * radius).square()
    ).sum(dim=-1)
    assert torch.allclose(
        basic.expanded_schaffers_f6(x),
        expected_schaffer,
        atol=1e-12,
        rtol=1e-12,
    )


def test_cmaes_soft_weights_are_scale_invariant_and_converge():
    problem = Problem(sphere, 10, -5.0, 5.0)
    algorithm = CMAES(differentiable=True)
    algorithm.initialize(problem)
    # Fitness always arrives on the algorithm's device in production; match it
    # so the test exercises the real code path on CPU and accelerators alike.
    device = algorithm.selection_temperature.device
    fitness = torch.tensor([0.1, 0.4, 1.0, 2.0], device=device)
    weights = algorithm._soft_recombination_weights(fitness)
    scaled = algorithm._soft_recombination_weights(1000 * fitness + 123.0)
    assert torch.allclose(weights, scaled, atol=1e-6, rtol=1e-6)
    constant = algorithm._soft_recombination_weights(torch.full((4,), 123.0, device=device))
    assert torch.isfinite(constant).all()
    assert torch.allclose(constant, torch.full((4,), 0.25, device=device))

    result = minimize(
        problem,
        CMAES(differentiable=True),
        MaxEvaluations(2000),
        seed=0,
        verbose=False,
        lr_pop=-1,
    )
    assert result.best_fitness < 1e-8, result.best_fitness

    small, large = cmaes_small(), cmaes_large(pop_size_factor=3)
    small.initialize(problem)
    large.initialize(problem)
    assert large.pop_size == 3 * small.pop_size


def test_multibasin_rosenbrock_reference_optimum_is_in_bounds():
    for seed in range(100):
        function = MultiBasinRosenbrock(n_var=10, seed=seed)
        assert bool(((function.optimal_x >= function.xl) & (function.optimal_x <= function.xu)).all())
        value = float(function(function.optimal_x.reshape(1, -1)))
        assert math.isfinite(value)
        assert abs(value - function.optimal_value) < 1e-6


TESTS = (
    test_default_lr_is_classical_and_minus_one_learns,
    test_best_solution_fitness_pairing,
    test_constraints_and_result_feasibility_flag,
    test_maximize_nested_target,
    test_lshade_reduces_in_classical_and_differentiable_modes,
    test_bounded_polynomial_mutation_and_fixed_repairs,
    test_adaptive_crossovers_have_live_gradients,
    test_n_equals_d_parameter_disambiguation_and_shade_path,
    test_all_transform_wrappers_construct,
    test_cec_composition_centres_and_f9_optimum,
    test_cec_rosenbrock_noncyclic_and_schaffer_cyclic,
    test_cmaes_soft_weights_are_scale_invariant_and_converge,
    test_multibasin_rosenbrock_reference_optimum_is_in_bounds,
)


def run_all_tests():
    print("\n" + "#" * 60)
    print("# Audit Regression Tests")
    print("#" * 60)
    ok = True
    for index, test in enumerate(TESTS, 1):
        try:
            test()
            print(f"{index:2d}. {test.__name__}: PASS")
        except Exception:
            ok = False
            print(f"{index:2d}. {test.__name__}: FAIL")
            import traceback
            traceback.print_exc()
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
