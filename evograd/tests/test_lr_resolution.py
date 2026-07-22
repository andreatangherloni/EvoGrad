"""Regression coverage for the 0.4.0 learning-rate resolution semantics.

Contract under test (per gradient channel; ``lr_pop`` drives the population
via ``differentiable=True``, ``lr_hyper`` drives the hyperparameters via
``adaptive=True`` / differentiable operators):

- ``None`` (auto): per-algorithm default when the channel exposes learnable
  parameters AND the objective provides a gradient; classical fallback with a
  warning otherwise.
- ``0``: explicit off, with a warning.
- ``> 0``: used as-is, still subject to the objective providing a gradient
  (warning + classical fallback if it does not).
- ``< 0``: ValueError (the former ``-1`` sentinel was removed).

The objective-gradient probe must be RNG-neutral and excluded from n_evals,
and a channel whose parameters receive no gradient on the first generation
(e.g. CMA-ES's population) is dropped with a warning.
"""

import warnings

import torch

from evograd.algorithms import CMAES, DE, GA, PSO
from evograd.core import MaxEvaluations, Problem, minimize


def sphere(x):
    return x.square().sum(dim=-1)


def blackbox_sphere(x):
    with torch.no_grad():
        return x.square().sum(dim=-1)


def stochastic_blackbox(x):
    # Consumes RNG internally AND detaches: exercises the probe's
    # snapshot/restore path on a black-box objective.
    with torch.no_grad():
        return x.square().sum(dim=-1) + 0.0 * torch.rand(
            x.shape[0], device=x.device
        )


def _problem(objective, n_var=3):
    return Problem(objective, n_var, -5.0, 5.0, device="cpu")


def _run(algorithm, objective, budget=48, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(
            _problem(objective),
            algorithm,
            MaxEvaluations(budget),
            seed=3,
            verbose=False,
            **kwargs,
        )
    return result, [str(item.message) for item in caught]


def _algorithms(**kwargs):
    return (
        GA(pop_size=8, **kwargs),
        DE(pop_size=8, **kwargs),
        PSO(pop_size=8, **kwargs),
        CMAES(pop_size=8, **kwargs),
    )


def test_auto_enables_defaults_on_differentiable_objective():
    for algorithm in _algorithms(differentiable=True, adaptive=True):
        result, _ = _run(algorithm, sphere)
        channels = result.extra["gradient_channels"]
        # The hyperparameter channel always carries gradient; the population
        # channel may be dropped by the first-generation diagnostic (CMA-ES).
        assert channels["hyperparams"] is True, type(algorithm).__name__
        assert result.extra["lr_hyper_effective"] is not None
        assert result.extra["lr_pop_effective"] is not None


def test_auto_falls_back_to_classical_on_blackbox_objective():
    for algorithm in _algorithms(differentiable=True, adaptive=True):
        result, warns = _run(algorithm, blackbox_sphere)
        channels = result.extra["gradient_channels"]
        assert channels == {"population": False, "hyperparams": False}
        assert any("does not provide a gradient" in w for w in warns)
        # The probe is excluded from the evaluation budget.
        assert result.n_evals == 48, type(algorithm).__name__


def test_explicit_lr_with_blackbox_objective_warns_and_degrades():
    algorithm = DE(pop_size=8, differentiable=True, adaptive=True)
    result, warns = _run(algorithm, blackbox_sphere, lr_pop=1e-3, lr_hyper=1e-3)
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": False,
    }
    assert any("lr_pop=0.001 was requested" in w for w in warns)
    assert any("lr_hyper=0.001 was requested" in w for w in warns)


def test_zero_explicitly_disables_with_warning():
    algorithm = DE(pop_size=8, differentiable=True, adaptive=True)
    hyper_before = {
        n: p.detach().clone()
        for n, p in algorithm.named_parameters()
        if n != "_population"
    }
    result, warns = _run(algorithm, sphere, lr_pop=0, lr_hyper=0)
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": False,
    }
    assert sum("explicitly disables" in w for w in warns) == 2
    for name, before in hyper_before.items():
        current = dict(algorithm.named_parameters())[name].detach()
        assert torch.equal(current, before), name


def test_lr_on_inactive_flag_warns():
    algorithm = DE(pop_size=8, differentiable=False, adaptive=False)
    result, warns = _run(algorithm, sphere, lr_pop=0.01, lr_hyper=0.01)
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": False,
    }
    assert any("no learnable parameters" in w for w in warns)


def test_negative_lr_and_clip_raise():
    for kwargs in (
        dict(lr_pop=-1),
        dict(lr_hyper=-1),
        dict(grad_clip_pop=-1),
        dict(grad_clip_hyper=-1),
        dict(lr_pop=-0.5),
    ):
        try:
            _run(DE(pop_size=8), sphere, **kwargs)
        except ValueError as exc:
            assert "-1 sentinel was removed" in str(exc), kwargs
        else:
            raise AssertionError(f"{kwargs} must raise ValueError")


def test_cmaes_population_channel_dropped_by_diagnostic():
    # CMA-ES's loss is built from offspring resampled from the mean, so the
    # population Parameter never receives gradient: the SGD channel must be
    # dropped on the first generation with a warning, while the
    # hyperparameter channel keeps learning.
    algorithm = CMAES(pop_size=8, differentiable=True, adaptive=True)
    result, warns = _run(algorithm, sphere)
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": True,
    }
    assert any("received no gradient on the first generation" in w for w in warns)


def test_probe_is_rng_neutral_on_blackbox_runs():
    # With identical flags, an auto black-box run (which probes the objective,
    # then falls back to the classical loop) must follow the exact same
    # trajectory as an lr=0 run (which never probes): the probe may not
    # consume RNG — including for objectives that draw random numbers
    # internally.
    for objective in (blackbox_sphere, stochastic_blackbox):
        no_probe, _ = _run(
            DE(pop_size=8, differentiable=True, adaptive=True),
            objective,
            lr_pop=0,
            lr_hyper=0,
        )
        with_probe, _ = _run(
            DE(pop_size=8, differentiable=True, adaptive=True), objective
        )
        assert no_probe.history["best_fitness"] == with_probe.history[
            "best_fitness"
        ], objective.__name__
        assert torch.equal(no_probe.best_solution, with_probe.best_solution)


def test_flags_off_with_default_lrs_is_silent():
    # Approved decision-table row 7: flags off + lr omitted stays classical
    # with NO warning — old classical call sites keep working quietly.
    result, warns = _run(
        DE(pop_size=8, differentiable=False, adaptive=False), sphere
    )
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": False,
    }
    assert warns == [], warns


def test_maximize_auto_resolution():
    # Auto-resolution must work through maximize()'s _NegatedProblem wrapper:
    # negation preserves the objective's grad_fn, so channels turn on.
    from evograd.core import maximize

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = maximize(
            _problem(lambda x: -x.square().sum(dim=-1)),
            DE(pop_size=8, differentiable=True, adaptive=True),
            MaxEvaluations(48),
            seed=3,
            verbose=False,
        )
    assert result.extra["gradient_channels"]["hyperparams"] is True
    assert not any("does not provide a gradient" in str(w.message) for w in caught)


def _initialized_de_with_hyper_optimizer(lr=1e-3):
    # Learnable parameters only exist post-initialize; initialize() is
    # idempotent, so minimize()'s own initialize call becomes a no-op.
    algorithm = DE(pop_size=8, differentiable=True, adaptive=True)
    algorithm.initialize(_problem(sphere))
    hyper = [
        p for n, p in algorithm.named_parameters()
        if p.requires_grad and n != "_population"
    ]
    return algorithm, torch.optim.Adam(hyper, lr=lr)


def test_user_optimizer_channels_reflect_actual_coverage():
    # A supplied optimizer covering only the hyperparameters must not report
    # the population channel as gradient-driven.
    algorithm, opt = _initialized_de_with_hyper_optimizer()
    result, _ = _run(algorithm, sphere, optimizer=opt)
    assert result.extra["gradient_channels"] == {
        "population": False,
        "hyperparams": True,
    }

    # Negative sentinels raise on the optimizer= path too.
    algorithm2, opt2 = _initialized_de_with_hyper_optimizer()
    try:
        _run(algorithm2, sphere, optimizer=opt2, lr_pop=-1)
    except ValueError as exc:
        assert "-1 sentinel was removed" in str(exc)
    else:
        raise AssertionError("lr_pop=-1 with optimizer= must raise ValueError")

    # Provided lr args are flagged as ignored.
    algorithm3, opt3 = _initialized_de_with_hyper_optimizer()
    _, warns3 = _run(algorithm3, sphere, optimizer=opt3, lr_hyper=0.5)
    assert any("ignored when optimizer=" in w for w in warns3)


def numpy_blackbox(x):
    # Draws from NumPy's global RNG and detaches: the probe must snapshot and
    # restore the NumPy stream too, not only torch's.
    import numpy as np

    with torch.no_grad():
        noise = float(np.random.rand())
        return x.square().sum(dim=-1) + 0.0 * noise


def test_probe_is_rng_neutral_for_numpy_streams():
    no_probe, _ = _run(
        DE(pop_size=8, differentiable=True, adaptive=True),
        numpy_blackbox,
        lr_pop=0,
        lr_hyper=0,
    )
    with_probe, _ = _run(
        DE(pop_size=8, differentiable=True, adaptive=True), numpy_blackbox
    )
    assert no_probe.history["best_fitness"] == with_probe.history["best_fitness"]


def test_probe_sees_constraint_penalty_gradient():
    # A black-box objective with differentiable declared constraints still
    # yields a differentiable loss (the exterior penalty), so gradient mode
    # must stay available — the probe checks the same composite the loss uses.
    problem = Problem(
        blackbox_sphere,
        3,
        -5.0,
        5.0,
        constraints=[(lambda x: x.sum(dim=-1) - 1.0, "ineq")],
        device="cpu",
    )
    algorithm = GA(pop_size=8, differentiable=True, adaptive=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(
            problem, algorithm, MaxEvaluations(48), seed=3, verbose=False,
            lr_pop=0.01,
        )
    assert result.extra["gradient_channels"]["population"] is True
    assert not any(
        "does not provide a gradient" in str(w.message) for w in caught
    )


def batch_only_sphere(x):
    # Differentiable, but rejects the single-row probe: the probe must retry
    # with a population-sized batch.
    assert x.shape[0] > 1, "requires batched input"
    return x.square().sum(dim=-1)


def test_probe_retries_with_population_batch():
    result, warns = _run(
        DE(pop_size=8, differentiable=True, adaptive=True), batch_only_sphere
    )
    assert result.extra["gradient_channels"]["hyperparams"] is True
    assert not any("does not provide a gradient" in w for w in warns)


def test_inconclusive_probe_honors_explicit_lr():
    # If the probe itself cannot run, auto stays classical but an explicitly
    # requested learning rate is honored unverified.
    import sys

    minimize_module = sys.modules["evograd.core.minimize"]
    original = minimize_module._probe_objective_gradient
    minimize_module._probe_objective_gradient = lambda *a, **k: None
    try:
        auto, _ = _run(DE(pop_size=8, differentiable=True, adaptive=True), sphere)
        assert auto.extra["gradient_channels"] == {
            "population": False,
            "hyperparams": False,
        }
        explicit, _ = _run(
            DE(pop_size=8, differentiable=True, adaptive=True),
            sphere,
            lr_pop=0.01,
            lr_hyper=0.001,
        )
        assert explicit.extra["gradient_channels"] == {
            "population": True,
            "hyperparams": True,
        }
    finally:
        minimize_module._probe_objective_gradient = original


def test_auto_matches_explicit_default_values():
    # None (auto) must resolve to exactly the per-algorithm defaults: a run
    # with explicit default values follows the identical trajectory.
    from evograd.core.minimize import _OPT_DEFAULTS

    defaults = _OPT_DEFAULTS["DE"]
    auto, _ = _run(DE(pop_size=8, differentiable=True, adaptive=True), sphere)
    explicit, _ = _run(
        DE(pop_size=8, differentiable=True, adaptive=True),
        sphere,
        lr_pop=defaults["lr_pop"],
        lr_hyper=defaults["lr_hyper"],
    )
    assert auto.history["best_fitness"] == explicit.history["best_fitness"]
    assert torch.equal(auto.best_solution, explicit.best_solution)
