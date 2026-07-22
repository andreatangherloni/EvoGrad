"""
Minimisation function for EvoGrad optimisation.

This module provides the main entry point for running optimisation,
following pymoo's interface style where algorithm initialisation
happens inside the minimize function.

Example:
    >>> from evograd.core.problem import Problem
    >>> from evograd.core.minimize import minimize
    >>> from evograd.core.termination import MaxEvaluations
    >>> from evograd.algorithms import GA
    >>> 
    >>> # Define problem
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Create algorithm (not initialized)
    >>> algorithm = GA(pop_size=100, eliminate_duplicates=True)
    >>> 
    >>> # Run optimisation
    >>> result = minimize(
    ...     problem,
    ...     algorithm,
    ...     termination=MaxEvaluations(10000),
    ...     seed=42,
    ...     verbose=True,
    ...     )
    >>> 
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> print(f"Best solution: {result.best_solution}")
"""

from __future__ import annotations

import random
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch

from evograd.core.result import Result, ResultBuilder
from evograd.operators.repair import clamp_to_bounds
from evograd.core.termination import (
    Termination,
    TerminationCollection,
    TargetReached,
    default_termination,
)
from evograd.utils.callbacks import (
    Callback,
    CallbackList,
    CallbackState,
    HistoryCallback,
    PrintCallback,
)
from evograd.utils.device import set_seed

if TYPE_CHECKING:
    from evograd.core.algorithm import Algorithm
    from evograd.core.problem import Problem

__all__ = [
    "minimize",
]

_OPT_DEFAULTS = {
    "GA":    dict(lr_pop=3e-4,  lr_hyper=0.001,  grad_clip_pop=0.2, grad_clip_hyper=0.2, pop_momentum=0.0),
    "DE":    dict(lr_pop=0.01,  lr_hyper=0.001,  grad_clip_pop=0.5, grad_clip_hyper=0.3, pop_momentum=0.9),
    "PSO":   dict(lr_pop=0.001, lr_hyper=0.001,  grad_clip_pop=1.0, grad_clip_hyper=0.1, pop_momentum=0.9),    
    "CMAES": dict(lr_pop=0.003, lr_hyper=0.0003, grad_clip_pop=0.5, grad_clip_hyper=0.1, pop_momentum=0.9),
}

def minimize(
    problem: Problem,
    algorithm: Algorithm,
    termination: Optional[Termination] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    callback: Optional[Union[Callback, List[Callback]]] = None,
    copy_algorithm: bool = False,
    save_history: bool = True,
    initialize: bool = True,
    # Differentiable mode options
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_pop: Optional[float] = None,
    lr_hyper: Optional[float] = None,
    grad_clip_pop: Optional[float] = None,
    grad_clip_hyper: Optional[float] = None,
    scheduler: Optional[str] = None,
    scheduler_patience: int = 50,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    reduction: str = "mean",
    live_selection: bool = True,
) -> Result:
    
    """
    Minimise an objective function using a population-based algorithm.
    
    This function initialises the algorithm with the problem and runs
    the optimisation loop until termination criteria are met. Follows
    pymoo's interface style.
    
    Differentiable Mode
    -------------------
    EvoGrad automatically detects learnable parameters (nn.Parameter with
    requires_grad=True) and uses backpropagation to update them, provided the
    objective actually provides a gradient. This is verified once per run
    with a cheap probe: an evaluation at the midpoint of the box bounds —
    including the exterior constraint penalty for constrained problems, i.e.
    the same composite the differentiable loss uses — whose output is checked
    for a ``grad_fn`` (RNG-neutral for torch/NumPy/Python generators and
    excluded from ``n_evals``; retried with a population-sized batch if the
    objective rejects single rows). If the objective is black-box (its output
    is detached), every gradient channel falls back to the classical update
    with a warning. Learnable parameters cover:
    
    - Population updates: algorithm.differentiable=True
    - Operator hyperparameters: operator.differentiable=True  
    - Adaptive PSO coefficients: adaptive=True (w, c1, c2 per particle)
    - Any other learnable parameters in the algorithm
    
    Thus, EvoGrad supports four combinations of differentiability:
    
    1. algorithm.differentiable=False, operators.differentiable=False
       → Pure classical EA, no backpropagation
       
    2. algorithm.differentiable=False, operators.differentiable=True
       → Classical EA dynamics, but learn operator hyperparameters
       (e.g., crossover eta, mutation rate, PSO w/c1/c2)
         
    3. algorithm.differentiable=True, operators.differentiable=False
       → Gradient-based population updates (local search), fixed operators
       
    4. algorithm.differentiable=True, operators.differentiable=True
       → Full end-to-end differentiable optimisation
        
    Args:
        problem: Problem instance defining the objective function,
            bounds, and constraints.
        algorithm: Algorithm instance (e.g., GA, DE, PSO, CMAES).
            Will be initialized inside this function.
        termination: When to stop optimisation. Must be a Termination
            instance (e.g., MaxEvaluations(10000)). If None, uses
            default (10000 evaluations).
        seed: Random seed for reproducibility. Applied before
            algorithm initialisation.
        verbose: If True, print progress during optimisation.
        callback: Single Callback or list of Callbacks for monitoring.
            HistoryCallback is always included automatically.
        copy_algorithm: If True, create a copy of the algorithm to
            preserve the original. Default False.
        save_history: If True (default), save convergence history
            in result. Set False to reduce memory for long runs.
        initialize: If True (default), initialize the algorithm with the
            problem. Set to False to continue optimization with an already
            initialized algorithm (e.g., when switching problems at runtime
            while preserving population state and hyperparameters).
            The algorithm must have been previously initialized. When False,
            the termination budget is additive (e.g., MaxEvaluations(500)
            will run 500 more evaluations from the current state).

        # Differentiable mode options (used if learnable params exist):
        optimizer: PyTorch optimizer for gradient-based updates.
            If None, SGD (population) / Adam (hyperparameters) are built from
            the resolved learning rates below.
        lr_pop: Population learning rate (SGD; requires
            ``algorithm.differentiable=True``). ``None`` (default) resolves
            automatically: if the population is learnable and the objective
            provides a gradient (checked once with an RNG-neutral probe
            evaluation), the per-algorithm default is used; if the population
            is learnable but the objective is black-box, the update stays
            classical with a warning; if the population is not learnable at
            all, it stays classical silently. ``0`` explicitly disables
            population gradient updates (warning). Positive values are used
            verbatim when the objective provides a gradient (warning +
            classical fallback when it provably does not; an inconclusive
            probe honors the explicit request). Negative values raise (the
            former ``-1`` sentinel was removed in 0.4.0). Note: the resolved
            ``lr_pop`` — default or explicit — is additionally scaled by
            ``1/sqrt(n_var)``; ``result.extra['lr_pop_effective']`` reports
            the scaled value.
        lr_hyper: Hyperparameter learning rate (Adam; requires
            ``adaptive=True`` and/or differentiable operators). Same
            resolution rules as ``lr_pop``, but without the dimension
            scaling. Both are ignored (with a warning) when ``optimizer=``
            is supplied.
        grad_clip_pop: Population gradient clipping. ``None`` selects the
            per-algorithm default when the population channel is
            gradient-driven; ``0`` disables clipping; positive values are
            used as-is; negative values raise.
        grad_clip_hyper: Hyperparameter gradient clipping. Same rules as
            ``grad_clip_pop``.
        scheduler: Learning rate scheduler type:
            - 'plateau': Reduce on plateau (default)
            - 'step': Reduce every N generations
            - 'cosine': Cosine annealing
            - 'exponential': Exponential decay
            - None: No scheduler
        scheduler_patience: Generations without improvement before
            reducing LR (for 'plateau' scheduler).
        scheduler_factor: Factor to multiply LR when reducing.
        min_lr: Minimum learning rate.
        reduction: Reduction used to turn the (n_offsprings,) offspring
            fitness into the scalar loss in differentiable mode:
            'mean' (default), 'sum', or 'min'. Only used when backprop is
            active; ignored in classical mode.
        live_selection: If True (default), selection routing carries gradient
            to the population via a per-generation re-evaluation of the current
            population. This is a real auxiliary objective pass, possibly at
            gradient-moved coordinates, but is excluded from the counted
            offspring-candidate budget ``n_evals`` by convention. If False,
            selection uses cached detached fitness (memetic; cheaper and exact
            objective-call accounting). Only used when backprop is active.

    Returns:
        Result object containing:
            - best_solution: Best solution found
            - best_fitness: Best fitness value
            - population: Final population
            - fitness: Final fitness values
            - n_evals: Counted initial/offspring evaluations (excluding live
              parent graph-reconstruction passes)
            - n_gen: Total generations
            - history: Convergence history (if save_history=True)
            - success: Whether target was reached
    
    Example:
        >>> # Pure classical EA (no learnable parameters, no probe, no backprop)
        >>> result = minimize(problem, GA(pop_size=100, differentiable=False,
        ...                               adaptive=False), seed=42)
        >>>
        >>> # Learn operator hyperparameters with classical dynamics
        >>> from evograd.operators import SBX, PolynomialMutation
        >>> algorithm = GA(
        ...     pop_size=100,
        ...     crossover=SBX(eta=15, differentiable=True),
        ...     mutation=PolynomialMutation(eta=20, differentiable=True),
        ...     differentiable=False,  # Population not updated via gradients
        ...     )
        >>> result = minimize(problem, algorithm, termination=MaxEvaluations(10000))
        >>>
        >>> # Full differentiable mode: learning rates resolve to the
        >>> # per-algorithm defaults automatically when the objective
        >>> # provides a gradient
        >>> algorithm = GA(pop_size=100, differentiable=True)
        >>> result = minimize(problem, algorithm)
        >>>
        >>> # Continue optimization with a different problem (e.g., surrogate -> true)
        >>> # First optimize with surrogate problem
        >>> pso = PSO(pop_size=100, differentiable=True)
        >>> result1 = minimize(surrogate_problem, pso, termination=MaxEvaluations(10000))
        >>> # Then continue with true problem (preserves velocities, personal bests)
        >>> result2 = minimize(true_problem, pso, termination=MaxEvaluations(500),
        ...                    initialize=False)

    Note:
        By default (initialize=True), the algorithm is initialized inside this
        function. Do not call algorithm.initialize() before passing to minimize().

        When initialize=False, the algorithm must have been previously initialized
        (e.g., from a prior minimize() call). This allows switching problems at
        runtime while preserving population state, velocities, and personal bests.
    """
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    
    # Set seed first for reproducibility
    if seed is not None:
        set_seed(seed)
    
    # Copy algorithm if requested
    if copy_algorithm:
        import copy
        algorithm = copy.deepcopy(algorithm)
    
    # Parse termination criteria
    termination = _parse_termination(termination)

    # Configure population-reduction schedules from the same evaluation budget
    # used by the optimisation loop. Callers may still override this explicitly.
    max_evals = _find_max_evaluations(termination)
    if (
        max_evals is not None
        and hasattr(algorithm, "set_max_evals")
        and getattr(algorithm, "_max_evals", None) is None
    ):
        algorithm.set_max_evals(max_evals)
    
    # Setup callbacks
    callbacks = _setup_callbacks(callback, verbose, save_history)

    # Initialize algorithm with problem (or continue with existing state)
    if initialize:
        algorithm.initialize(problem)
    else:
        # Continue with existing algorithm state but update problem reference
        # This preserves population, velocities, personal bests, etc.
        if not hasattr(algorithm, 'generation') or algorithm.generation == 0:
            raise ValueError(
                "initialize=False requires a previously initialized algorithm. "
                "Run minimize() with initialize=True first."
            )
        # Update problem reference and bounds
        algorithm.problem = problem
        algorithm.xl = problem.xl
        algorithm.xu = problem.xu

        # Re-evaluate the current population on the new problem so that
        # fitness values (including personal bests in PSO) are consistent
        # with the new objective.  Without this, stale fitness values from
        # the old problem prevent the algorithm from accepting any new
        # solutions (e.g., surrogate fitness ~0.003 vs ODE fitness ~200).
        with torch.no_grad():
            new_fitness = algorithm._evaluate(algorithm.population)
            algorithm.state.fitness = new_fitness
            algorithm.state.best_fitness = float('inf')
            algorithm.state.update_best(algorithm.population, new_fitness)

            # PSO: re-evaluate personal bests on the new problem
            if hasattr(algorithm, '_p_best') and hasattr(algorithm, '_p_best_fitness'):
                pb_fitness = algorithm._evaluate(algorithm._p_best)
                algorithm._p_best_fitness.copy_(pb_fitness)

        # Update termination budget to add to existing evaluations/generations
        _update_termination_budget(termination, algorithm)

    # Setup result builder
    builder = ResultBuilder()
    builder.set_problem(problem)
    builder.set_algorithm(algorithm)
    
    # -------------------------------------------------------------------------
    # Setup differentiable mode
    # -------------------------------------------------------------------------
    
    # Collect all learnable parameters (nn.Parameter with requires_grad=True)
    # learnable_params = [p for p in algorithm.parameters() if p.requires_grad]
    
    pop_params   = []
    hyper_params = []
    
    for name, p in algorithm.named_parameters():
        if not p.requires_grad:
            continue
        if name == "_population":
            pop_params.append(p)
        else:
            hyper_params.append(p)
    
    # The former -1 sentinel raises on every path, including optimizer=.
    for _lr_label, _lr_value in (("lr_pop", lr_pop), ("lr_hyper", lr_hyper)):
        if _lr_value is not None and _lr_value < 0:
            raise ValueError(
                f"{_lr_label}={_lr_value} is invalid: negative learning "
                "rates are not supported and the former -1 sentinel was "
                f"removed in 0.4.0. Omit the argument ({_lr_label}=None) to "
                "use the per-algorithm default, or pass 0 to disable "
                "gradient updates explicitly."
            )

    # The objective-gradient probe: one RNG-neutral evaluation at the bounds
    # midpoint, shared by both channels and run only when a channel needs it.
    # Tri-state: True / False / None (the probe itself could not run).
    _probe_cache: dict = {}

    def _objective_provides_gradient() -> Optional[bool]:
        if "result" not in _probe_cache:
            batch_hint = int(
                getattr(algorithm, "n_offsprings", None)
                or getattr(algorithm, "pop_size", 1)
                or 1
            )
            _probe_cache["result"] = _probe_objective_gradient(
                problem, batch_hint=batch_hint
            )
        return _probe_cache["result"]

    alg_defaults = _OPT_DEFAULTS.get(
        algorithm.__class__.__name__, _OPT_DEFAULTS["GA"]
    )

    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[Optional[torch.optim.lr_scheduler.LRScheduler]] = []
    lr_pop_eff: Optional[float] = None
    lr_hyper_eff: Optional[float] = None

    if optimizer is not None:
        # User-supplied optimizer(s): honored only when there is something to
        # optimise and the objective is not provably black-box (an
        # inconclusive probe honors the explicit request).
        if lr_pop is not None or lr_hyper is not None:
            warnings.warn(
                "lr_pop/lr_hyper are ignored when optimizer= is supplied; "
                "the provided optimizer's own learning rates are used.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not (pop_params or hyper_params):
            warnings.warn(
                "An optimizer was supplied but the algorithm exposes no "
                "learnable parameters (differentiable=False and "
                "adaptive=False); running the classical update instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif _objective_provides_gradient() is False:
            warnings.warn(
                "An optimizer was supplied but the objective does not provide "
                "a gradient (its output carries no grad_fn); running the "
                "classical update instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if isinstance(optimizer, (list, tuple)):
                optimizers.extend(list(optimizer))
            else:
                optimizers.append(optimizer)
        # Channel status reflects what the supplied optimizers actually cover.
        _covered = {
            id(p)
            for opt in optimizers
            for group in opt.param_groups
            for p in group["params"]
        }
        pop_grad_on = any(id(p) in _covered for p in pop_params)
        hyper_grad_on = any(id(p) in _covered for p in hyper_params)
    else:
        lr_pop_eff = _resolve_channel_lr(
            lr_pop,
            len(pop_params) > 0,
            alg_defaults["lr_pop"],
            "lr_pop",
            "algorithm.differentiable=False",
            _objective_provides_gradient,
        )
        lr_hyper_eff = _resolve_channel_lr(
            lr_hyper,
            len(hyper_params) > 0,
            alg_defaults["lr_hyper"],
            "lr_hyper",
            "adaptive=False and no differentiable operators",
            _objective_provides_gradient,
        )
        # Dimension-scale the population learning rate.
        if lr_pop_eff is not None:
            lr_pop_eff = lr_pop_eff / (problem.n_var ** 0.5)

        if lr_pop_eff is not None:
            optimizers.append(
                torch.optim.SGD(
                    pop_params, lr=lr_pop_eff,
                    momentum=alg_defaults["pop_momentum"],
                )
            )
        if lr_hyper_eff is not None:
            optimizers.append(torch.optim.Adam(hyper_params, lr=lr_hyper_eff))
        pop_grad_on = lr_pop_eff is not None
        hyper_grad_on = lr_hyper_eff is not None

    grad_clip_pop_eff = _resolve_channel_clip(
        grad_clip_pop, pop_grad_on, alg_defaults["grad_clip_pop"], "grad_clip_pop"
    )
    grad_clip_hyper_eff = _resolve_channel_clip(
        grad_clip_hyper, hyper_grad_on, alg_defaults["grad_clip_hyper"],
        "grad_clip_hyper",
    )

    use_backprop = len(optimizers) > 0

    if use_backprop:
        est_gens = _estimate_total_generations(termination, algorithm)
        for opt in optimizers:
            schedulers.append(
                _create_scheduler(
                    opt,
                    scheduler,
                    scheduler_patience,
                    scheduler_factor,
                    min_lr,
                    total_generations=est_gens,
                )
            )

    # Report which channels ended up gradient-driven. The dict is shared with
    # the first-generation diagnostic, which may still drop a channel whose
    # parameters turn out to receive no gradient (e.g. CMA-ES's population).
    gradient_channels = {"population": pop_grad_on, "hyperparams": hyper_grad_on}
    builder.add_extra("gradient_channels", gradient_channels)
    builder.add_extra("lr_pop_effective", lr_pop_eff)
    builder.add_extra("lr_hyper_effective", lr_hyper_eff)
    
    # -------------------------------------------------------------------------
    # Create callback state
    # -------------------------------------------------------------------------
    
    state = CallbackState(
        generation=algorithm.generation,
        n_evals=algorithm.n_evals,
        max_evals=getattr(termination, 'max_evals', None),
        max_generations=getattr(termination, 'max_gens', None),
        best_fitness=algorithm.best_fitness,
        best_solution=algorithm.best_solution,
        current_fitness=algorithm.fitness,
        current_population=algorithm.population,
        algorithm=algorithm,
        hyperparams=algorithm._get_hyperparams(),
    )
    
    # -------------------------------------------------------------------------
    # Optimisation loop
    # -------------------------------------------------------------------------
    
    builder.start()
    start_time = time.perf_counter()
    
    # Notify callbacks of start
    _call_callbacks(callbacks, "on_optimisation_start", state)
    
    # Reset termination state
    termination.reset()

    first_diff_step = True
    while not termination.should_terminate(algorithm):
        # Check callback early stopping
        if state.stop_optimisation:
            break

        # Generation start callback
        _call_callbacks(callbacks, "on_generation_start", state)

        # Run one generation
        if use_backprop:
            _step_differentiable(
                algorithm,
                optimizers,
                schedulers,
                pop_params,
                hyper_params,
                grad_clip_pop_eff,
                grad_clip_hyper_eff,
                reduction,
                live_selection,
                diagnose=first_diff_step,
                gradient_channels=gradient_channels,
            )
            first_diff_step = False
            if not optimizers:
                # Every optimizer was dropped by the no-gradient diagnostic;
                # continue with the classical update.
                use_backprop = False
        else:
            algorithm.step()
        
        # Update callback state
        state.generation = algorithm.generation
        state.n_evals = algorithm.n_evals
        state.best_fitness = algorithm.best_fitness
        state.best_solution = algorithm.best_solution
        state.current_fitness = algorithm.fitness
        state.current_population = algorithm.population
        state.hyperparams = algorithm._get_hyperparams()
        state.elapsed_time = time.perf_counter() - start_time
        
        # Generation end callback
        _call_callbacks(callbacks, "on_generation_end", state)
    
    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------
    
    # Determine success (check if TargetReached was met)
    success = _check_target_reached(termination, algorithm)
    
    # Build result
    builder.finish(algorithm, termination, success)

    if problem.has_constraints() and algorithm.best_solution is not None:
        feasible = bool(problem.is_feasible(algorithm.best_solution))
        builder.add_extra("best_feasible", feasible)
        builder.add_extra(
            "best_constraint_violation",
            float(problem.evaluate_constraints(algorithm.best_solution)["cv"]),
        )
        # Expose the raw objective value: best_fitness includes the constraint
        # penalty, so for an infeasible best it is an inflated composite rather
        # than the true objective. This lets callers recover the real value.
        builder.add_extra(
            "best_raw_objective",
            float(
                problem.evaluate(algorithm.best_solution.reshape(1, -1)).reshape(-1)[0]
            ),
        )
        if not feasible:
            warnings.warn(
                "The returned best solution is infeasible under the declared "
                "constraints; increase Problem.constraint_penalty.",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # Get history from callbacks
    if save_history:
        history = _collect_history(callbacks)
        builder.set_history(history)
    
    result = builder.build()
    
    # Final callback
    _call_callbacks(callbacks, "on_optimisation_end", state)
    
    return result


# =============================================================================
# Helper Functions
# =============================================================================

def _snapshot_rng_states(device: torch.device) -> dict:
    """Snapshot torch (CPU and, if relevant, device), NumPy and Python
    random-number states — the full scope seeded by ``set_seed``."""
    states = {
        "cpu": torch.get_rng_state(),
        "py": random.getstate(),
        "np": np.random.get_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        states["mps"] = torch.mps.get_rng_state()
    return states


def _restore_rng_states(states: dict) -> None:
    torch.set_rng_state(states["cpu"])
    random.setstate(states["py"])
    np.random.set_state(states["np"])
    if "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])
    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


def _probe_objective_gradient(problem: "Problem", batch_hint: int = 1) -> Optional[bool]:
    """Check whether the objective provides a gradient.

    Evaluates the same composite the differentiable loss is built from — the
    objective plus, for constrained problems, the exterior constraint penalty
    (mirroring ``Algorithm._evaluate``) — at the midpoint of the box bounds
    with a grad-requiring input, and checks the output for a ``grad_fn``. A
    detached / black-box composite (e.g. an objective that trains a model
    internally and returns a plain number, with no differentiable
    constraints) produces no graph, so gradient-based updates cannot receive
    any real signal from it.

    A single row is tried first; if the objective rejects it (e.g. it
    requires population-sized batches, or contains BatchNorm), the probe
    retries with ``batch_hint`` identical rows. If every attempt raises, the
    probe is *inconclusive* and returns ``None``: auto-resolution then stays
    classical, while explicitly requested learning rates / optimizers are
    honored.

    The probe is excluded from ``n_evals`` and RNG-neutral: torch, NumPy and
    Python random states are snapshotted and restored, so internally
    stochastic objectives do not perturb seeded reproducibility.
    """
    states = _snapshot_rng_states(problem.device)

    def _attempt(n_rows: int) -> bool:
        x = ((problem.xl + problem.xu) / 2.0).detach().clone().unsqueeze(0)
        if n_rows > 1:
            x = x.repeat(n_rows, 1)
        x.requires_grad_(True)
        with torch.enable_grad():
            fitness = problem.evaluate(x)
            if problem.has_constraints():
                violation = problem.evaluate_constraints(x)["cv"]
                fitness = fitness + problem.constraint_penalty * violation
        return fitness.grad_fn is not None

    try:
        batch_sizes = [1] + ([int(batch_hint)] if int(batch_hint) > 1 else [])
        last_exc: Optional[Exception] = None
        for n_rows in batch_sizes:
            try:
                return _attempt(n_rows)
            except Exception as exc:
                last_exc = exc
        warnings.warn(
            "The objective-gradient probe could not run "
            f"({type(last_exc).__name__}: {last_exc}); auto-resolution stays "
            "classical, explicitly requested learning rates / optimizers are "
            "honored unverified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    finally:
        _restore_rng_states(states)


def _resolve_channel_lr(
    value,
    channel_active: bool,
    default_value: float,
    label: str,
    flag_hint: str,
    objective_provides_gradient,
):
    """Resolve one gradient channel's learning rate.

    Sentinel convention (0.4.0):
        - ``None`` means "auto": if the channel exposes learnable parameters
          and the objective provides a gradient, the per-algorithm default is
          used; otherwise the channel stays classical (with a warning when
          the flag exposed learnable parameters; silently when it did not).
        - ``0`` explicitly disables the channel (with a warning).
        - Positive values are used verbatim, still subject to the objective
          providing a gradient: a probe that proves the objective black-box
          degrades to classical with a warning, while an *inconclusive*
          probe (the probe evaluation itself failed) honors the explicit
          request unverified.
        - Negative values raise: the former ``-1`` sentinel was removed.

    ``objective_provides_gradient`` is a tri-state callable returning
    True / False / None (inconclusive). Returns the effective learning rate,
    or ``None`` if the channel is off; the caller additionally scales the
    population learning rate by 1/sqrt(n_var).
    """
    if value is not None and value < 0:
        raise ValueError(
            f"{label}={value} is invalid: negative learning rates are not "
            "supported and the former -1 sentinel was removed in 0.4.0. "
            f"Omit the argument ({label}=None) to use the per-algorithm "
            "default, or pass 0 to disable gradient updates explicitly."
        )
    if not channel_active:
        if value is not None and value > 0:
            warnings.warn(
                f"{label}={value} was given but the algorithm exposes no "
                f"learnable parameters for this channel ({flag_hint}); "
                "gradient updates for it stay disabled.",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    if value is None:
        provides = objective_provides_gradient()
        if provides is True:
            return default_value
        if provides is None:
            # Inconclusive probe: auto stays conservative/classical (the
            # probe itself already warned with the failure detail).
            return None
        warnings.warn(
            f"The objective does not provide a gradient (its output carries "
            f"no grad_fn), so {label} cannot resolve to the per-algorithm "
            "default; gradient updates for this channel stay disabled.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if value == 0:
        warnings.warn(
            f"{label}=0 explicitly disables gradient updates for this channel.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    provides = objective_provides_gradient()
    if provides is False:
        warnings.warn(
            f"{label}={value} was requested but the objective does not "
            "provide a gradient (its output carries no grad_fn); gradient "
            "updates for this channel stay disabled.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    # provides is True, or inconclusive with an explicit request: honor it.
    return float(value)


def _resolve_channel_clip(value, channel_on: bool, default_value: float, label: str):
    """Resolve one channel's gradient-clipping threshold.

    ``None`` selects the per-algorithm default when the channel is
    gradient-driven; ``0`` disables clipping; positive values are used as-is;
    negative values raise (the former ``-1`` sentinel was removed).
    """
    if value is not None and value < 0:
        raise ValueError(
            f"{label}={value} is invalid: the former -1 sentinel was removed "
            f"in 0.4.0. Omit the argument ({label}=None) for the per-algorithm "
            "default, or pass 0 to disable clipping."
        )
    if not channel_on:
        return None
    if value is None:
        return default_value
    if value == 0:
        return None
    return float(value)

def _parse_termination(termination: Optional[Termination]) -> Termination:
    """
    Parse termination argument into Termination instance.

    Args:
        termination: Termination instance or None for default.

    Returns:
        Termination instance.
    """
    if termination is None:
        return default_termination()

    if isinstance(termination, Termination):
        return termination

    raise TypeError(
        f"termination must be a Termination instance or None. "
        f"Got {type(termination).__name__}. "
        f"Example: termination=MaxEvaluations(10000)"
    )


def _find_max_evaluations(termination: Termination) -> Optional[int]:
    """Extract the first MaxEvaluations budget from a termination tree."""
    from evograd.core.termination import MaxEvaluations, TerminationCollection

    if isinstance(termination, MaxEvaluations):
        return termination.max_evals
    if isinstance(termination, TerminationCollection):
        for criterion in termination.criteria:
            value = _find_max_evaluations(criterion)
            if value is not None:
                return value
    return None


def _update_termination_budget(
    termination: Termination,
    algorithm: "Algorithm",
) -> None:
    """
    Update termination budget when continuing optimization (initialize=False).

    Adds the current algorithm's evaluations/generations to the termination
    criterion's budget, so the new budget is additive rather than absolute.

    Args:
        termination: The termination criterion to update.
        algorithm: The algorithm with current evaluation/generation counts.
    """
    from evograd.core.termination import (
        MaxEvaluations,
        MaxGenerations,
        TerminationCollection,
    )

    def _update_single(term: Termination) -> None:
        if isinstance(term, MaxEvaluations):
            # Add current evaluations to budget
            term.max_evals += algorithm.n_evals
        elif isinstance(term, MaxGenerations):
            # Add current generations to budget
            term.max_gens += algorithm.generation

    if isinstance(termination, TerminationCollection):
        # Update all criteria in the collection
        for criterion in termination.criteria:
            _update_single(criterion)
    else:
        _update_single(termination)


def _setup_callbacks(
    callback: Optional[Union[Callback, List[Callback]]],
    verbose: bool,
    save_history: bool,
) -> List[Callback]:
    """Setup callback list with defaults."""
    callbacks = []
    
    # Always include history callback if saving history
    if save_history:
        callbacks.append(HistoryCallback())
    
    # Add user callbacks
    if callback is not None:
        if isinstance(callback, list):
            callbacks.extend(callback)
        elif isinstance(callback, CallbackList):
            callbacks.extend(callback.callbacks)
        else:
            callbacks.append(callback)
    
    # Add print callback if verbose (and not already present)
    if verbose:
        has_print = any(isinstance(cb, PrintCallback) for cb in callbacks)
        if not has_print:
            callbacks.append(PrintCallback(every=1))
    
    return callbacks


def _call_callbacks(callbacks: List[Callback], method: str, state: CallbackState) -> None:
    """Call a method on all callbacks."""
    for cb in callbacks:
        getattr(cb, method)(state)


def _collect_history(callbacks: List[Callback]) -> Dict[str, List[Any]]:
    """Collect history from HistoryCallback if present."""
    for cb in callbacks:
        if isinstance(cb, HistoryCallback):
            return cb.to_dict()
    return {}


def _check_target_reached(termination: Termination, algorithm: Algorithm) -> bool:
    """Recursively check whether any target criterion was reached."""
    if isinstance(termination, TargetReached):
        best = algorithm.best_fitness
        if termination.minimize:
            return best <= termination.target_fitness
        else:
            return best >= termination.target_fitness
    
    if isinstance(termination, TerminationCollection):
        return any(
            _check_target_reached(criterion, algorithm)
            for criterion in termination.criteria
        )
    
    return False


def _estimate_total_generations(termination: Termination, algorithm: "Algorithm") -> int:
    """
    Estimate the total number of generations from the termination criterion.

    Used to set ``T_max`` for the cosine-annealing scheduler so that the
    learning-rate schedule matches the actual optimisation budget.

    Falls back to 10 000 if no budget can be inferred.
    """
    from evograd.core.termination import MaxEvaluations, MaxGenerations, TerminationCollection

    def _extract(term: Termination) -> Optional[int]:
        if isinstance(term, MaxGenerations):
            return term.max_gens
        if isinstance(term, MaxEvaluations):
            pop = max(algorithm.pop_size, 1)
            return term.max_evals // pop
        return None

    if isinstance(termination, TerminationCollection):
        for criterion in termination.criteria:
            val = _extract(criterion)
            if val is not None:
                return val

    val = _extract(termination)
    if val is not None:
        return val

    return 10_000  # safe fallback


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Optional[str],
    patience: int,
    factor: float,
    min_lr: float,
    total_generations: int = 10_000,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler.

    Args:
        total_generations: Estimated total generations for the optimisation
            run. Used as ``T_max`` for the cosine-annealing scheduler.
    """
    if scheduler_type is None:
        return None

    scheduler_type = scheduler_type.lower()

    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=patience,
            gamma=factor,
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_generations,
            eta_min=min_lr,
        )
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=factor ** (1.0 / patience),
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Use 'plateau', 'step', 'cosine', or 'exponential'."
        )


def _step_differentiable(
    algorithm: Algorithm,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[Optional[torch.optim.lr_scheduler.LRScheduler]],
    pop_params: Optional[List],
    hyper_params: Optional[List],
    grad_clip_pop: Optional[float],
    grad_clip_hyper: Optional[float],
    reduction: str = "mean",
    live_selection: bool = True,
    diagnose: bool = False,
    gradient_channels: Optional[dict] = None,
) -> float:
    """
    Perform one generation step with gradient-based updates.
    
    Gradients automatically flow to all nn.Parameter tensors:
    - Population (if algorithm.differentiable=True)
    - Operator params (if operator.differentiable=True)
    - Adaptive coefficients (if adaptive=True)
    
    Args:
        algorithm: The algorithm instance.
        optimizer: PyTorch optimizer.
        scheduler: Optional LR scheduler.
        pop_params: population parameters.
        hyper_params: hyperparam parameters.
        grad_clip_pop: Maximum gradient norm for clipping the population gradient.
        grad_clip_hyper: Maximum gradient norm for clipping the hyperparam gradient.
    
    Returns:
        Loss value (best fitness).
    """
    if algorithm.differentiable and isinstance(algorithm.population, torch.nn.Parameter):
        algorithm.population.requires_grad_(True)
    
    # Clear every algorithm gradient, including intentionally non-optimised
    # parameter groups, so partial optimisation cannot accumulate stale grads.
    algorithm.zero_grad(set_to_none=True)

    # Zero optimizer-owned gradients as well (supports external parameters).
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    
    # Forward pass (builds computation graph)
    loss = algorithm.forward(reduction=reduction, live_selection=live_selection)
    
    # Backward pass
    loss.backward()

    # First-generation diagnostic: a channel whose parameters receive no
    # gradient at all cannot be moved by its optimizer (e.g. CMA-ES's
    # population Parameter never enters the loss graph -- offspring are
    # resampled from the mean). Drop such optimizers instead of silently
    # stepping them for the whole run.
    if diagnose and optimizers:
        pop_ids = {id(p) for p in (pop_params or [])}
        hyper_ids = {id(p) for p in (hyper_params or [])}
        kept_opts: List[torch.optim.Optimizer] = []
        kept_scheds: List[Optional[torch.optim.lr_scheduler.LRScheduler]] = []
        for opt, sch in zip(optimizers, schedulers):
            params = [p for group in opt.param_groups for p in group["params"]]
            if params and all(p.grad is None for p in params):
                param_ids = {id(p) for p in params}
                if param_ids <= pop_ids:
                    label = "population"
                elif param_ids <= hyper_ids:
                    label = "hyperparameter"
                else:
                    label = "supplied"
                warnings.warn(
                    f"The {label} optimizer received no gradient on the first "
                    "generation (every parameter's grad is None); it is "
                    "dropped for the remainder of the run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            kept_opts.append(opt)
            kept_scheds.append(sch)
        optimizers[:] = kept_opts
        schedulers[:] = kept_scheds
        # Recompute channel status from the optimizers that actually remain,
        # so result.extra['gradient_channels'] stays truthful for mixed or
        # partially covering optimizers as well.
        if gradient_channels is not None:
            kept_ids = {
                id(p)
                for opt in optimizers
                for group in opt.param_groups
                for p in group["params"]
            }
            gradient_channels["population"] = bool(kept_ids & pop_ids)
            gradient_channels["hyperparams"] = bool(kept_ids & hyper_ids)

    # Gradient clipping
    if grad_clip_pop is not None and pop_params:
        torch.nn.utils.clip_grad_norm_(pop_params, grad_clip_pop)

    if grad_clip_hyper is not None and hyper_params:
        torch.nn.utils.clip_grad_norm_(hyper_params, grad_clip_hyper)
            
    # Optimizer step
    for opt in optimizers:
        opt.step()

    # Projected step: the gradient update can push the population Parameter
    # outside [xl, xu]. Clamp it back before it is committed / re-evaluated so
    # differentiable runs never evaluate or return out-of-bounds solutions
    # (mirrors the per-trial clamping every operator applies). This is a no-op
    # when the step stayed in bounds, and only the decision-variable population
    # is clamped — hyperparameters (e.g. CMA-ES mean, learnable rates) are not.
    if (
        isinstance(algorithm.population, torch.nn.Parameter)
        and getattr(algorithm, "xl", None) is not None
        and getattr(algorithm, "xu", None) is not None
    ):
        with torch.no_grad():
            algorithm.population.data.copy_(
                clamp_to_bounds(algorithm.population.data, algorithm.xl, algorithm.xu)
            )

    # Commit evolutionary changes
    old_population_param = algorithm.population if isinstance(algorithm.population, torch.nn.Parameter) else None
    algorithm.update_state()

    # Population-size-changing algorithms (notably differentiable L-SHADE)
    # may replace their Parameter. Keep existing optimisers attached to the
    # live population for the next generation.
    new_population_param = algorithm.population if isinstance(algorithm.population, torch.nn.Parameter) else None
    if old_population_param is not None and new_population_param is not old_population_param:
        for opt in optimizers:
            for group in opt.param_groups:
                group["params"] = [
                    new_population_param if p is old_population_param else p
                    for p in group["params"]
                ]
            opt.state.pop(old_population_param, None)
        if pop_params is not None:
            pop_params[:] = [new_population_param]
    
    # Scheduler step
    for sch in schedulers:
        if sch is None:
            continue
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(loss.item())
        else:
            sch.step()
    
    return float(loss.detach())
