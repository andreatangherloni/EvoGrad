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
    ... )
    >>> 
    >>> print(f"Best fitness: {result.best_fitness}")
    >>> print(f"Best solution: {result.best_solution}")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from evograd.core.result import Result, ResultBuilder
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


def minimize(
    problem: Problem,
    algorithm: Algorithm,
    termination: Optional[Termination] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    callback: Optional[Union[Callback, List[Callback]]] = None,
    copy_algorithm: bool = False,
    save_history: bool = True,
    # Differentiable mode options
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.01,
    grad_clip: Optional[float] = None,
    scheduler: Optional[str] = "plateau",
    scheduler_patience: int = 50,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> Result:
    """
    Minimise an objective function using a population-based algorithm.
    
    This function initialises the algorithm with the problem and runs
    the optimisation loop until termination criteria are met. Follows
    pymoo's interface style.
    
    Differentiable Mode
    -------------------
    EvoGrad automatically detects learnable parameters (nn.Parameter with
    requires_grad=True) and uses backpropagation to update them. This covers:
    
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
        
        # Differentiable mode options (used if learnable params exist):
        optimizer: PyTorch optimizer for gradient-based updates.
            If None, Adam is used with specified lr.
        lr: Learning rate for gradient-based updates (default: 0.01).
        grad_clip: Maximum gradient norm for clipping (None = no clipping).
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
    
    Returns:
        Result object containing:
            - best_solution: Best solution found
            - best_fitness: Best fitness value
            - population: Final population
            - fitness: Final fitness values
            - n_evals: Total evaluations
            - n_gen: Total generations
            - history: Convergence history (if save_history=True)
            - success: Whether target was reached
    
    Example:
        >>> # Basic usage (classical EA)
        >>> result = minimize(problem, GA(pop_size=100), seed=42)
        >>> 
        >>> # Learn operator hyperparameters with classical dynamics
        >>> from evograd.operators import SBX, PolynomialMutation
        >>> algorithm = GA(
        ...     pop_size=100,
        ...     crossover=SBX(eta=15, differentiable=True),
        ...     mutation=PolynomialMutation(eta=20, differentiable=True),
        ...     differentiable=False,  # Population not updated via gradients
        ... )
        >>> result = minimize(problem, algorithm, termination=MaxEvaluations(10000))
        >>> 
        >>> # Full differentiable mode
        >>> algorithm = GA(pop_size=100, differentiable=True)
        >>> result = minimize(problem, algorithm, lr=0.01, grad_clip=1.0)
    
    Note:
        The algorithm is initialized inside this function. Do not call
        algorithm.initialize() before passing to minimize().
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
    
    # Setup callbacks
    callbacks = _setup_callbacks(callback, verbose, save_history)
    
    # Initialize algorithm with problem
    algorithm.initialize(problem)
    
    # Setup result builder
    builder = ResultBuilder()
    builder.set_problem(problem)
    builder.set_algorithm(algorithm)
    
    # -------------------------------------------------------------------------
    # Setup differentiable mode
    # -------------------------------------------------------------------------
    
    # Collect all learnable parameters (nn.Parameter with requires_grad=True)
    learnable_params = [p for p in algorithm.parameters() if p.requires_grad]
    use_backprop = len(learnable_params) > 0
    lr_scheduler = None
    
    if use_backprop:
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(learnable_params, lr=lr)
        
        # Create LR scheduler
        lr_scheduler = _create_scheduler(
            optimizer,
            scheduler,
            scheduler_patience,
            scheduler_factor,
            min_lr,
        )
    else:
        optimizer = None
    
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
                optimizer,
                lr_scheduler,
                grad_clip,
            )
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
    """Check if target was reached (for TargetReached termination)."""
    if isinstance(termination, TargetReached):
        best = algorithm.best_fitness
        if termination.minimize:
            return best <= termination.target_fitness
        else:
            return best >= termination.target_fitness
    
    if isinstance(termination, TerminationCollection):
        for criterion in termination.criteria:
            if isinstance(criterion, TargetReached):
                best = algorithm.best_fitness
                if criterion.minimize:
                    if best <= criterion.target_fitness:
                        return True
                else:
                    if best >= criterion.target_fitness:
                        return True
    
    return False


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Optional[str],
    patience: int,
    factor: float,
    min_lr: float,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler."""
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
            T_max=10000,
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
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    grad_clip: Optional[float],
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
        grad_clip: Maximum gradient norm for clipping.
    
    Returns:
        Loss value (best fitness).
    """
    # Zero gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass (builds computation graph)
    loss = algorithm.forward()
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(algorithm.parameters(), grad_clip)
    
    # Optimizer step
    optimizer.step()
    
    # Commit evolutionary changes
    algorithm.update_state()
    
    # Scheduler step
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss.item())
        else:
            scheduler.step()
    
    return float(loss)



# """
# Minimisation function for EvoGrad optimisation.

# This module provides the main entry point for running optimisation,
# following pymoo's interface style where algorithm initialisation
# happens inside the minimize function.

# Example:
#     >>> from evograd.core import minimize, Problem, MaxEvaluations
#     >>> from evograd.algorithms import GA
#     >>> 
#     >>> # Define problem
#     >>> problem = Problem(
#     ...     objective=lambda x: (x**2).sum(dim=-1),
#     ...     n_var=30,
#     ...     xl=-100.0,
#     ...     xu=100.0,
#     ... )
#     >>> 
#     >>> # Create algorithm (not initialized)
#     >>> algorithm = GA(pop_size=100, eliminate_duplicates=True)
#     >>> 
#     >>> # Run optimisation
#     >>> result = minimize(
#     ...     problem,
#     ...     algorithm,
#     ...     termination=MaxEvaluations(10000),
#     ...     seed=42,
#     ...     verbose=True,
#     ... )
#     >>> 
#     >>> print(f"Best fitness: {result.best_fitness}")
#     >>> print(f"Best solution: {result.best_solution}")
# """

# from __future__ import annotations

# import time
# from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# import torch

# from evograd.core.result import Result, ResultBuilder
# from evograd.core.termination import (
#     MaxEvaluations,
#     MaxGenerations,
#     Termination,
#     TerminationCollection,
#     TargetReached,
#     default_termination,
# )
# from evograd.utils.callbacks import (
#     Callback,
#     CallbackList,
#     CallbackState,
#     HistoryCallback,
#     PrintCallback,
# )
# from evograd.utils.device import set_seed

# if TYPE_CHECKING:
#     from evograd.core.algorithm import Algorithm
#     from evograd.core.problem import Problem

# __all__ = [
#     "minimize",
# ]


# def minimize(
#     problem: Problem,
#     algorithm: Algorithm,
#     termination: Optional[Union[Termination, Tuple[str, int]]] = None,
#     seed: Optional[int] = None,
#     verbose: bool = True,
#     callback: Optional[Union[Callback, List[Callback]]] = None,
#     copy_algorithm: bool = False,
#     save_history: bool = True,
#     return_least_infeasible: bool = False,
#     # Differentiable mode options
#     optimizer: Optional[torch.optim.Optimizer] = None,
#     lr: float = 0.01,
#     grad_clip: Optional[float] = None,
#     scheduler: Optional[str] = "plateau",
#     scheduler_patience: int = 50,
#     scheduler_factor: float = 0.5,
#     min_lr: float = 1e-6,
# ) -> Result:
#     """
#     Minimise an objective function using a population-based algorithm.
    
#     This function initialises the algorithm with the problem and runs
#     the optimisation loop until termination criteria are met. Follows
#     pymoo's interface style.
    
#     Args:
#         problem: Problem instance defining the objective function,
#             bounds, and constraints.
#         algorithm: Algorithm instance (e.g., GA, DE, PSO, CMAES).
#             Will be initialized inside this function.
#         termination: When to stop optimisation. Can be:
#             - Termination instance (e.g., MaxEvaluations(10000))
#             - Tuple of (criterion, value), e.g., ('n_evals', 10000)
#             - None: Uses default (10000 evaluations)
#         seed: Random seed for reproducibility. Applied before
#             algorithm initialisation.
#         verbose: If True, print progress during optimisation.
#         callback: Single Callback or list of Callbacks for monitoring.
#             HistoryCallback is always included automatically.
#         copy_algorithm: If True, create a copy of the algorithm to
#             preserve the original. Default False.
#         save_history: If True (default), save convergence history
#             in result. Set False to reduce memory for long runs.
#         return_least_infeasible: If True and no feasible solution found,
#             return the least infeasible solution.
        
#         # Differentiable mode options (only used if algorithm.differentiable=True):
#         optimizer: PyTorch optimizer for gradient-based updates.
#             If None, Adam is used with specified lr.
#         lr: Learning rate for gradient-based updates (default: 0.01).
#         grad_clip: Maximum gradient norm for clipping (None = no clipping).
#         scheduler: Learning rate scheduler type:
#             - 'plateau': Reduce on plateau (default)
#             - 'step': Reduce every N generations
#             - 'cosine': Cosine annealing
#             - None: No scheduler
#         scheduler_patience: Generations without improvement before
#             reducing LR (for 'plateau' scheduler).
#         scheduler_factor: Factor to multiply LR when reducing.
#         min_lr: Minimum learning rate.
    
#     Returns:
#         Result object containing:
#             - best_solution: Best solution found
#             - best_fitness: Best fitness value
#             - population: Final population
#             - fitness: Final fitness values
#             - n_evals: Total evaluations
#             - n_gen: Total generations
#             - history: Convergence history (if save_history=True)
#             - success: Whether target was reached
    
#     Example:
#         >>> # Basic usage
#         >>> result = minimize(problem, GA(pop_size=100), seed=42)
#         >>> 
#         >>> # With termination criteria
#         >>> result = minimize(
#         ...     problem, algorithm,
#         ...     termination=MaxEvaluations(50000) | TargetReached(1e-6),
#         ... )
#         >>> 
#         >>> # With callbacks
#         >>> from evograd.utils.callbacks import EarlyStoppingCallback
#         >>> result = minimize(
#         ...     problem, algorithm,
#         ...     callback=[EarlyStoppingCallback(patience=100)],
#         ... )
#         >>> 
#         >>> # Differentiable mode with custom optimizer
#         >>> optimizer = torch.optim.Adam(algorithm.parameters(), lr=0.001)
#         >>> result = minimize(
#         ...     problem, algorithm,
#         ...     optimizer=optimizer,
#         ...     grad_clip=1.0,
#         ... )
    
#     Note:
#         The algorithm is initialized inside this function. Do not call
#         algorithm.initialize() before passing to minimize().
#     """
#     # -------------------------------------------------------------------------
#     # Setup
#     # -------------------------------------------------------------------------
    
#     # Set seed first for reproducibility
#     if seed is not None:
#         set_seed(seed)
    
#     # Copy algorithm if requested
#     if copy_algorithm:
#         import copy
#         algorithm = copy.deepcopy(algorithm)
    
#     # Parse termination criteria
#     termination = _parse_termination(termination)
    
#     # Setup callbacks
#     callbacks = _setup_callbacks(callback, verbose, save_history)
    
#     # Initialize algorithm with problem
#     algorithm.initialize(problem)
    
#     # Setup result builder
#     builder = ResultBuilder()
#     builder.set_problem(problem)
#     builder.set_algorithm(algorithm)
    
#     # -------------------------------------------------------------------------
#     # Setup differentiable mode
#     # -------------------------------------------------------------------------
    
#     if algorithm.differentiable:
#         # Create optimizer if not provided
#         if optimizer is None:
#             optimizer = torch.optim.Adam(algorithm.parameters(), lr=lr)
        
#         # Create LR scheduler
#         lr_scheduler = _create_scheduler(
#             optimizer,
#             scheduler,
#             scheduler_patience,
#             scheduler_factor,
#             min_lr,
#         )
#     else:
#         optimizer = None
#         lr_scheduler = None
    
#     # -------------------------------------------------------------------------
#     # Create callback state
#     # -------------------------------------------------------------------------
    
#     state = CallbackState(
#         generation=algorithm.generation,
#         n_evals=algorithm.n_evals,
#         max_evals=getattr(termination, 'max_evals', None),
#         max_generations=getattr(termination, 'max_gens', None),
#         best_fitness=algorithm.best_fitness,
#         best_solution=algorithm.best_solution,
#         current_fitness=algorithm.fitness,
#         current_population=algorithm.population,
#         algorithm=algorithm,
#         hyperparams=algorithm._get_hyperparams(),
#     )
    
#     # -------------------------------------------------------------------------
#     # Optimisation loop
#     # -------------------------------------------------------------------------
    
#     builder.start()
#     start_time = time.perf_counter()
    
#     # Notify callbacks of start
#     _call_callbacks(callbacks, "on_optimisation_start", state)
    
#     # Reset termination state
#     termination.reset()
    
#     while not termination.should_terminate(algorithm):
#         # Check callback early stopping
#         if state.stop_optimisation:
#             break
        
#         # Generation start callback
#         _call_callbacks(callbacks, "on_generation_start", state)
        
#         # Run one generation
#         if algorithm.differentiable:
#             _step_differentiable(
#                 algorithm,
#                 optimizer,
#                 lr_scheduler,
#                 grad_clip,
#             )
#         else:
#             algorithm.step()
        
#         # Update callback state
#         state.generation = algorithm.generation
#         state.n_evals = algorithm.n_evals
#         state.best_fitness = algorithm.best_fitness
#         state.best_solution = algorithm.best_solution
#         state.current_fitness = algorithm.fitness
#         state.current_population = algorithm.population
#         state.hyperparams = algorithm._get_hyperparams()
#         state.elapsed_time = time.perf_counter() - start_time
        
#         # Generation end callback
#         _call_callbacks(callbacks, "on_generation_end", state)
    
#     # -------------------------------------------------------------------------
#     # Finalize
#     # -------------------------------------------------------------------------
    
#     # Determine success (check if TargetReached was met)
#     success = _check_target_reached(termination, algorithm)
    
#     # Build result
#     builder.finish(algorithm, termination, success)
    
#     # Get history from callbacks
#     if save_history:
#         history = _collect_history(callbacks)
#         builder.set_history(history)
    
#     result = builder.build()
    
#     # Final callback
#     _call_callbacks(callbacks, "on_optimisation_end", state)
    
#     return result


# # =============================================================================
# # Helper Functions
# # =============================================================================

# def _parse_termination(
#     termination: Optional[Union[Termination, Tuple[str, int]]]
# ) -> Termination:
#     """
#     Parse termination argument into Termination instance.
    
#     Args:
#         termination: Various termination specifications.
    
#     Returns:
#         Termination instance.
#     """
#     if termination is None:
#         return default_termination()
    
#     if isinstance(termination, Termination):
#         return termination
    
#     if isinstance(termination, tuple):
#         criterion, value = termination
#         criterion = criterion.lower()
        
#         if criterion in ('n_eval', 'n_evals', 'evals', 'evaluations'):
#             return MaxEvaluations(value)
#         elif criterion in ('n_gen', 'n_gens', 'gen', 'generations'):
#             return MaxGenerations(value)
#         else:
#             raise ValueError(
#                 f"Unknown termination criterion: {criterion}. "
#                 f"Use 'n_evals' or 'n_gen'."
#             )
    
#     raise TypeError(
#         f"termination must be Termination, tuple, or None. "
#         f"Got {type(termination)}."
#     )


# def _setup_callbacks(
#     callback: Optional[Union[Callback, List[Callback]]],
#     verbose: bool,
#     save_history: bool,
# ) -> List[Callback]:
#     """Setup callback list with defaults."""
#     callbacks = []
    
#     # Always include history callback if saving history
#     if save_history:
#         callbacks.append(HistoryCallback())
    
#     # Add user callbacks
#     if callback is not None:
#         if isinstance(callback, list):
#             callbacks.extend(callback)
#         elif isinstance(callback, CallbackList):
#             callbacks.extend(callback.callbacks)
#         else:
#             callbacks.append(callback)
    
#     # Add print callback if verbose (and not already present)
#     if verbose:
#         has_print = any(isinstance(cb, PrintCallback) for cb in callbacks)
#         if not has_print:
#             callbacks.append(PrintCallback(every=1))
    
#     return callbacks


# def _call_callbacks(callbacks: List[Callback], method: str, state: CallbackState) -> None:
#     """Call a method on all callbacks."""
#     for cb in callbacks:
#         getattr(cb, method)(state)


# def _collect_history(callbacks: List[Callback]) -> Dict[str, List[Any]]:
#     """Collect history from HistoryCallback if present."""
#     for cb in callbacks:
#         if isinstance(cb, HistoryCallback):
#             return cb.to_dict()
#     return {}


# def _check_target_reached(termination: Termination, algorithm: Algorithm) -> bool:
#     """Check if target was reached (for TargetReached termination)."""
#     if isinstance(termination, TargetReached):
#         best = algorithm.best_fitness
#         if termination.minimize:
#             return best <= termination.target_fitness
#         else:
#             return best >= termination.target_fitness
    
#     if isinstance(termination, TerminationCollection):
#         for criterion in termination.criteria:
#             if isinstance(criterion, TargetReached):
#                 best = algorithm.best_fitness
#                 if criterion.minimize:
#                     if best <= criterion.target_fitness:
#                         return True
#                 else:
#                     if best >= criterion.target_fitness:
#                         return True
    
#     return False


# def _create_scheduler(
#     optimizer: torch.optim.Optimizer,
#     scheduler_type: Optional[str],
#     patience: int,
#     factor: float,
#     min_lr: float,
# ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
#     """Create learning rate scheduler."""
#     if scheduler_type is None:
#         return None
    
#     scheduler_type = scheduler_type.lower()
    
#     if scheduler_type == "plateau":
#         return torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="min",
#             factor=factor,
#             patience=patience,
#             min_lr=min_lr,
#         )
#     elif scheduler_type == "step":
#         return torch.optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=patience,
#             gamma=factor,
#         )
#     elif scheduler_type == "cosine":
#         # Cosine needs total steps, use large default
#         return torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=10000,
#             eta_min=min_lr,
#         )
#     elif scheduler_type == "exponential":
#         return torch.optim.lr_scheduler.ExponentialLR(
#             optimizer,
#             gamma=factor ** (1.0 / patience),
#         )
#     else:
#         raise ValueError(
#             f"Unknown scheduler type: {scheduler_type}. "
#             f"Use 'plateau', 'step', 'cosine', or 'exponential'."
#         )


# def _step_differentiable(
#     algorithm: Algorithm,
#     optimizer: torch.optim.Optimizer,
#     scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
#     grad_clip: Optional[float],
# ) -> float:
#     """
#     Perform one differentiable generation step.
    
#     Args:
#         algorithm: The algorithm instance.
#         optimizer: PyTorch optimizer.
#         scheduler: Optional LR scheduler.
#         grad_clip: Maximum gradient norm for clipping.
    
#     Returns:
#         Best fitness after this step.
#     """
#     # Zero gradients
#     optimizer.zero_grad(set_to_none=True)
    
#     # Forward pass (builds computation graph)
#     loss = algorithm.forward()
    
#     # Backward pass
#     loss.backward()
    
#     # Gradient clipping
#     if grad_clip is not None:
#         torch.nn.utils.clip_grad_norm_(algorithm.parameters(), grad_clip)
    
#     # Optimizer step
#     optimizer.step()
    
#     # Commit evolutionary changes
#     algorithm.update_state()
    
#     # Scheduler step
#     if scheduler is not None:
#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(loss.item())
#         else:
#             scheduler.step()
    
#     return float(loss)
