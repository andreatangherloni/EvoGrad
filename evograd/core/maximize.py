"""
Maximisation function for EvoGrad optimisation.

This module provides a maximisation interface that internally
converts to minimisation by negating the objective function.

Example:
    >>> from evograd.core.problem import Problem
    >>> from evograd.core.maximize import maximize
    >>> from evograd.core.termination import MaxEvaluations
    >>> from evograd.algorithms import GA
    >>> 
    >>> # Define problem (we want to MAXIMIZE this)
    >>> problem = Problem(
    ...     objective=lambda x: torch.sin(x).sum(dim=-1),  # Want to maximize
    ...     n_var=30,
    ...     xl=-3.14,
    ...     xu=3.14,
    ... )
    >>> 
    >>> # Create algorithm
    >>> algorithm = GA(pop_size=100)
    >>> 
    >>> # Run maximisation
    >>> result = maximize(
    ...     problem,
    ...     algorithm,
    ...     termination=MaxEvaluations(10000),
    ...     seed=42,
    ... )
    >>> 
    >>> # Note: result.best_fitness is the ACTUAL fitness (not negated)
    >>> print(f"Best (max) fitness: {result.best_fitness}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import Tensor

from evograd.utils.callbacks import Callback
from evograd.core.minimize import minimize
from evograd.core.problem import Problem
from evograd.core.result import Result
from evograd.core.termination import Termination

if TYPE_CHECKING:
    from evograd.core.algorithm import Algorithm

__all__ = [
    "maximize",
]


class _NegatedProblem(Problem):
    """
    Wrapper that negates the objective function for maximisation.
    
    This is an internal class used by maximize() to convert
    maximisation to minimisation.
    
    Instead of calling super().__init__() with all parameters,
    we directly copy the relevant attributes from the original
    problem to avoid parameter validation issues.
    """
    
    def __init__(self, problem: Problem) -> None:
        # Initialize nn.Module directly (skip Problem.__init__)
        torch.nn.Module.__init__(self)
        
        # Copy all relevant attributes from original problem
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self._objective = None  # We override _evaluate
        self.name = f"Negated({problem.name})" if problem.name else "Negated"
        self.device = problem.device
        self.dtype = problem.dtype
        
        # Register bounds as buffers (copy from original)
        self.register_buffer("xl", problem.xl.clone())
        self.register_buffer("xu", problem.xu.clone())
        
        # Copy constraint info
        self._constraints = problem._constraints.copy() if hasattr(problem, '_constraints') else []
        self.n_ieq_constr = problem.n_ieq_constr
        self.n_eq_constr = problem.n_eq_constr
        self.n_constr = problem.n_constr
        
        # Store reference to original problem
        self._original_problem = problem
    
    def _evaluate(self, x: Tensor) -> Tensor:
        """Negate the original objective."""
        return -self._original_problem._evaluate(x)
    
    @property
    def original_problem(self) -> Problem:
        """Access the original (non-negated) problem."""
        return self._original_problem


class _NegatedResult(Result):
    """
    Result wrapper that negates fitness values back.
    
    Used to return the actual (non-negated) fitness values
    to the user after maximisation.
    """
    
    @classmethod
    def from_minimization_result(cls, result: Result) -> Result:
        """Create a maximisation result from a minimisation result."""
        # Negate fitness values
        best_fitness = -result.best_fitness
        
        fitness = result.fitness
        if fitness is not None:
            fitness = -fitness
        
        # Negate history values
        history = result.history.copy()
        if "best_fitness" in history:
            history["best_fitness"] = [-f for f in history["best_fitness"]]
        if "mean_fitness" in history:
            history["mean_fitness"] = [-f for f in history["mean_fitness"]]
        
        return Result(
            best_solution=result.best_solution,
            best_fitness=best_fitness,
            population=result.population,
            fitness=fitness,
            n_evals=result.n_evals,
            n_gen=result.n_gen,
            success=result.success,
            termination_reason=result.termination_reason,
            history=history,
            hyperparams=result.hyperparams,
            algorithm_state=result.algorithm_state,
            problem_name=result.problem_name,
            algorithm_name=result.algorithm_name,
            start_time=result.start_time,
            end_time=result.end_time,
            elapsed_time=result.elapsed_time,
            device=result.device,
            extra=result.extra,
        )


def maximize(
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
    Maximise an objective function using a population-based algorithm.
    
    This function converts maximisation to minimisation by negating
    the objective, runs the optimisation, and then negates the results
    back to return actual fitness values.
    
    Args:
        problem: Problem instance defining the objective function,
            bounds, and constraints. The objective will be MAXIMISED.
        algorithm: Algorithm instance (e.g., GA, DE, PSO, CMAES).
            Will be initialized inside this function.
        termination: When to stop optimisation. Must be a Termination
            instance (e.g., MaxEvaluations(10000)). If None, uses
            default (10000 evaluations).
        seed: Random seed for reproducibility.
        verbose: If True, print progress during optimisation.
        callback: Single Callback or list of Callbacks for monitoring.
        copy_algorithm: If True, create a copy of the algorithm.
        save_history: If True (default), save convergence history.
        initialize: If True (default), initialize the algorithm with the
            problem. Set to False to continue optimization with an already
            initialized algorithm (e.g., when switching problems at runtime
            while preserving population state and hyperparameters).
            The algorithm must have been previously initialized. When False,
            the termination budget is additive (e.g., MaxEvaluations(500)
            will run 500 more evaluations from the current state).

        # Differentiable mode options:
        optimizer: PyTorch optimizer for gradient-based updates.
        lr_pop: Learning rate for population updates.
        lr_hyper: Learning rate for hyperparameter updates.
        grad_clip_pop: Maximum gradient norm for population clipping.
        grad_clip_hyper: Maximum gradient norm for hyperparameter clipping.
        scheduler: Learning rate scheduler type.
        scheduler_patience: Generations before reducing LR.
        scheduler_factor: Factor to multiply LR when reducing.
        min_lr: Minimum learning rate.
    
    Returns:
        Result object with:
            - best_solution: Best solution found
            - best_fitness: Best (maximum) fitness value
            - population: Final population
            - fitness: Final fitness values (actual, not negated)
            - history: Convergence history (with actual fitness values)
    
    Example:
        >>> from evograd.core.problem import Problem
        >>> from evograd.core.maximize import maximize
        >>> from evograd.core.termination import MaxEvaluations
        >>> from evograd.algorithms import GA
        >>> 
        >>> # Maximize a function
        >>> problem = Problem(
        ...     objective=lambda x: torch.sin(x).sum(dim=-1),
        ...     n_var=10,
        ...     xl=-3.14,
        ...     xu=3.14,
        ... )
        >>> 
        >>> result = maximize(problem, GA(pop_size=100), termination=MaxEvaluations(10000), seed=42)
        >>> print(f"Maximum value: {result.best_fitness}")
    
    Note:
        The returned fitness values are the ACTUAL values (not negated).
        If you're looking for minimum fitness, use minimize() instead.
    """
    # Wrap problem to negate objective
    negated_problem = _NegatedProblem(problem)

    # Run minimisation on negated problem
    result = minimize(
        problem=negated_problem,
        algorithm=algorithm,
        termination=termination,
        seed=seed,
        verbose=verbose,
        callback=callback,
        copy_algorithm=copy_algorithm,
        save_history=save_history,
        initialize=initialize,
        optimizer=optimizer,
        lr_pop=lr_pop,
        lr_hyper=lr_hyper,
        grad_clip_pop=grad_clip_pop,
        grad_clip_hyper=grad_clip_hyper,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        min_lr=min_lr,
        reduction=reduction,
        live_selection=live_selection,
    )
    
    # Fix problem name in result
    result.problem_name = problem.name
    
    # Negate fitness values back to actual values
    return _NegatedResult.from_minimization_result(result)
