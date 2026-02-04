"""
EvoGrad core module.

This module provides the foundational classes for building
population-based optimisation algorithms:

    - Algorithm: Abstract base class for all algorithms
    - AlgorithmState: Container for algorithm state
    - Problem: Objective function + bounds + constraints
    - Result: Container for optimisation results
    - Termination: Termination criteria (MaxEvaluations, etc.)
    - minimize/maximize: Main entry points for optimisation

Example:
    >>> from evograd.core.problem import Problem
    >>> from evograd.core.minimize import minimize
    >>> from evograd.core.termination import MaxEvaluations
    >>> from evograd.algorithms import GA
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=10,
    ...     xl=-5.0,
    ...     xu=5.0,
    ... )
    >>> 
    >>> algorithm = GA(pop_size=100)
    >>> result = minimize(problem, algorithm, termination=MaxEvaluations(10000))
"""

from evograd.core.algorithm import (
    Algorithm,
    AlgorithmState,
)
from evograd.core.problem import Problem
from evograd.core.result import Result, ResultBuilder
from evograd.core.termination import (
    Termination,
    MaxEvaluations,
    MaxGenerations,
    TargetReached,
    ToleranceReached,
    TimeLimit,
    NoTermination,
    TerminationCollection,
    default_termination,
)
from evograd.core.minimize import minimize
from evograd.core.maximize import maximize

__all__ = [
    # Algorithm
    "Algorithm",
    "AlgorithmState",
    # Problem
    "Problem",
    # Result
    "Result",
    "ResultBuilder",
    # Termination
    "Termination",
    "MaxEvaluations",
    "MaxGenerations",
    "TargetReached",
    "ToleranceReached",
    "TimeLimit",
    "NoTermination",
    "TerminationCollection",
    "default_termination",
    # Optimization functions
    "minimize",
    "maximize",
]