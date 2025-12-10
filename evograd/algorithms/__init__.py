"""
EvoGrad Algorithms Module.

This module provides implementations of population-based optimisation
algorithms that leverage the EvoGrad framework for both classical and
differentiable operation.

Available Algorithms:
    - GA: Genetic Algorithm with SBX crossover and polynomial mutation
    - DE: Differential Evolution with multiple mutation/crossover variants
    - PSO: Particle Swarm Optimisation with adaptive coefficients
    - CMAES: Covariance Matrix Adaptation Evolution Strategy

All algorithms inherit from the base Algorithm class and follow
the dependency injection pattern for operators.

Modes:
    Each algorithm supports four operating modes controlled by two flags:
    
    - adaptive=False, differentiable=False: Classical algorithm
    - adaptive=True, differentiable=False: Operators/hyperparameters are
        differentiable and learned via backpropagation
    - adaptive=False, differentiable=True: Population is differentiable
        and learned via backpropagation
    - adaptive=True, differentiable=True: Both operators/hyperparameters
        and population are differentiable

Example:
    >>> from evograd.algorithms import GA, DE, PSO, CMAES
    >>> from evograd.core import Problem, minimize
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Genetic Algorithm
    >>> ga = GA(pop_size=100, differentiable=True)
    >>> result = minimize(problem, ga, max_evals=10000)
    >>> 
    >>> # Differential Evolution with adaptive hyperparameters
    >>> de = DE(pop_size=100, variant="DE/rand/1/bin", adaptive=True)
    >>> result = minimize(problem, de, max_evals=10000)
    >>> 
    >>> # Particle Swarm Optimisation
    >>> pso = PSO(pop_size=100, adaptive=True, differentiable=True)
    >>> result = minimize(problem, pso, max_evals=10000)
    >>> 
    >>> # CMA-ES with learnable adaptation coefficients
    >>> cmaes = CMAES(sigma=0.5, adaptive=True)
    >>> result = minimize(problem, cmaes, max_evals=10000)
"""

from evograd.algorithms.ga import GA, ga_default, ga_steady_state, ga_comma
from evograd.algorithms.de import (
    DE,
    DEVariant,
    de_rand_1_bin,
    de_best_1_bin,
    de_current_to_best_1_bin,
)
from evograd.algorithms.pso import (
    PSO,
    pso_default,
    pso_constriction,
    pso_adaptive,
)
from evograd.algorithms.cmaes import (
    CMAES,
    cmaes_default,
    cmaes_small,
    cmaes_large,
    cmaes_adaptive,
)

__all__ = [
    # Genetic Algorithm
    "GA",
    "ga_default",
    "ga_steady_state",
    "ga_comma",
    # Differential Evolution
    "DE",
    "DEVariant",
    "de_rand_1_bin",
    "de_best_1_bin",
    "de_current_to_best_1_bin",
    # Particle Swarm Optimisation
    "PSO",
    "pso_default",
    "pso_constriction",
    "pso_adaptive",
    # CMA-ES
    "CMAES",
    "cmaes_default",
    "cmaes_small",
    "cmaes_large",
    "cmaes_adaptive",
]