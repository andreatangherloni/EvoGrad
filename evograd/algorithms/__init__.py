"""
EvoGrad Algorithms Module.

This module provides implementations of population-based optimisation
algorithms that leverage the EvoGrad framework for both classical and
differentiable operation.

Available Algorithms:
    - GA: Genetic Algorithm with SBX crossover and polynomial mutation
    - DE: Differential Evolution with multiple mutation strategies
    - SHADE: Success-History based Adaptive DE
    - LSHADE: SHADE with Linear Population Size Reduction
    - PSO: Particle Swarm Optimisation
    - CMAES: Covariance Matrix Adaptation Evolution Strategy

All algorithms inherit from the base Algorithm class and follow
the dependency injection pattern for operators.

Example:
    >>> from evograd.algorithms import GA, DE, SHADE, PSO, CMAES
    >>> from evograd.core import Problem, minimize, MaxEvaluations
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
    >>> result = minimize(problem, ga, MaxEvaluations(10000))
    >>> 
    >>> # Differential Evolution
    >>> de = DE(pop_size=100, variant="DE/rand/1/bin", adaptive=True)
    >>> result = minimize(problem, de, MaxEvaluations(10000))
    >>> 
    >>> # SHADE (Self-Adaptive DE)
    >>> shade = SHADE(pop_size=100, memory_size=100)
    >>> result = minimize(problem, shade, MaxEvaluations(10000))
    >>> 
    >>> # L-SHADE (SHADE with population reduction)
    >>> lshade = LSHADE(pop_size_init=18*30, pop_size_min=4)
    >>> result = minimize(problem, lshade, MaxEvaluations(10000))
    >>> 
    >>> # Particle Swarm Optimisation
    >>> pso = PSO(pop_size=100, adaptive=True)
    >>> result = minimize(problem, pso, MaxEvaluations(10000))
    >>> 
    >>> # CMA-ES
    >>> cmaes = CMAES(pop_size=50, adaptive=True)
    >>> result = minimize(problem, cmaes, MaxEvaluations(10000))
"""

# Genetic Algorithm
from evograd.algorithms.ga import (
    GA,
    ga_default,
    ga_steady_state,
    ga_comma)

# Differential Evolution
from .de import (
    DE,
    DEVariant,
    de_default,
    de_rand_1_bin,
    de_best_1_bin,
    de_current_to_best_1_bin,
)

# SHADE and L-SHADE
from .shade import (
    SHADE,
    LSHADE,
    SHADEMemory,
    shade_default,
    shade_adaptive,
    lshade_default,
    lshade_adaptive,
)

# Particle Swarm Optimisation
from .pso import (
    PSO,
    pso_default,
    pso_constriction)

# CMA-ES
from .cmaes import (
    CMAES,
    cmaes_default,
    cmaes_small,
    cmaes_large,
    cmaes_adaptive,
    cmaes_ipop,
    cmaes_bipop,
)


__all__ = [
    # GAs
    "GA",
    "ga_default",
    "ga_steady_state",
    "ga_comma",
    
    # DE
    "DE",
    "DEVariant",
    "de_default",
    "de_rand_1_bin",
    "de_best_1_bin",
    "de_current_to_best_1_bin",
    
    # SHADE family
    "SHADE",
    "LSHADE",
    "SHADEMemory",
    "shade_default",
    "shade_adaptive",
    "lshade_default",
    "lshade_adaptive",
    
    # PSO
    "PSO",
    "pso_default",
    "pso_constriction",
    
    # CMA-ES
    "CMAES",
    "cmaes_default",
    "cmaes_small",
    "cmaes_large",
    "cmaes_adaptive",
    "cmaes_ipop",
    "cmaes_bipop",
]
