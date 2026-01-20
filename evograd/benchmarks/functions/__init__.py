"""
EvoGrad Benchmark Functions.

This package provides a comprehensive set of benchmark functions for
testing and comparing optimization algorithms.

Categories:
- Classical: Standard test functions (Sphere, Rosenbrock, Rastrigin, etc.)
- CEC 2017: Competition benchmark suite (F1-F30)
- Smoothed Funnel: Multi-basin problems designed for differentiable EAs
- Transforms: Wrappers for creating shifted/rotated variants

Usage:
    >>> from evograd.benchmarks.functions import Sphere, Rastrigin
    >>> from evograd.benchmarks.functions import CEC2017_F1, get_cec2017_function
    >>> from evograd.benchmarks.functions import SmoothedMultiFunnel
    >>> 
    >>> # Create function with default bounds
    >>> sphere = Sphere(n_var=30)
    >>> 
    >>> # Create function with custom bounds
    >>> sphere = Sphere(n_var=30, xl=-5.0, xu=5.0)
    >>> 
    >>> # Evaluate population
    >>> x = torch.randn(100, 30)
    >>> f = sphere(x)  # Shape: [100]
    >>> 
    >>> # Get bounds
    >>> lb, ub = sphere.bounds
    >>>
    >>> # CEC 2017 functions
    >>> f1 = CEC2017_F1(n_var=10)
    >>> f15 = get_cec2017_function(15, n_var=10)
    >>>
    >>> # Smoothed multi-funnel (designed for differentiable EAs)
    >>> funnel = SmoothedMultiFunnel(n_var=10, tau=1.0, delta=10.0)
"""

from .base import BenchmarkFunction, CompositeFunction

from .classical import (
    # Unimodal
    Sphere,
    Ellipsoid,
    SumOfDifferentPowers,
    Schwefel222,
    Cigar,
    Discus,
    BentCigar,
    Rosenbrock,
    DixonPrice,
    Powell,
    Trid,
    # Multimodal
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
    Levy,
    Michalewicz,
    Zakharov,
    Weierstrass,
    Alpine,
    Salomon,
    StyblinskiTang,
    # Registry
    CLASSICAL_FUNCTIONS,
)

from .transforms import (
    ShiftedFunction,
    RotatedFunction,
    ShiftedRotatedFunction,
    ScaledFunction,
    AsymmetricFunction,
    OscillatedFunction,
    BiasedFunction,
    generate_rotation_matrix,
    generate_shift_vector,
)

# CEC 2017 Benchmark Suite
from .cec2017 import (
    # Simple functions (F1-F10)
    CEC2017_F1,
    CEC2017_F2,
    CEC2017_F3,
    CEC2017_F4,
    CEC2017_F5,
    CEC2017_F6,
    CEC2017_F7,
    CEC2017_F8,
    CEC2017_F9,
    CEC2017_F10,
    # Hybrid functions (F11-F20)
    CEC2017_F11,
    CEC2017_F12,
    CEC2017_F13,
    CEC2017_F14,
    CEC2017_F15,
    CEC2017_F16,
    CEC2017_F17,
    CEC2017_F18,
    CEC2017_F19,
    CEC2017_F20,
    # Composition functions (F21-F30)
    CEC2017_F21,
    CEC2017_F22,
    CEC2017_F23,
    CEC2017_F24,
    CEC2017_F25,
    CEC2017_F26,
    CEC2017_F27,
    CEC2017_F28,
    CEC2017_F29,
    CEC2017_F30,
    # Registries
    CEC2017_FUNCTIONS,
    ALL_CEC2017_CLASSES,
    # Utilities
    get_function as get_cec2017_function,
)

# Smoothed Multi-Funnel Functions (designed for differentiable EAs)
from .smoothed_funnel import (
    # Core utilities
    log_sum_exp_min,
    random_orthogonal_matrix,
    # Benchmark functions
    SmoothedMultiFunnel,
    MultiBasinRosenbrock,
    DeceptiveLandscape,
    # Registry
    SMOOTHED_FUNNEL_FUNCTIONS,
)

# Combined registry of all functions
ALL_FUNCTIONS = {
    **CLASSICAL_FUNCTIONS,
    **CEC2017_FUNCTIONS,
    **SMOOTHED_FUNNEL_FUNCTIONS,
}

__all__ = [
    # Base
    "BenchmarkFunction",
    "CompositeFunction",
    # Classical - Unimodal
    "Sphere",
    "Ellipsoid",
    "SumOfDifferentPowers",
    "Schwefel222",
    "Cigar",
    "Discus",
    "BentCigar",
    "Rosenbrock",
    "DixonPrice",
    "Powell",
    "Trid",
    # Classical - Multimodal
    "Rastrigin",
    "Ackley",
    "Griewank",
    "Schwefel",
    "Levy",
    "Michalewicz",
    "Zakharov",
    "Weierstrass",
    "Alpine",
    "Salomon",
    "StyblinskiTang",
    # Transforms
    "ShiftedFunction",
    "RotatedFunction",
    "ShiftedRotatedFunction",
    "ScaledFunction",
    "AsymmetricFunction",
    "OscillatedFunction",
    "BiasedFunction",
    "generate_rotation_matrix",
    "generate_shift_vector",
    # CEC 2017 - Simple (F1-F10)
    "CEC2017_F1",
    "CEC2017_F2",
    "CEC2017_F3",
    "CEC2017_F4",
    "CEC2017_F5",
    "CEC2017_F6",
    "CEC2017_F7",
    "CEC2017_F8",
    "CEC2017_F9",
    "CEC2017_F10",
    # CEC 2017 - Hybrid (F11-F20)
    "CEC2017_F11",
    "CEC2017_F12",
    "CEC2017_F13",
    "CEC2017_F14",
    "CEC2017_F15",
    "CEC2017_F16",
    "CEC2017_F17",
    "CEC2017_F18",
    "CEC2017_F19",
    "CEC2017_F20",
    # CEC 2017 - Composition (F21-F30)
    "CEC2017_F21",
    "CEC2017_F22",
    "CEC2017_F23",
    "CEC2017_F24",
    "CEC2017_F25",
    "CEC2017_F26",
    "CEC2017_F27",
    "CEC2017_F28",
    "CEC2017_F29",
    "CEC2017_F30",
    # CEC 2017 utilities
    "CEC2017_FUNCTIONS",
    "ALL_CEC2017_CLASSES",
    "get_cec2017_function",
    # Smoothed Multi-Funnel (for differentiable EAs)
    "log_sum_exp_min",
    "random_orthogonal_matrix",
    "SmoothedMultiFunnel",
    "MultiBasinRosenbrock",
    "DeceptiveLandscape",
    "SMOOTHED_FUNNEL_FUNCTIONS",
    # Registries
    "CLASSICAL_FUNCTIONS",
    "ALL_FUNCTIONS",
]
