"""
CEC 2017 Benchmark Functions for PyTorch.

This package provides a comprehensive implementation of the CEC 2017
benchmark suite for real-parameter single objective optimization.

The suite consists of 30 functions:
- F1-F10: Simple functions (shifted and rotated basic functions)
- F11-F20: Hybrid functions (combinations of basic functions)
- F21-F30: Composition functions (weighted combinations)

All functions:
- Accept PyTorch tensors
- Support batch evaluation
- Have bounds of [-100, 100]
- Have optimal values of F_i* = i * 100

Usage:
    >>> from evograd.benchmarks.functions.cec2017 import CEC2017_F1, CEC2017_F10
    >>> 
    >>> # Create function with default settings
    >>> f1 = CEC2017_F1(n_var=10)
    >>> 
    >>> # Evaluate population
    >>> x = torch.randn(100, 10)
    >>> fitness = f1(x)  # Shape: [100]
    >>> 
    >>> # Get optimal value
    >>> print(f"Optimal value: {f1.optimal_value}")  # 100.0

Supported dimensions: 2, 10, 20, 30, 50, 100 (for official data)
Other dimensions will use randomly generated transforms.

Reference:
    N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu, and P. N. Suganthan,
    "Problem Definitions and Evaluation Criteria for the CEC 2017 
    Special Session and Competition on Single Objective Real-Parameter
    Numerical Optimization," Technical Report, 2016.
"""

# Basic functions (stateless, for internal use)
from .basic import (
    bent_cigar,
    sum_diff_pow,
    zakharov,
    rosenbrock,
    rastrigin,
    expanded_schaffers_f6,
    lunacek_bi_rastrigin,
    non_cont_rastrigin,
    levy,
    modified_schwefel,
    high_conditioned_elliptic,
    discus,
    ackley,
    weierstrass,
    griewank,
    katsuura,
    happy_cat,
    h_g_bat,
    expanded_griewanks_plus_rosenbrock,
    schaffers_f7,
    BASIC_FUNCTIONS,
)

# Simple functions (F1-F10)
from .simple import (
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
    SIMPLE_FUNCTIONS,
)

# Hybrid functions (F11-F20)
from .hybrid import (
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
    HYBRID_FUNCTIONS,
)

# Composition functions (F21-F30)
from .composition import (
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
    COMPOSITION_FUNCTIONS,
)

# Data utilities
from .data import (
    data_available,
    get_rotation,
    get_rotation_cf,
    get_shift,
    get_shift_cf,
    get_shuffle,
    get_shuffle_cf,
    generate_rotation_matrix,
    generate_shift_vector,
    generate_shuffle,
    shift_rotate,
    shuffle_and_partition,
    SUPPORTED_DIMS,
)


# Combined registry of all CEC 2017 functions
CEC2017_FUNCTIONS = {
    **SIMPLE_FUNCTIONS,
    **HYBRID_FUNCTIONS,
    **COMPOSITION_FUNCTIONS,
}

# List of all function classes for iteration
ALL_CEC2017_CLASSES = [
    # Simple (F1-F10)
    CEC2017_F1, CEC2017_F2, CEC2017_F3, CEC2017_F4, CEC2017_F5,
    CEC2017_F6, CEC2017_F7, CEC2017_F8, CEC2017_F9, CEC2017_F10,
    # Hybrid (F11-F20)
    CEC2017_F11, CEC2017_F12, CEC2017_F13, CEC2017_F14, CEC2017_F15,
    CEC2017_F16, CEC2017_F17, CEC2017_F18, CEC2017_F19, CEC2017_F20,
    # Composition (F21-F30)
    CEC2017_F21, CEC2017_F22, CEC2017_F23, CEC2017_F24, CEC2017_F25,
    CEC2017_F26, CEC2017_F27, CEC2017_F28, CEC2017_F29, CEC2017_F30,
]


def get_function(func_num: int, n_var: int = 10, **kwargs):
    """
    Get CEC 2017 function by number.
    
    Args:
        func_num: Function number (1-30).
        n_var: Number of variables.
        **kwargs: Additional arguments passed to the function constructor.
    
    Returns:
        BenchmarkFunction instance.
    
    Example:
        >>> f1 = get_function(1, n_var=10)
        >>> f15 = get_function(15, n_var=30)
    """
    if func_num < 1 or func_num > 30:
        raise ValueError(f"Function number must be 1-30, got {func_num}")
    
    func_class = ALL_CEC2017_CLASSES[func_num - 1]
    return func_class(n_var=n_var, **kwargs)


__all__ = [
    # Basic functions
    "bent_cigar",
    "sum_diff_pow",
    "zakharov",
    "rosenbrock",
    "rastrigin",
    "expanded_schaffers_f6",
    "lunacek_bi_rastrigin",
    "non_cont_rastrigin",
    "levy",
    "modified_schwefel",
    "high_conditioned_elliptic",
    "discus",
    "ackley",
    "weierstrass",
    "griewank",
    "katsuura",
    "happy_cat",
    "h_g_bat",
    "expanded_griewanks_plus_rosenbrock",
    "schaffers_f7",
    "BASIC_FUNCTIONS",
    # Simple functions (F1-F10)
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
    "SIMPLE_FUNCTIONS",
    # Hybrid functions (F11-F20)
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
    "HYBRID_FUNCTIONS",
    # Composition functions (F21-F30)
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
    "COMPOSITION_FUNCTIONS",
    # Combined registries
    "CEC2017_FUNCTIONS",
    "ALL_CEC2017_CLASSES",
    # Utilities
    "get_function",
    "data_available",
    "get_rotation",
    "get_rotation_cf",
    "get_shift",
    "get_shift_cf",
    "get_shuffle",
    "get_shuffle_cf",
    "generate_rotation_matrix",
    "generate_shift_vector",
    "generate_shuffle",
    "shift_rotate",
    "shuffle_and_partition",
    "SUPPORTED_DIMS",
]
