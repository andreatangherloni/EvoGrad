"""
CEC 2017 Hybrid Functions (F11-F20).

Hybrid functions combine multiple basic functions applied to different
partitions of the input vector after shuffling.

Reference: CEC 2017 Competition on Real-Parameter Single Objective Optimization

F11: Hybrid Function 1 (N=3)
F12: Hybrid Function 2 (N=3)
F13: Hybrid Function 3 (N=3)
F14: Hybrid Function 4 (N=4)
F15: Hybrid Function 5 (N=4)
F16: Hybrid Function 6 (N=4)
F17: Hybrid Function 7 (N=5)
F18: Hybrid Function 8 (N=5)
F19: Hybrid Function 9 (N=5)
F20: Hybrid Function 10 (N=6)
"""

from typing import List, Optional, Tuple, Callable

import torch
from torch import Tensor

from ..base import BenchmarkFunction
from . import basic
from . import data as cec_data


class CEC2017HybridFunction(BenchmarkFunction):
    """Base class for CEC 2017 hybrid functions."""
    
    def __init__(
        self,
        func_num: int,
        n_var: int = 10,
        rotation: Optional[Tensor] = None,
        shift: Optional[Tensor] = None,
        shuffle: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize CEC 2017 hybrid function.
        
        Args:
            func_num: Function number (11-20).
            n_var: Number of variables.
            rotation: Optional rotation matrix.
            shift: Optional shift vector.
            shuffle: Optional shuffle permutation.
            seed: Random seed for generating transforms if not provided.
        """
        super().__init__(n_var=n_var, xl=-100.0, xu=100.0)
        
        self.func_num = func_num
        self.bias = func_num * 100.0
        
        # Load or generate transforms
        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = cec_data.get_rotation(func_num, n_var, seed=seed)
        
        if shift is not None:
            self.shift = shift
        else:
            self.shift = cec_data.get_shift(func_num, n_var, seed=seed)
        
        if shuffle is not None:
            self.shuffle = shuffle
        else:
            self.shuffle = cec_data.get_shuffle(func_num, n_var, seed=seed)
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def _shift_rotate(self, x: Tensor) -> Tensor:
        """Apply shift and rotation transformation."""
        shift = self.shift.to(x.device, x.dtype)
        rotation = self.rotation.to(x.device, x.dtype)
        return cec_data.shift_rotate(x, shift, rotation)
    
    def _shuffle_and_partition(self, x: Tensor, partitions: List[float]) -> List[Tensor]:
        """Apply shuffle and partition."""
        shuffle = self.shuffle.to(x.device)
        return cec_data.shuffle_and_partition(x, shuffle, partitions)
    
    def _evaluate_hybrid(
        self,
        x: Tensor,
        partitions: List[float],
        funcs: List[Callable],
    ) -> Tensor:
        """
        Evaluate hybrid function.
        
        Args:
            x: Input tensor of shape [..., n_var].
            partitions: List of partition fractions.
            funcs: List of basic functions to apply to each partition.
        
        Returns:
            Function values.
        """
        z = self._shift_rotate(x)
        parts = self._shuffle_and_partition(z, partitions)
        
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for part, func in zip(parts, funcs):
            result = result + func(part)
        
        return result + self.bias


class CEC2017_F11(CEC2017HybridFunction):
    """
    F11: Hybrid Function 1 (N=3)
    
    Components: Zakharov, Rosenbrock, Rastrigin
    Partitions: 0.2, 0.4, 0.4
    
    Optimal value: F11* = 1100
    """
    name = "cec2017_f11"
    optimal_value = 1100.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=11, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.4, 0.4]
        self.funcs = [basic.zakharov, basic.rosenbrock, basic.rastrigin]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F12(CEC2017HybridFunction):
    """
    F12: Hybrid Function 2 (N=3)
    
    Components: High Conditioned Elliptic, Modified Schwefel, Bent Cigar
    Partitions: 0.3, 0.3, 0.4
    
    Optimal value: F12* = 1200
    """
    name = "cec2017_f12"
    optimal_value = 1200.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=12, n_var=n_var, **kwargs)
        self.partitions = [0.3, 0.3, 0.4]
        self.funcs = [
            basic.high_conditioned_elliptic,
            basic.modified_schwefel,
            basic.bent_cigar,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F13(CEC2017HybridFunction):
    """
    F13: Hybrid Function 3 (N=3)
    
    Components: Bent Cigar, Rosenbrock, Lunacek Bi-Rastrigin
    Partitions: 0.3, 0.3, 0.4
    
    Optimal value: F13* = 1300
    """
    name = "cec2017_f13"
    optimal_value = 1300.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=13, n_var=n_var, **kwargs)
        self.partitions = [0.3, 0.3, 0.4]
        self.funcs = [
            basic.bent_cigar,
            basic.rosenbrock,
            basic.lunacek_bi_rastrigin,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F14(CEC2017HybridFunction):
    """
    F14: Hybrid Function 4 (N=4)
    
    Components: High Conditioned Elliptic, Ackley, Schaffer's F7, Rastrigin
    Partitions: 0.2, 0.2, 0.2, 0.4
    
    Optimal value: F14* = 1400
    """
    name = "cec2017_f14"
    optimal_value = 1400.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=14, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.2, 0.2, 0.4]
        self.funcs = [
            basic.high_conditioned_elliptic,
            basic.ackley,
            basic.schaffers_f7,
            basic.rastrigin,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F15(CEC2017HybridFunction):
    """
    F15: Hybrid Function 5 (N=4)
    
    Components: Bent Cigar, HGBat, Rastrigin, Rosenbrock
    Partitions: 0.2, 0.2, 0.3, 0.3
    
    Optimal value: F15* = 1500
    """
    name = "cec2017_f15"
    optimal_value = 1500.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=15, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.2, 0.3, 0.3]
        self.funcs = [
            basic.bent_cigar,
            basic.h_g_bat,
            basic.rastrigin,
            basic.rosenbrock,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F16(CEC2017HybridFunction):
    """
    F16: Hybrid Function 6 (N=4)
    
    Components: Expanded Schaffer's F6, HGBat, Rosenbrock, Modified Schwefel
    Partitions: 0.2, 0.2, 0.3, 0.3
    
    Optimal value: F16* = 1600
    """
    name = "cec2017_f16"
    optimal_value = 1600.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=16, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.2, 0.3, 0.3]
        self.funcs = [
            basic.expanded_schaffers_f6,
            basic.h_g_bat,
            basic.rosenbrock,
            basic.modified_schwefel,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F17(CEC2017HybridFunction):
    """
    F17: Hybrid Function 7 (N=5)
    
    Components: Katsuura, Ackley, Expanded Griewank's plus Rosenbrock,
                Modified Schwefel, Rastrigin
    Partitions: 0.1, 0.2, 0.2, 0.2, 0.3
    
    Optimal value: F17* = 1700
    """
    name = "cec2017_f17"
    optimal_value = 1700.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=17, n_var=n_var, **kwargs)
        self.partitions = [0.1, 0.2, 0.2, 0.2, 0.3]
        self.funcs = [
            basic.katsuura,
            basic.ackley,
            basic.expanded_griewanks_plus_rosenbrock,
            basic.modified_schwefel,
            basic.rastrigin,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F18(CEC2017HybridFunction):
    """
    F18: Hybrid Function 8 (N=5)
    
    Components: High Conditioned Elliptic, Ackley, Rastrigin, HGBat, Discus
    Partitions: 0.2, 0.2, 0.2, 0.2, 0.2
    
    Optimal value: F18* = 1800
    """
    name = "cec2017_f18"
    optimal_value = 1800.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=18, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.funcs = [
            basic.high_conditioned_elliptic,
            basic.ackley,
            basic.rastrigin,
            basic.h_g_bat,
            basic.discus,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F19(CEC2017HybridFunction):
    """
    F19: Hybrid Function 9 (N=5)
    
    Components: Bent Cigar, Rastrigin, Expanded Griewank's plus Rosenbrock,
                Weierstrass, Expanded Schaffer's F6
    Partitions: 0.2, 0.2, 0.2, 0.2, 0.2
    
    Optimal value: F19* = 1900
    """
    name = "cec2017_f19"
    optimal_value = 1900.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=19, n_var=n_var, **kwargs)
        self.partitions = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.funcs = [
            basic.bent_cigar,
            basic.rastrigin,
            basic.expanded_griewanks_plus_rosenbrock,
            basic.weierstrass,
            basic.expanded_schaffers_f6,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


class CEC2017_F20(CEC2017HybridFunction):
    """
    F20: Hybrid Function 10 (N=6)
    
    Components: Happy Cat, Katsuura, Ackley, Rastrigin, Modified Schwefel,
                Schaffer's F7
    Partitions: 0.1, 0.1, 0.2, 0.2, 0.2, 0.2
    
    Optimal value: F20* = 2000
    """
    name = "cec2017_f20"
    optimal_value = 2000.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=20, n_var=n_var, **kwargs)
        self.partitions = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        self.funcs = [
            basic.happy_cat,
            basic.katsuura,
            basic.ackley,
            basic.rastrigin,
            basic.modified_schwefel,
            basic.schaffers_f7,
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._evaluate_hybrid(x, self.partitions, self.funcs)


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

HYBRID_FUNCTIONS = {
    "cec2017_f11": CEC2017_F11,
    "cec2017_f12": CEC2017_F12,
    "cec2017_f13": CEC2017_F13,
    "cec2017_f14": CEC2017_F14,
    "cec2017_f15": CEC2017_F15,
    "cec2017_f16": CEC2017_F16,
    "cec2017_f17": CEC2017_F17,
    "cec2017_f18": CEC2017_F18,
    "cec2017_f19": CEC2017_F19,
    "cec2017_f20": CEC2017_F20,
}

# List for iteration
all_functions = [
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
]
