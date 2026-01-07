"""
CEC 2017 Simple Functions (F1-F10).

Shifted and rotated versions of basic benchmark functions.

Reference: CEC 2017 Competition on Real-Parameter Single Objective Optimization

F1:  Shifted and Rotated Bent Cigar Function
F2:  (Deprecated) Shifted and Rotated Sum of Different Powers Function
F3:  Shifted and Rotated Zakharov Function
F4:  Shifted and Rotated Rosenbrock's Function
F5:  Shifted and Rotated Rastrigin's Function
F6:  Shifted and Rotated Schaffer's F7 Function
F7:  Shifted and Rotated Lunacek Bi-Rastrigin's Function
F8:  Shifted and Rotated Non-Continuous Rastrigin's Function
F9:  Shifted and Rotated Levy Function
F10: Shifted and Rotated Schwefel's Function
"""

from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor

from ..base import BenchmarkFunction
from . import basic
from . import data as cec_data


class CEC2017Function(BenchmarkFunction):
    """Base class for CEC 2017 functions with shift and rotation support."""
    
    def __init__(
        self,
        func_num: int,
        n_var: int = 10,
        rotation: Optional[Tensor] = None,
        shift: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize CEC 2017 function.
        
        Args:
            func_num: Function number (1-30).
            n_var: Number of variables (2, 10, 20, 30, 50, or 100 for official data).
            rotation: Optional rotation matrix. If None, loads from data or generates.
            shift: Optional shift vector. If None, loads from data or generates.
            seed: Random seed for generating transforms if not provided.
        """
        super().__init__(n_var=n_var, xl=-100.0, xu=100.0)
        
        self.func_num = func_num
        self.bias = func_num * 100.0  # F_i* = i * 100
        
        # Load or generate rotation matrix
        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = cec_data.get_rotation(func_num, n_var, seed=seed)
        
        # Load or generate shift vector
        if shift is not None:
            self.shift = shift
        else:
            self.shift = cec_data.get_shift(func_num, n_var, seed=seed)
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def _shift_rotate(self, x: Tensor) -> Tensor:
        """Apply shift and rotation transformation."""
        shift = self.shift.to(x.device, x.dtype)
        rotation = self.rotation.to(x.device, x.dtype)
        return cec_data.shift_rotate(x, shift, rotation)


class CEC2017_F1(CEC2017Function):
    """
    F1: Shifted and Rotated Bent Cigar Function
    
    Properties:
        - Unimodal
        - Non-separable
        - Ill-conditioned
        - Optimal value: F1* = 100
    """
    name = "cec2017_f1"
    optimal_value = 100.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=1, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.bent_cigar(z) + self.bias


class CEC2017_F2(CEC2017Function):
    """
    F2: (DEPRECATED) Shifted and Rotated Sum of Different Powers Function
    
    Note: This function was deprecated from the CEC 2017 benchmark suite
    due to numerical instability. It is included for completeness.
    
    Properties:
        - Unimodal
        - Non-separable
        - Optimal value: F2* = 200
    """
    name = "cec2017_f2"
    optimal_value = 200.0
    _warned = False
    
    def __init__(self, n_var: int = 10, **kwargs):
        if not CEC2017_F2._warned:
            warnings.warn(
                "F2 has been deprecated from the CEC 2017 benchmark suite "
                "due to numerical instability.",
                DeprecationWarning,
            )
            CEC2017_F2._warned = True
        super().__init__(func_num=2, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.sum_diff_pow(z) + self.bias


class CEC2017_F3(CEC2017Function):
    """
    F3: Shifted and Rotated Zakharov Function
    
    Properties:
        - Unimodal
        - Non-separable
        - Optimal value: F3* = 300
    """
    name = "cec2017_f3"
    optimal_value = 300.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=3, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.zakharov(z) + self.bias


class CEC2017_F4(CEC2017Function):
    """
    F4: Shifted and Rotated Rosenbrock's Function
    
    Properties:
        - Multimodal (for high dimensions)
        - Non-separable
        - Optimal value: F4* = 400
    """
    name = "cec2017_f4"
    optimal_value = 400.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=4, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.rosenbrock(z) + self.bias


class CEC2017_F5(CEC2017Function):
    """
    F5: Shifted and Rotated Rastrigin's Function
    
    Properties:
        - Highly multimodal
        - Non-separable (after rotation)
        - Optimal value: F5* = 500
    """
    name = "cec2017_f5"
    optimal_value = 500.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=5, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.rastrigin(z) + self.bias


class CEC2017_F6(CEC2017Function):
    """
    F6: Shifted and Rotated Schaffer's F7 Function
    
    Properties:
        - Multimodal
        - Non-separable
        - Optimal value: F6* = 600
    """
    name = "cec2017_f6"
    optimal_value = 600.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=6, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.schaffers_f7(z) + self.bias


class CEC2017_F7(CEC2017Function):
    """
    F7: Shifted and Rotated Lunacek Bi-Rastrigin's Function
    
    Properties:
        - Multimodal with two global optima regions
        - Non-separable
        - Optimal value: F7* = 700
    """
    name = "cec2017_f7"
    optimal_value = 700.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=7, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        # Special case: pass shift and rotation directly to function
        shift = self.shift.to(x.device, x.dtype)
        rotation = self.rotation.to(x.device, x.dtype)
        return basic.lunacek_bi_rastrigin(x, shift, rotation) + self.bias


class CEC2017_F8(CEC2017Function):
    """
    F8: Shifted and Rotated Non-Continuous Rastrigin's Function
    
    Properties:
        - Multimodal
        - Non-separable
        - Non-continuous
        - Optimal value: F8* = 800
    """
    name = "cec2017_f8"
    optimal_value = 800.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=8, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        # Special case: pass shift and rotation directly to function
        shift = self.shift.to(x.device, x.dtype)
        rotation = self.rotation.to(x.device, x.dtype)
        return basic.non_cont_rastrigin(x, shift, rotation) + self.bias


class CEC2017_F9(CEC2017Function):
    """
    F9: Shifted and Rotated Levy Function
    
    Properties:
        - Multimodal
        - Non-separable
        - Optimal value: F9* = 900
    """
    name = "cec2017_f9"
    optimal_value = 900.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=9, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.levy(z) + self.bias


class CEC2017_F10(CEC2017Function):
    """
    F10: Shifted and Rotated Schwefel's Function
    
    Properties:
        - Multimodal
        - Non-separable
        - Optimal far from origin
        - Optimal value: F10* = 1000
    """
    name = "cec2017_f10"
    optimal_value = 1000.0
    
    def __init__(self, n_var: int = 10, **kwargs):
        super().__init__(func_num=10, n_var=n_var, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        z = self._shift_rotate(x)
        return basic.modified_schwefel(z) + self.bias


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

SIMPLE_FUNCTIONS = {
    "cec2017_f1": CEC2017_F1,
    "cec2017_f2": CEC2017_F2,
    "cec2017_f3": CEC2017_F3,
    "cec2017_f4": CEC2017_F4,
    "cec2017_f5": CEC2017_F5,
    "cec2017_f6": CEC2017_F6,
    "cec2017_f7": CEC2017_F7,
    "cec2017_f8": CEC2017_F8,
    "cec2017_f9": CEC2017_F9,
    "cec2017_f10": CEC2017_F10,
}

# List for iteration
all_functions = [
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
]
