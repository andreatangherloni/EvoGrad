"""
Shared base for CEC 2017 functions.

Single source of truth for the bias, shift/rotation loading, default bounds, and
the shift+rotate transform used by the simple and hybrid CEC 2017 families
(previously copy-pasted in simple.py and hybrid.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor

from ..base import BenchmarkFunction
from . import data as cec_data


class CEC2017Base(BenchmarkFunction):
    """Base for shifted+rotated CEC 2017 functions (search space [-100, 100]^D)."""

    def __init__(
        self,
        func_num: int,
        n_var: int = 10,
        rotation: Optional[Tensor] = None,
        shift: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(n_var=n_var, xl=-100.0, xu=100.0)

        self.func_num = func_num
        self.bias = func_num * 100.0  # F_i* = i * 100

        if rotation is not None:
            self.rotation = rotation
        else:
            self.rotation = cec_data.get_rotation(func_num, n_var, seed=seed)

        if shift is not None:
            self.shift = shift
        else:
            self.shift = cec_data.get_shift(func_num, n_var, seed=seed)

    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)

    def _shift_rotate(self, x: Tensor) -> Tensor:
        """Apply the shift + rotation transform (single source of truth)."""
        shift = self.shift.to(x.device, x.dtype)
        rotation = self.rotation.to(x.device, x.dtype)
        return cec_data.shift_rotate(x, shift, rotation)
