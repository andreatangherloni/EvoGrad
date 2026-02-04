"""
Device utilities for EvoGrad.

This module provides functions for:
- Automatic device detection (CUDA > MPS > CPU)
- Tensor conversion with device/dtype handling
- Device-aware operations

Design Goals
------------
- Seamless GPU acceleration when available
- Consistent API across different hardware backends
- Safe tensor conversion from various input types

Example
-------
>>> from evograd.utils.device import get_device, ensure_tensor
>>> device = get_device()  # Automatically selects best device
>>> x = ensure_tensor([1.0, 2.0, 3.0], device=device)
>>> x.device
device(type='cuda', index=0)  # or 'mps' or 'cpu'
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch

# Type alias for values that can be converted to tensors
TensorLike = Union[
    torch.Tensor,
    np.ndarray,
    float,
    int,
    List[float],
    List[int],
    Sequence[float],
    Sequence[int],
]


def get_device(
    preference: Optional[str] = None,
    fallback: str = "cpu",
) -> torch.device:
    """Get the best available device for computation.

    Priority order (when preference is None):
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU

    Parameters
    ----------
    preference:
        Explicit device preference. If specified and available, this device
        is returned. Options: "cuda", "mps", "cpu", or a specific device
        string like "cuda:0".
    fallback:
        Device to use if the preferred device is not available.
        Default is "cpu".

    Returns
    -------
    torch.device
        The selected device.

    Examples
    --------
    >>> device = get_device()  # Auto-detect best device
    >>> device = get_device("cuda")  # Prefer CUDA, fall back to CPU
    >>> device = get_device("cuda:1")  # Specific GPU
    """
    if preference is not None:
        # User specified a preference
        pref_lower = preference.lower()

        if pref_lower.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(preference)
            else:
                return torch.device(fallback)

        elif pref_lower == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device(fallback)

        elif pref_lower == "cpu":
            return torch.device("cpu")

        else:
            # Try to use the preference as-is
            try:
                return torch.device(preference)
            except RuntimeError:
                return torch.device(fallback)

    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_default_dtype() -> torch.dtype:
    """Get the default floating-point dtype for EvoGrad.

    Returns
    -------
    torch.dtype
        Default is torch.float32, which offers a good balance between
        precision and performance for evolutionary computation.
    """
    return torch.float32


def ensure_tensor(
    value: TensorLike,
    dim: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    copy: bool = False,
) -> torch.Tensor:
    """Convert a value to a tensor with specified properties.

    This function handles various input types and ensures the result
    has the correct device, dtype, and optionally shape.

    Parameters
    ----------
    value:
        Input value to convert. Can be:
        - A scalar (int, float)
        - A list or sequence of numbers
        - A numpy array
        - An existing torch tensor
    dim:
        If provided, the tensor is broadcast/repeated to have this length
        along the first (and only) dimension. Only valid for 1D outputs.
        If value is a scalar, it is repeated `dim` times.
        If value is already a 1D tensor of length `dim`, it is unchanged.
        If value is a 1D tensor of length 1, it is repeated `dim` times.
    device:
        Target device. If None, uses the input tensor's device (if it's
        already a tensor) or the default device.
    dtype:
        Target dtype. If None, uses float32 for floating-point values
        or the input tensor's dtype.
    copy:
        If True, always create a new tensor even if the input already
        satisfies all requirements.

    Returns
    -------
    torch.Tensor
        The converted tensor.

    Raises
    ------
    ValueError
        If `dim` is specified but the input cannot be broadcast to that shape.

    Examples
    --------
    >>> # Scalar to tensor
    >>> ensure_tensor(3.14)
    tensor(3.1400)

    >>> # Scalar broadcast to dimension
    >>> ensure_tensor(-100.0, dim=10)
    tensor([-100., -100., -100., -100., -100., -100., -100., -100., -100., -100.])

    >>> # List to tensor with device
    >>> ensure_tensor([1, 2, 3], device="cuda")
    tensor([1., 2., 3.], device='cuda:0')

    >>> # Numpy array conversion
    >>> import numpy as np
    >>> ensure_tensor(np.array([1.0, 2.0]))
    tensor([1., 2.])
    """
    # Determine target dtype
    if dtype is None:
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
        else:
            dtype = get_default_dtype()

    # Determine target device
    if device is None:
        if isinstance(value, torch.Tensor):
            device = value.device
        else:
            device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Convert to tensor
    if isinstance(value, torch.Tensor):
        tensor = value
        needs_conversion = (
            copy
            or tensor.device != device
            or tensor.dtype != dtype
        )
        if needs_conversion:
            tensor = tensor.to(device=device, dtype=dtype)
            if copy and tensor.data_ptr() == value.data_ptr():
                tensor = tensor.clone()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).to(device=device, dtype=dtype)
    elif isinstance(value, (int, float)):
        tensor = torch.tensor(value, device=device, dtype=dtype)
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value, device=device, dtype=dtype)
    else:
        # Try generic conversion
        try:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Cannot convert {type(value).__name__} to tensor: {e}"
            ) from e

    # Handle dimension broadcasting
    if dim is not None:
        dim = int(dim)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        if tensor.ndim == 0:
            # Scalar: repeat to create 1D tensor
            tensor = tensor.expand(dim).clone()
        elif tensor.ndim == 1:
            if tensor.shape[0] == dim:
                # Already correct size
                pass
            elif tensor.shape[0] == 1:
                # Single element: broadcast
                tensor = tensor.expand(dim).clone()
            else:
                raise ValueError(
                    f"Cannot broadcast tensor of shape {tuple(tensor.shape)} "
                    f"to dim={dim}. Expected shape ({dim},) or (1,)."
                )
        else:
            raise ValueError(
                f"Cannot broadcast {tensor.ndim}D tensor to 1D. "
                f"Got shape {tuple(tensor.shape)}."
            )

    return tensor


def ensure_bounds(
    lower: TensorLike,
    upper: TensorLike,
    dim: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert and validate lower/upper bounds.

    Parameters
    ----------
    lower:
        Lower bounds. Scalar or 1D tensor of length `dim`.
    upper:
        Upper bounds. Scalar or 1D tensor of length `dim`.
    dim:
        Number of dimensions (variables).
    device:
        Target device.
    dtype:
        Target dtype.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (lower, upper) tensors, each of shape (dim,).

    Raises
    ------
    ValueError
        If bounds have incompatible shapes or if lower > upper for any dimension.

    Examples
    --------
    >>> lb, ub = ensure_bounds(-100.0, 100.0, dim=10)
    >>> lb.shape, ub.shape
    (torch.Size([10]), torch.Size([10]))

    >>> lb, ub = ensure_bounds([-1, -2, -3], [1, 2, 3], dim=3)
    """
    lb = ensure_tensor(lower, dim=dim, device=device, dtype=dtype)
    ub = ensure_tensor(upper, dim=dim, device=device, dtype=dtype)

    # Validate bounds
    if torch.any(lb > ub):
        violations = (lb > ub).nonzero(as_tuple=False).view(-1)
        raise ValueError(
            f"Lower bounds must be <= upper bounds. "
            f"Violations at indices: {violations.tolist()}"
        )

    return lb, ub


def to_device(
    *tensors: torch.Tensor,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, ...]:
    """Move multiple tensors to a device.

    Parameters
    ----------
    *tensors:
        Tensors to move.
    device:
        Target device.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Tuple of tensors on the target device.

    Examples
    --------
    >>> x, y, z = to_device(x, y, z, device="cuda")
    """
    if isinstance(device, str):
        device = torch.device(device)

    return tuple(t.to(device) for t in tensors)


def sync_device() -> None:
    """Synchronize the current device (useful for timing).

    For CUDA devices, this calls torch.cuda.synchronize().
    For other devices, this is a no-op.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_info(device: Optional[Union[str, torch.device]] = None) -> dict[str, int]:
    """Get memory information for a device.

    Parameters
    ----------
    device:
        Device to query. If None, uses the default device.

    Returns
    -------
    dict[str, int]
        Dictionary with keys:
        - "allocated": Currently allocated memory (bytes)
        - "reserved": Currently reserved memory (bytes)
        - "max_allocated": Peak allocated memory (bytes)
        For CPU and MPS, returns empty dict or partial info.

    Examples
    --------
    >>> info = get_memory_info("cuda")
    >>> print(f"Allocated: {info['allocated'] / 1e9:.2f} GB")
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
        }
    elif device.type == "mps":
        # MPS has limited memory introspection
        try:
            return {
                "allocated": torch.mps.current_allocated_memory(),
            }
        except AttributeError:
            return {}
    else:
        return {}


def set_seed(
    seed: int,
    deterministic: bool = False,
) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed:
        Random seed value.
    deterministic:
        If True, enables deterministic algorithms in PyTorch.
        This may reduce performance but ensures reproducibility.

    Notes
    -----
    This sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and all CUDA devices)
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)


class DeviceContext:
    """Context manager for temporary device switching.

    This is useful when you need to perform operations on a specific
    device and want to ensure cleanup.

    Parameters
    ----------
    device:
        Device to use within the context.

    Examples
    --------
    >>> with DeviceContext("cuda"):
    ...     x = torch.randn(100, 100)  # Created on CUDA
    ...     result = x @ x.T
    """

    def __init__(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self._previous_device: Optional[int] = None

    def __enter__(self) -> torch.device:
        # Store current default device (if CUDA)
        if torch.cuda.is_available() and self.device.type == "cuda":
            self._previous_device = torch.cuda.current_device()
            if self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Restore previous device
        if self._previous_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(self._previous_device)
        return False  # Don't suppress exceptions


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# Default device (lazily initialized)
_default_device: Optional[torch.device] = None


def default_device() -> torch.device:
    """Get the module's default device (cached).

    This is initialized once on first call and reused. Use `get_device()`
    if you need fresh detection or want to specify preferences.
    """
    global _default_device
    if _default_device is None:
        _default_device = get_device()
    return _default_device


def reset_default_device() -> None:
    """Reset the cached default device.

    Call this if the hardware configuration has changed and you want
    to re-detect the best device.
    """
    global _default_device
    _default_device = None