"""
CEC 2017 Data Loader.

Loads rotation matrices, shift vectors, and shuffle permutations from data.pkl.
Also provides utilities to generate random transforms if data file is not available.

The data.pkl file contains pre-computed transforms for reproducibility with
the official CEC 2017 benchmark suite.
"""

import os
import pickle
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


# =============================================================================
# DATA LOADING
# =============================================================================

_DATA_LOADED = False
_DATA_CACHE: Dict = {}

# Supported dimensions
SUPPORTED_DIMS = [2, 10, 20, 30, 50, 100]


def _load_data():
    """Load data from pickle file if available."""
    global _DATA_LOADED, _DATA_CACHE
    
    if _DATA_LOADED:
        return
    
    data_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
    
    if os.path.exists(data_path):
        try:
            with open(data_path, 'rb') as f:
                _DATA_CACHE = pickle.load(f)
            _DATA_LOADED = True
        except Exception as e:
            print(f"Warning: Could not load CEC 2017 data file: {e}")
            print("Using randomly generated transforms instead.")
            _DATA_LOADED = False
    else:
        _DATA_LOADED = False


def data_available() -> bool:
    """Check if official CEC 2017 data is available."""
    _load_data()
    return _DATA_LOADED and len(_DATA_CACHE) > 0


# =============================================================================
# ROTATION MATRICES
# =============================================================================

def get_rotation(func_num: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get rotation matrix for a simple/hybrid function (f1-f20).
    
    Args:
        func_num: Function number (1-20, 0-indexed internally).
        n_var: Number of variables (2, 10, 20, 30, 50, or 100).
        seed: Random seed for generating matrix if data not available.
    
    Returns:
        Rotation matrix of shape [n_var, n_var].
    """
    _load_data()
    
    func_idx = func_num - 1  # Convert to 0-indexed
    
    if _DATA_LOADED and n_var in SUPPORTED_DIMS:
        key = f'M_D{n_var}'
        if key in _DATA_CACHE:
            mat = _DATA_CACHE[key][func_idx]
            return torch.tensor(mat, dtype=torch.float32)
    
    # Generate random orthogonal matrix
    return generate_rotation_matrix(n_var, seed=seed)


def get_rotation_cf(func_num: int, comp_idx: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get rotation matrix for a composition function (f21-f30).
    
    Args:
        func_num: Function number (21-30, 0-indexed internally as 0-9).
        comp_idx: Component index within the composition (0-9).
        n_var: Number of variables (2, 10, 20, 30, 50, or 100).
        seed: Random seed for generating matrix if data not available.
    
    Returns:
        Rotation matrix of shape [n_var, n_var].
    """
    _load_data()
    
    func_idx = func_num - 21  # Convert to 0-indexed for composition functions
    
    if _DATA_LOADED and n_var in SUPPORTED_DIMS:
        # Try both key formats
        key = f'M_cf_D{n_var}' if n_var > 2 else f'M_cf_d{n_var}'
        if key not in _DATA_CACHE:
            key = f'M_cf_D{n_var}'
        if key in _DATA_CACHE:
            mat = _DATA_CACHE[key][func_idx][comp_idx]
            return torch.tensor(mat, dtype=torch.float32)
    
    # Generate random orthogonal matrix
    return generate_rotation_matrix(n_var, seed=seed)


# =============================================================================
# SHIFT VECTORS
# =============================================================================

def get_shift(func_num: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get shift vector for a simple/hybrid function (f1-f20).
    
    Args:
        func_num: Function number (1-20, 0-indexed internally).
        n_var: Number of variables.
        seed: Random seed for generating vector if data not available.
    
    Returns:
        Shift vector of shape [n_var].
    """
    _load_data()
    
    func_idx = func_num - 1  # Convert to 0-indexed
    
    if _DATA_LOADED and 'shift' in _DATA_CACHE:
        shift = _DATA_CACHE['shift'][func_idx][:n_var]
        return torch.tensor(shift, dtype=torch.float32)
    
    # Generate random shift vector within bounds
    return generate_shift_vector(n_var, -80.0, 80.0, seed=seed)


def get_shift_cf(func_num: int, comp_idx: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get shift vector for a composition function (f21-f30).
    
    Args:
        func_num: Function number (21-30, 0-indexed internally as 0-9).
        comp_idx: Component index within the composition (0-9).
        n_var: Number of variables.
        seed: Random seed for generating vector if data not available.
    
    Returns:
        Shift vector of shape [n_var].
    """
    _load_data()
    
    func_idx = func_num - 21  # Convert to 0-indexed
    
    if _DATA_LOADED and 'shift_cf' in _DATA_CACHE:
        shift = _DATA_CACHE['shift_cf'][func_idx][comp_idx][:n_var]
        return torch.tensor(shift, dtype=torch.float32)
    
    # Generate random shift vector
    return generate_shift_vector(n_var, -80.0, 80.0, seed=seed)


# =============================================================================
# SHUFFLE VECTORS
# =============================================================================

def get_shuffle(func_num: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get shuffle permutation for a hybrid function (f11-f20).
    
    Args:
        func_num: Function number (11-20, 0-indexed internally as 0-9).
        n_var: Number of variables (10, 30, 50, or 100).
        seed: Random seed for generating permutation if data not available.
    
    Returns:
        Shuffle permutation of shape [n_var] (0-indexed).
    """
    _load_data()
    
    func_idx = func_num - 11  # Convert to 0-indexed for hybrid functions
    
    if _DATA_LOADED and n_var in [10, 30, 50, 100]:
        key = f'shuffle_D{n_var}'
        if key in _DATA_CACHE:
            shuffle = _DATA_CACHE[key][func_idx]
            return torch.tensor(shuffle, dtype=torch.long)
    
    # Generate random permutation
    return generate_shuffle(n_var, seed=seed)


def get_shuffle_cf(func_num: int, comp_idx: int, n_var: int, seed: Optional[int] = None) -> Tensor:
    """
    Get shuffle permutation for composition function f29 or f30.
    
    Args:
        func_num: Function number (29 or 30).
        comp_idx: Component index (0-2).
        n_var: Number of variables (10, 30, 50, or 100).
        seed: Random seed for generating permutation if data not available.
    
    Returns:
        Shuffle permutation of shape [n_var] (0-indexed).
    """
    _load_data()
    
    func_idx = func_num - 29  # 0 for f29, 1 for f30
    
    if _DATA_LOADED and n_var in [10, 30, 50, 100]:
        key = f'shuffle_cf_D{n_var}'
        if key in _DATA_CACHE:
            shuffle = _DATA_CACHE[key][func_idx][comp_idx]
            return torch.tensor(shuffle, dtype=torch.long)
    
    # Generate random permutation
    return generate_shuffle(n_var, seed=seed)


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def generate_rotation_matrix(n: int, seed: Optional[int] = None) -> Tensor:
    """
    Generate a random orthogonal rotation matrix using QR decomposition.
    
    Args:
        n: Dimension of the matrix.
        seed: Optional random seed.
    
    Returns:
        Orthogonal matrix of shape [n, n].
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    A = torch.randn(n, n)
    Q, R = torch.linalg.qr(A)
    
    # Ensure proper rotation (det = 1)
    d = torch.diag(R)
    ph = torch.sign(d)
    Q = Q * ph.unsqueeze(0)
    
    return Q


def generate_shift_vector(
    n: int,
    xl: float = -80.0,
    xu: float = 80.0,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate a random shift vector within bounds.
    
    Args:
        n: Dimension of the vector.
        xl: Lower bound.
        xu: Upper bound.
        seed: Optional random seed.
    
    Returns:
        Shift vector of shape [n].
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    return xl + (xu - xl) * torch.rand(n)


def generate_shuffle(n: int, seed: Optional[int] = None) -> Tensor:
    """
    Generate a random permutation.
    
    Args:
        n: Length of permutation.
        seed: Optional random seed.
    
    Returns:
        Permutation of shape [n] (0-indexed).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    return torch.randperm(n)


# =============================================================================
# TRANSFORM UTILITIES
# =============================================================================

def shift_rotate(x: Tensor, shift: Tensor, rotation: Tensor) -> Tensor:
    """
    Apply shift and rotation transformation.
    
    z = R @ (x - shift)
    
    Args:
        x: Input tensor of shape [..., n_var].
        shift: Shift vector of shape [n_var].
        rotation: Rotation matrix of shape [n_var, n_var].
    
    Returns:
        Transformed tensor of shape [..., n_var].
    """
    shifted = x - shift
    return shifted @ rotation.T


def shuffle_and_partition(
    x: Tensor,
    shuffle: Tensor,
    partitions: list,
) -> list:
    """
    Apply shuffle permutation and partition into parts.
    
    Args:
        x: Input tensor of shape [..., n_var].
        shuffle: Permutation of shape [n_var] (0-indexed).
        partitions: List of partition fractions (should sum to 1.0).
    
    Returns:
        List of tensors, one per partition.
    """
    nx = x.shape[-1]
    
    # Apply shuffle
    x_shuffled = x[..., shuffle]
    
    # Partition
    parts = []
    start = 0
    for i, p in enumerate(partitions[:-1]):
        end = start + int(torch.ceil(torch.tensor(p * nx)).item())
        parts.append(x_shuffled[..., start:end])
        start = end
    parts.append(x_shuffled[..., start:])
    
    return parts
