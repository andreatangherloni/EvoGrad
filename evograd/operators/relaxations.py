"""
Differentiable relaxations for discrete operations.

This module provides continuous relaxations of discrete operations
used throughout EvoGrad operators, enabling gradient flow through
categorical and Bernoulli sampling.

Functions:
    - gumbel_softmax: Categorical sampling relaxation
    - binary_concrete: Bernoulli sampling relaxation
    - expand_param: Parameter expansion utility
"""

import torch
from torch import Tensor
from typing import Optional, Union

__all__ = [
    "gumbel_softmax",
    "binary_concrete", 
    "expand_param",
    "standard_normal",
    "log_param",
]


def gumbel_softmax(
    logits: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    dim: int = -1,
    eps: float = 1e-10,
) -> Tensor:
    """
    Gumbel-Softmax with straight-through estimator.
    
    Provides a differentiable approximation to categorical sampling.
    Forward pass returns hard one-hot vectors, backward pass uses
    soft gradients through the softmax.
    
    Args:
        logits: Unnormalized log probabilities [..., n_categories].
        temperature: Temperature for softmax (lower = harder).
        dim: Dimension to apply softmax.
        eps: Small constant for numerical stability.
    
    Returns:
        One-hot vectors with soft gradients [..., n_categories].
    """
    device = logits.device
    
    # Ensure temperature is on correct device
    if isinstance(temperature, Tensor):
        temperature = temperature.to(device)
    
    # Sample Gumbel noise
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    gumbels = -torch.log(-torch.log(u))
    
    # Softmax with temperature
    y_soft = torch.softmax((logits + gumbels) / temperature, dim=dim)
    
    # Straight-through: hard forward, soft backward
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    
    return (y_hard - y_soft).detach() + y_soft


def binary_concrete(
    logits: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    eps: float = 1e-10,
) -> Tensor:
    """
    Binary-Concrete (Gumbel-Sigmoid) with straight-through estimator.
    
    Provides a differentiable approximation to Bernoulli sampling.
    Forward pass returns hard 0/1 values, backward pass uses
    soft gradients through the sigmoid.
    
    Args:
        logits: Unnormalized log-odds [...].
        temperature: Temperature for sigmoid (lower = harder).
        eps: Small constant for numerical stability.
    
    Returns:
        Binary mask with soft gradients [...].
    """
    device = logits.device
    
    # Ensure temperature is on correct device
    if isinstance(temperature, Tensor):
        temperature = temperature.to(device)
    
    # Sample logistic noise (difference of Gumbels)
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    noise = torch.log(u) - torch.log(1 - u)
    
    # Sigmoid with temperature
    y_soft = torch.sigmoid((logits + noise) / temperature)
    
    # Straight-through: hard forward, soft backward
    y_hard = (y_soft > 0.5).float()
    
    return (y_hard - y_soft).detach() + y_soft


def expand_param(
    param: Union[Tensor, float, None],
    default: Union[Tensor, float],
    n_pop: int,
    n_var: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Expand a parameter to shape [N, D] supporting four configurations.
    
    Handles parameter expansion for per-individual and per-gene
    parameter support in operators like SHADE.
    
    Args:
        param: Optional override parameter. Can be:
            - None: Use default
            - scalar: Same value for all [N, D]
            - [D]: Per-gene, broadcast to [N, D]
            - [N]: Per-individual, broadcast to [N, D]
            - [N, D]: Use as-is
        default: Default value if param is None.
        n_pop: Number of individuals (N).
        n_var: Number of variables (D).
        device: Target device.
        dtype: Target dtype.
    
    Returns:
        Tensor of shape [N, D].
    
    Example:
        >>> # Per-individual CR for SHADE
        >>> cr = torch.rand(100)  # [N]
        >>> cr_expanded = expand_param(cr, 0.9, 100, 30, device, dtype)
        >>> # cr_expanded.shape == [100, 30]
    """
    val = default if param is None else param
    
    # Convert to tensor on correct device
    if not isinstance(val, Tensor):
        val = torch.tensor(val, device=device, dtype=dtype)
    else:
        val = val.to(device=device, dtype=dtype)
    
    # Expand based on input shape
    if val.dim() == 0:
        # Scalar -> [N, D]
        return val.expand(n_pop, n_var)
    elif val.dim() == 1:
        if val.shape[0] == n_var:
            # [D] -> [N, D] (per-gene)
            return val.unsqueeze(0).expand(n_pop, -1)
        elif val.shape[0] == n_pop:
            # [N] -> [N, D] (per-individual)
            return val.unsqueeze(1).expand(-1, n_var)
        else:
            raise ValueError(
                f"1D param must have size n_var={n_var} or n_pop={n_pop}, "
                f"got {val.shape[0]}"
            )
    elif val.dim() == 2:
        if val.shape == (n_pop, n_var):
            return val
        else:
            raise ValueError(
                f"2D param must have shape [{n_pop}, {n_var}], "
                f"got {list(val.shape)}"
            )
    else:
        raise ValueError(f"param must be 0D, 1D, or 2D, got {val.dim()}D")


def standard_normal(
    *shape: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Draw standard-normal (pathwise / reparameterization) noise.

    Single source for the ``torch.randn(..., device=, dtype=)`` reparameterization
    draw, reused by CMA-ES sampling, Gaussian mutation, and the samplers.
    """
    return torch.randn(*shape, device=device, dtype=dtype, generator=generator)


def log_param(
    value: Union[float, Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> "torch.nn.Parameter":
    """Learnable positive hyperparameter stored on the log scale.

    Single source for the repeated ``nn.Parameter(torch.tensor(v).log())`` idiom
    used for sigma, F, eta, temperature, etc. Recover the value via ``.exp()``.
    """
    return torch.nn.Parameter(torch.tensor(value, device=device, dtype=dtype).log())
