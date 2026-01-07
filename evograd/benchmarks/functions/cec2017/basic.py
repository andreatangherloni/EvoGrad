"""
CEC 2017 Basic Functions.

Core mathematical functions used by CEC 2017 benchmark suite.
These are stateless functions that operate on already-transformed inputs.

Reference: CEC 2017 Competition on Real-Parameter Single Objective Optimization
"""

import torch
from torch import Tensor
from typing import Optional


def bent_cigar(x: Tensor) -> Tensor:
    """
    Bent Cigar function.
    f(x) = x_1^2 + 10^7 * sum(x_i^2) for i > 1
    """
    return x[..., 0] ** 2 + 10e6 * (x[..., 1:] ** 2).sum(dim=-1)


def sum_diff_pow(x: Tensor) -> Tensor:
    """
    Sum of Different Powers function.
    f(x) = sum(|x_i|^(i+1))
    """
    n = x.shape[-1]
    i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (torch.abs(x) ** i).sum(dim=-1)


def zakharov(x: Tensor) -> Tensor:
    """
    Zakharov function.
    f(x) = sum(x_i^2) + (0.5 * sum(i * x_i))^2 + (0.5 * sum(i * x_i))^4
    """
    n = x.shape[-1]
    i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    sum_sq = (x ** 2).sum(dim=-1)
    sum_ix = (i * x).sum(dim=-1)
    term = 0.5 * sum_ix
    return sum_sq + term ** 2 + term ** 4


def rosenbrock(x: Tensor) -> Tensor:
    """
    Rosenbrock function (CEC scaled version).
    Scales input by 0.02048 and shifts by 1.
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """
    x = 0.02048 * x + 1.0
    x_i = x[..., :-1]
    x_ip1 = x[..., 1:]
    term1 = 100 * (x_ip1 - x_i ** 2) ** 2
    term2 = (x_i - 1) ** 2
    return (term1 + term2).sum(dim=-1)


def rastrigin(x: Tensor) -> Tensor:
    """
    Rastrigin function (CEC scaled version).
    Scales input by 0.0512.
    f(x) = sum(x_i^2 - 10*cos(2*pi*x_i) + 10)
    """
    x = 0.0512 * x
    return (x ** 2 - 10 * torch.cos(2 * torch.pi * x) + 10).sum(dim=-1)


def expanded_schaffers_f6(x: Tensor) -> Tensor:
    """
    Expanded Schaffer's F6 function.
    f(x) = sum(0.5 + (sin^2(sqrt(x_i^2 + x_{i+1}^2)) - 0.5) / (1 + 0.001*(x_i^2 + x_{i+1}^2))^2)
    """
    x_i = x[..., :-1]
    x_ip1 = x[..., 1:]
    t = x_i ** 2 + x_ip1 ** 2
    sin_term = torch.sin(torch.sqrt(t)) ** 2 - 0.5
    denom = (1 + 0.001 * t) ** 2
    return (0.5 + sin_term / denom).sum(dim=-1)


def lunacek_bi_rastrigin(
    x: Tensor,
    shift: Optional[Tensor] = None,
    rotation: Optional[Tensor] = None,
) -> Tensor:
    """
    Lunacek Bi-Rastrigin function.
    A special case that requires shift and rotation to be passed directly.
    """
    nx = x.shape[-1]
    batch_shape = x.shape[:-1]
    
    if shift is None:
        shift = torch.zeros(nx, device=x.device, dtype=x.dtype)
    
    # Ensure shift has correct shape for broadcasting
    shift = shift.to(x.device, x.dtype)
    if shift.dim() == 1:
        shift = shift.unsqueeze(0)  # [1, nx]
    
    # Calculate coefficients
    mu0 = 2.5
    s = 1 - 1 / (2 * ((nx + 20) ** 0.5) - 8.2)
    mu1 = -((mu0 * mu0 - 1) / s) ** 0.5
    
    # Shift and scale
    y = 0.1 * (x - shift)
    
    tmpx = 2 * y.clone()
    # Flip sign where shift < 0
    mask = (shift < 0).expand_as(tmpx)
    tmpx = torch.where(mask, -tmpx, tmpx)
    
    z = tmpx.clone()
    tmpx = tmpx + mu0
    
    # Term 1: sum((tmpx - mu0)^2)
    t1 = ((tmpx - mu0) ** 2).sum(dim=-1)
    
    # Term 2: s * sum((tmpx - mu1)^2) + nx
    t2 = s * ((tmpx - mu1) ** 2).sum(dim=-1) + nx
    
    # Apply rotation if provided
    if rotation is not None:
        rotation = rotation.to(x.device, x.dtype)
        # z @ R.T for batch matrix multiplication
        z = z @ rotation.T
    
    # Cosine term
    cos_term = torch.cos(2.0 * torch.pi * z).sum(dim=-1)
    
    # Result
    result = torch.minimum(t1, t2) + 10.0 * (nx - cos_term)
    return result


def non_cont_rastrigin(
    x: Tensor,
    shift: Optional[Tensor] = None,
    rotation: Optional[Tensor] = None,
) -> Tensor:
    """
    Non-Continuous Rastrigin function.
    A special case that requires shift and rotation to be passed directly.
    """
    nx = x.shape[-1]
    
    if shift is None:
        shift = torch.zeros(nx, device=x.device, dtype=x.dtype)
    
    shift = shift.to(x.device, x.dtype)
    if shift.dim() == 1:
        shift = shift.unsqueeze(0)
    
    shifted = x - shift
    
    # Apply non-continuity
    x_mod = x.clone()
    mask = torch.abs(shifted) > 0.5
    x_mod = torch.where(
        mask,
        shift + torch.floor(2 * shifted + 0.5) * 0.5,
        x_mod
    )
    
    # Scale
    z = 0.0512 * shifted
    
    # Apply rotation if provided
    if rotation is not None:
        rotation = rotation.to(x.device, x.dtype)
        z = z @ rotation.T
    
    # Rastrigin formula
    result = (z ** 2 - 10 * torch.cos(2 * torch.pi * z) + 10).sum(dim=-1)
    return result


def levy(x: Tensor) -> Tensor:
    """
    Levy function (CEC version without scaling).
    """
    w = 1.0 + 0.25 * (x - 1.0)
    
    term1 = torch.sin(torch.pi * w[..., 0]) ** 2
    
    w_inner = w[..., :-1]
    term2 = ((w_inner - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w_inner + 1) ** 2)).sum(dim=-1)
    
    term3 = (w[..., -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[..., -1]) ** 2)
    
    return term1 + term2 + term3


def modified_schwefel(x: Tensor) -> Tensor:
    """
    Modified Schwefel function.
    """
    nx = x.shape[-1]
    x = 10.0 * x  # Scale to search range
    
    z = x + 420.9687462275036
    
    # Initialize result
    result = z * torch.sin(torch.sqrt(torch.abs(z)))
    
    # Handle z < -500
    mask1 = z < -500
    zm1 = torch.fmod(torch.abs(z), 500.0) - 500
    penalty1 = (z + 500) ** 2 / (10000 * nx)
    result = torch.where(mask1, zm1 * torch.sin(torch.sqrt(torch.abs(zm1))) - penalty1, result)
    
    # Handle z > 500
    mask2 = z > 500
    zm2 = 500 - torch.fmod(torch.abs(z), 500.0)
    penalty2 = (z - 500) ** 2 / (10000 * nx)
    result = torch.where(mask2, zm2 * torch.sin(torch.sqrt(torch.abs(zm2))) - penalty2, result)
    
    return 418.9829 * nx - result.sum(dim=-1)


def high_conditioned_elliptic(x: Tensor) -> Tensor:
    """
    High Conditioned Elliptic function.
    f(x) = sum(10^(6*(i-1)/(n-1)) * x_i^2)
    """
    n = x.shape[-1]
    i = torch.arange(n, device=x.device, dtype=x.dtype)
    factor = 6.0 / max(n - 1, 1)
    weights = 10 ** (i * factor)
    return (weights * x ** 2).sum(dim=-1)


def discus(x: Tensor) -> Tensor:
    """
    Discus function.
    f(x) = 10^6 * x_1^2 + sum(x_i^2) for i > 1
    """
    return 1e6 * x[..., 0] ** 2 + (x[..., 1:] ** 2).sum(dim=-1)


def ackley(x: Tensor) -> Tensor:
    """
    Ackley function.
    """
    n = x.shape[-1]
    sum_sq = (x ** 2).sum(dim=-1)
    sum_cos = torch.cos(2 * torch.pi * x).sum(dim=-1)
    return -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / n)) - torch.exp(sum_cos / n) + 20 + torch.e


def weierstrass(x: Tensor) -> Tensor:
    """
    Weierstrass function (CEC scaled version).
    Scales input by 0.005.
    """
    x = 0.005 * x
    nx = x.shape[-1]
    
    k = torch.arange(0, 21, device=x.device, dtype=x.dtype)
    a_k = 0.5 ** k
    b_k = torch.pi * (3.0 ** k)
    
    # Compute constant term
    const = (a_k * torch.cos(b_k)).sum()
    
    # Compute sum over dimensions
    # Shape: [..., nx, 21]
    inner = 2 * (x.unsqueeze(-1) + 0.5) * b_k
    result = (a_k * torch.cos(inner)).sum(dim=-1).sum(dim=-1)
    
    return result - nx * const


def griewank(x: Tensor) -> Tensor:
    """
    Griewank function (CEC scaled version).
    Scales input by 6.0.
    """
    x = 6.0 * x
    nx = x.shape[-1]
    i = torch.arange(1, nx + 1, device=x.device, dtype=x.dtype)
    
    sum_term = (x ** 2).sum(dim=-1) / 4000
    prod_term = torch.cos(x / torch.sqrt(i)).prod(dim=-1)
    
    return sum_term - prod_term + 1


def katsuura(x: Tensor) -> Tensor:
    """
    Katsuura function (CEC scaled version).
    Scales input by 0.05.
    """
    x = 0.05 * x
    nx = x.shape[-1]
    pw = 10.0 / (nx ** 1.2)
    
    # j = 1, 2, ..., 32
    j = torch.arange(1, 33, device=x.device, dtype=x.dtype)
    tj = 2.0 ** j  # [32]
    
    # tjx: [..., nx, 32]
    tjx = tj * x.unsqueeze(-1)
    t = torch.abs(tjx - torch.round(tjx)) / tj
    tsm = t.sum(dim=-1)  # [..., nx]
    
    # Product term
    i = torch.arange(1, nx + 1, device=x.device, dtype=x.dtype)
    prd = ((1 + i * tsm) ** pw).prod(dim=-1)
    
    df = 10.0 / (nx * nx)
    return df * prd - df


def happy_cat(x: Tensor) -> Tensor:
    """
    Happy Cat function (CEC scaled version).
    Scales input by 0.05 and shifts by -1.
    """
    x = 0.05 * x - 1
    nx = x.shape[-1]
    
    sum_x = x.sum(dim=-1)
    sum_sq = (x ** 2).sum(dim=-1)
    
    return torch.abs(sum_sq - nx) ** 0.25 + (0.5 * sum_sq + sum_x) / nx + 0.5


def h_g_bat(x: Tensor) -> Tensor:
    """
    HGBat function (CEC scaled version).
    Scales input by 0.05 and shifts by -1.
    """
    x = 0.05 * x - 1
    nx = x.shape[-1]
    
    sum_x = x.sum(dim=-1)
    sum_sq = (x ** 2).sum(dim=-1)
    
    return torch.abs(sum_sq ** 2 - sum_x ** 2) ** 0.5 + (0.5 * sum_sq + sum_x) / nx + 0.5


def expanded_griewanks_plus_rosenbrock(x: Tensor) -> Tensor:
    """
    Expanded Griewank's plus Rosenbrock function (CEC scaled version).
    Scales input by 0.05 and shifts by 1.
    """
    x = 0.05 * x + 1
    
    # Rosenbrock part for consecutive pairs
    x_i = x[..., :-1]
    x_ip1 = x[..., 1:]
    tmp1 = x_i ** 2 - x_ip1
    tmp2 = x_i - 1.0
    temp = 100 * tmp1 ** 2 + tmp2 ** 2
    
    # Griewank applied to Rosenbrock result
    grie = temp ** 2 / 4000 - torch.cos(temp) + 1
    
    # Wrap-around term (last to first)
    tmp1_wrap = x[..., -1:] ** 2 - x[..., :1]
    tmp2_wrap = x[..., -1:] - 1.0
    temp_wrap = 100 * tmp1_wrap ** 2 + tmp2_wrap ** 2
    grie_wrap = temp_wrap ** 2 / 4000 - torch.cos(temp_wrap) + 1
    
    return grie.sum(dim=-1) + grie_wrap.sum(dim=-1)


def schaffers_f7(x: Tensor) -> Tensor:
    """
    Schaffer's F7 function.
    """
    nx = x.shape[-1]
    
    x_i = x[..., :-1]
    x_ip1 = x[..., 1:]
    
    si = torch.sqrt(x_i ** 2 + x_ip1 ** 2)
    tmp = torch.sin(50 * si ** 0.2)
    
    # Note: Original CEC code has tmp squared (appears to be intentional)
    sm = (torch.sqrt(si) * (tmp ** 2 + 1)).sum(dim=-1)
    
    denom = (nx - 1) ** 2
    return (sm ** 2) / denom


# Registry of all basic functions
BASIC_FUNCTIONS = {
    "bent_cigar": bent_cigar,
    "sum_diff_pow": sum_diff_pow,
    "zakharov": zakharov,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "expanded_schaffers_f6": expanded_schaffers_f6,
    "lunacek_bi_rastrigin": lunacek_bi_rastrigin,
    "non_cont_rastrigin": non_cont_rastrigin,
    "levy": levy,
    "modified_schwefel": modified_schwefel,
    "high_conditioned_elliptic": high_conditioned_elliptic,
    "discus": discus,
    "ackley": ackley,
    "weierstrass": weierstrass,
    "griewank": griewank,
    "katsuura": katsuura,
    "happy_cat": happy_cat,
    "h_g_bat": h_g_bat,
    "expanded_griewanks_plus_rosenbrock": expanded_griewanks_plus_rosenbrock,
    "schaffers_f7": schaffers_f7,
}
