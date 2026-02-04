"""
Classical benchmark functions for optimization.

This module provides standard unimodal and multimodal test functions
commonly used in evolutionary computation literature.

Categories:
- Unimodal: Sphere, Ellipsoid, Schwefel222, Cigar, Discus, Rosenbrock
- Multimodal: Rastrigin, Ackley, Griewank, Schwefel, Levy, Michalewicz
- Other: Zakharov, Dixon-Price, Powell, Trid
"""

from typing import Tuple

import torch
from torch import Tensor

from .base import BenchmarkFunction


# =============================================================================
# UNIMODAL FUNCTIONS
# =============================================================================

class Sphere(BenchmarkFunction):
    """
    Sphere function (De Jong's function 1).
    
    f(x) = sum(x_i^2)
    
    Properties:
        - Unimodal, separable, convex
        - Continuous, differentiable
        - Global minimum: f(0,...,0) = 0
    """
    name = "sphere"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return (x ** 2).sum(dim=-1)


class Ellipsoid(BenchmarkFunction):
    """
    Ellipsoid function (Schwefel's problem 1.2 variant).
    
    f(x) = sum(10^(6*(i-1)/(n-1)) * x_i^2)
    
    Properties:
        - Unimodal, separable
        - Ill-conditioned (condition number ~10^6)
        - Global minimum: f(0,...,0) = 0
    """
    name = "ellipsoid"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        i = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = 10 ** (6 * i / max(n - 1, 1))
        return (weights * x ** 2).sum(dim=-1)


class SumOfDifferentPowers(BenchmarkFunction):
    """
    Sum of Different Powers function.
    
    f(x) = sum(|x_i|^(i+1))
    
    Properties:
        - Unimodal, separable
        - Different sensitivity per dimension
        - Global minimum: f(0,...,0) = 0
    """
    name = "sum_of_different_powers"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-1.0, 1.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        return (torch.abs(x) ** i).sum(dim=-1)


class Schwefel222(BenchmarkFunction):
    """
    Schwefel's Problem 2.22.
    
    f(x) = sum(|x_i|) + prod(|x_i|)
    
    Properties:
        - Unimodal, non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "schwefel222"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        abs_x = torch.abs(x)
        return abs_x.sum(dim=-1) + abs_x.prod(dim=-1)


class Cigar(BenchmarkFunction):
    """
    Cigar function.
    
    f(x) = x_1^2 + 10^6 * sum(x_i^2) for i>1
    
    Properties:
        - Unimodal, separable
        - Ill-conditioned (one narrow direction)
        - Global minimum: f(0,...,0) = 0
    """
    name = "cigar"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return x[..., 0] ** 2 + 1e6 * (x[..., 1:] ** 2).sum(dim=-1)


class Discus(BenchmarkFunction):
    """
    Discus (Tablet) function.
    
    f(x) = 10^6 * x_1^2 + sum(x_i^2) for i>1
    
    Properties:
        - Unimodal, separable
        - Ill-conditioned (one dominant direction)
        - Global minimum: f(0,...,0) = 0
    """
    name = "discus"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return 1e6 * x[..., 0] ** 2 + (x[..., 1:] ** 2).sum(dim=-1)


class BentCigar(BenchmarkFunction):
    """
    Bent Cigar function.
    
    f(x) = x_1^2 + 10^6 * sum(x_i^2) for i>1
    
    Properties:
        - Unimodal
        - Non-separable when rotated
        - Global minimum: f(0,...,0) = 0
    """
    name = "bent_cigar"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return x[..., 0] ** 2 + 1e6 * (x[..., 1:] ** 2).sum(dim=-1)


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function (Banana function).
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Properties:
        - Unimodal (for n<=3), multimodal (for n>3)
        - Non-separable
        - Global minimum: f(1,...,1) = 0
    """
    name = "rosenbrock"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-5.0, 10.0)
    
    def _compute_optimal_x(self) -> Tensor:
        return torch.ones(self.n_var)
    
    def __call__(self, x: Tensor) -> Tensor:
        x_i = x[..., :-1]
        x_ip1 = x[..., 1:]
        return (100 * (x_ip1 - x_i ** 2) ** 2 + (1 - x_i) ** 2).sum(dim=-1)


class DixonPrice(BenchmarkFunction):
    """
    Dixon-Price function.
    
    f(x) = (x_1 - 1)^2 + sum(i * (2*x_i^2 - x_{i-1})^2)
    
    Properties:
        - Unimodal
        - Non-separable
    """
    name = "dixon_price"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
    
    def _compute_optimal_x(self) -> Tensor:
        # x_i = 2^(-(2^i - 2)/2^i) for i = 1, ..., n
        i = torch.arange(1, self.n_var + 1, dtype=torch.float32)
        return 2.0 ** (-(2.0 ** i - 2.0) / (2.0 ** i))
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        term1 = (x[..., 0] - 1) ** 2
        i = torch.arange(2, n + 1, device=x.device, dtype=x.dtype)
        term2 = (i * (2 * x[..., 1:] ** 2 - x[..., :-1]) ** 2).sum(dim=-1)
        return term1 + term2


class Powell(BenchmarkFunction):
    """
    Powell function.
    
    f(x) = sum of grouped terms (requires n divisible by 4)
    
    Properties:
        - Unimodal
        - Non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "powell"
    optimal_value = 0.0
    
    def __init__(self, n_var: int = 24, **kwargs):
        # Ensure n_var is divisible by 4
        n_var = max(4, (n_var // 4) * 4)
        super().__init__(n_var=n_var, **kwargs)
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-4.0, 5.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        n_groups = n // 4
        
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        
        for j in range(n_groups):
            i = 4 * j
            term1 = (x[..., i] + 10 * x[..., i + 1]) ** 2
            term2 = 5 * (x[..., i + 2] - x[..., i + 3]) ** 2
            term3 = (x[..., i + 1] - 2 * x[..., i + 2]) ** 4
            term4 = 10 * (x[..., i] - x[..., i + 3]) ** 4
            result = result + term1 + term2 + term3 + term4
        
        return result


class Trid(BenchmarkFunction):
    """
    Trid function.
    
    f(x) = sum((x_i - 1)^2) - sum(x_i * x_{i-1})
    
    Properties:
        - Unimodal
        - Non-separable
    """
    name = "trid"
    
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        # Optimal value is -n(n+4)(n-1)/6
        self.optimal_value = -n_var * (n_var + 4) * (n_var - 1) / 6
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-self.n_var ** 2, self.n_var ** 2)
    
    def _compute_optimal_x(self) -> Tensor:
        # x_i = i(n + 1 - i)
        i = torch.arange(1, self.n_var + 1, dtype=torch.float32)
        return i * (self.n_var + 1 - i)
    
    def __call__(self, x: Tensor) -> Tensor:
        term1 = ((x - 1) ** 2).sum(dim=-1)
        term2 = (x[..., 1:] * x[..., :-1]).sum(dim=-1)
        return term1 - term2


# =============================================================================
# MULTIMODAL FUNCTIONS
# =============================================================================

class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function.
    
    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Properties:
        - Highly multimodal (10^n local minima)
        - Separable
        - Regular spacing of local minima
        - Global minimum: f(0,...,0) = 0
    """
    name = "rastrigin"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-5.12, 5.12)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        return 10 * n + (x ** 2 - 10 * torch.cos(2 * torch.pi * x)).sum(dim=-1)


class Ackley(BenchmarkFunction):
    """
    Ackley function.
    
    f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
    
    Properties:
        - Multimodal with large nearly flat outer region
        - Non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "ackley"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-32.768, 32.768)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        sum1 = (x ** 2).sum(dim=-1)
        sum2 = torch.cos(2 * torch.pi * x).sum(dim=-1)
        return (
            -20 * torch.exp(-0.2 * torch.sqrt(sum1 / n))
            - torch.exp(sum2 / n)
            + 20
            + torch.e
        )


class Griewank(BenchmarkFunction):
    """
    Griewank function.
    
    f(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1
    
    Properties:
        - Multimodal with regular local minima
        - Non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "griewank"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-600.0, 600.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        sum_term = (x ** 2).sum(dim=-1) / 4000
        prod_term = torch.cos(x / torch.sqrt(i)).prod(dim=-1)
        return sum_term - prod_term + 1


class Schwefel(BenchmarkFunction):
    """
    Schwefel function.
    
    f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    
    Properties:
        - Multimodal
        - Separable
        - Global optimum far from local optima
        - Global minimum: f(420.9687,...,420.9687) ≈ 0
    """
    name = "schwefel"
    
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.optimal_value = 0.0  # After normalization
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-500.0, 500.0)
    
    def _compute_optimal_x(self) -> Tensor:
        return torch.full((self.n_var,), 420.9687)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        return 418.9829 * n - (x * torch.sin(torch.sqrt(torch.abs(x)))).sum(dim=-1)


class Levy(BenchmarkFunction):
    """
    Levy function.
    
    f(x) = sin^2(pi*w_1) + sum((w_i-1)^2 * (1 + 10*sin^2(pi*w_i+1))) 
           + (w_n-1)^2 * (1 + sin^2(2*pi*w_n))
    where w_i = 1 + (x_i - 1)/4
    
    Properties:
        - Multimodal
        - Non-separable
        - Global minimum: f(1,...,1) = 0
    """
    name = "levy"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
    
    def _compute_optimal_x(self) -> Tensor:
        return torch.ones(self.n_var)
    
    def __call__(self, x: Tensor) -> Tensor:
        w = 1 + (x - 1) / 4
        term1 = torch.sin(torch.pi * w[..., 0]) ** 2
        term2 = (
            (w[..., :-1] - 1) ** 2
            * (1 + 10 * torch.sin(torch.pi * w[..., :-1] + 1) ** 2)
        ).sum(dim=-1)
        term3 = (w[..., -1] - 1) ** 2 * (
            1 + torch.sin(2 * torch.pi * w[..., -1]) ** 2
        )
        return term1 + term2 + term3


class Michalewicz(BenchmarkFunction):
    """
    Michalewicz function.
    
    f(x) = -sum(sin(x_i) * sin(i*x_i^2 / pi)^(2*m))
    
    Properties:
        - Multimodal with n! local minima
        - Separable
        - Steepness controlled by parameter m
    """
    name = "michalewicz"
    
    def __init__(self, n_var: int = 30, m: float = 10.0, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.m = m
        # Optimal value depends on n and m
        self.optimal_value = -4.687 if n_var >= 5 else -1.8013  # Approximate
    
    def default_bounds(self) -> Tuple[float, float]:
        return (0.0, torch.pi)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        return -(
            torch.sin(x) * torch.sin(i * x ** 2 / torch.pi) ** (2 * self.m)
        ).sum(dim=-1)


class Zakharov(BenchmarkFunction):
    """
    Zakharov function.
    
    f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
    
    Properties:
        - Multimodal
        - Non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "zakharov"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-5.0, 10.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        i = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        sum1 = (x ** 2).sum(dim=-1)
        sum2 = (0.5 * i * x).sum(dim=-1)
        return sum1 + sum2 ** 2 + sum2 ** 4


class Weierstrass(BenchmarkFunction):
    """
    Weierstrass function.
    
    f(x) = sum_i(sum_k(a^k * cos(2*pi*b^k*(x_i + 0.5)))) 
           - n * sum_k(a^k * cos(pi*b^k))
    
    Properties:
        - Multimodal
        - Continuous but not differentiable
        - Non-separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "weierstrass"
    optimal_value = 0.0
    
    def __init__(self, n_var: int = 30, a: float = 0.5, b: float = 3.0, k_max: int = 20, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.a = a
        self.b = b
        self.k_max = k_max
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-0.5, 0.5)
    
    def __call__(self, x: Tensor) -> Tensor:
        n = x.shape[-1]
        k = torch.arange(self.k_max + 1, device=x.device, dtype=x.dtype)
        
        a_k = self.a ** k  # Shape: [k_max+1]
        b_k = self.b ** k  # Shape: [k_max+1]
        
        # Constant term: sum_k(a^k * cos(pi * b^k))
        const = (a_k * torch.cos(torch.pi * b_k)).sum()
        
        # Sum over dimensions
        # x shape: [..., n], need to compute for each dimension
        result = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for d in range(n):
            x_d = x[..., d]  # Shape: [...]
            # sum_k(a^k * cos(2*pi*b^k*(x_d + 0.5)))
            # Broadcast: [..., 1] * [k_max+1] -> [..., k_max+1]
            inner = 2 * torch.pi * b_k * (x_d.unsqueeze(-1) + 0.5)
            result = result + (a_k * torch.cos(inner)).sum(dim=-1)
        
        return result - n * const


class Alpine(BenchmarkFunction):
    """
    Alpine function (No. 1).
    
    f(x) = sum(|x_i * sin(x_i) + 0.1 * x_i|)
    
    Properties:
        - Multimodal
        - Separable
        - Global minimum: f(0,...,0) = 0
    """
    name = "alpine"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-10.0, 10.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        return torch.abs(x * torch.sin(x) + 0.1 * x).sum(dim=-1)


class Salomon(BenchmarkFunction):
    """
    Salomon function.
    
    f(x) = 1 - cos(2*pi*||x||) + 0.1*||x||
    
    Properties:
        - Multimodal
        - Non-separable (radially symmetric)
        - Global minimum: f(0,...,0) = 0
    """
    name = "salomon"
    optimal_value = 0.0
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-100.0, 100.0)
    
    def __call__(self, x: Tensor) -> Tensor:
        norm = torch.sqrt((x ** 2).sum(dim=-1))
        return 1 - torch.cos(2 * torch.pi * norm) + 0.1 * norm


class StyblinskiTang(BenchmarkFunction):
    """
    Styblinski-Tang function.
    
    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
    
    Properties:
        - Multimodal
        - Separable
        - Global minimum: f(-2.9035,...,-2.9035) ≈ -39.16599*n
    """
    name = "styblinski_tang"
    
    def __init__(self, n_var: int = 30, **kwargs):
        super().__init__(n_var=n_var, **kwargs)
        self.optimal_value = -39.16599 * n_var
    
    def default_bounds(self) -> Tuple[float, float]:
        return (-5.0, 5.0)
    
    def _compute_optimal_x(self) -> Tensor:
        return torch.full((self.n_var,), -2.903534)
    
    def __call__(self, x: Tensor) -> Tensor:
        return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum(dim=-1)


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

CLASSICAL_FUNCTIONS = {
    # Unimodal
    "sphere": Sphere,
    "ellipsoid": Ellipsoid,
    "sum_of_different_powers": SumOfDifferentPowers,
    "schwefel222": Schwefel222,
    "cigar": Cigar,
    "discus": Discus,
    "bent_cigar": BentCigar,
    "rosenbrock": Rosenbrock,
    "dixon_price": DixonPrice,
    "powell": Powell,
    "trid": Trid,
    # Multimodal
    "rastrigin": Rastrigin,
    "ackley": Ackley,
    "griewank": Griewank,
    "schwefel": Schwefel,
    "levy": Levy,
    "michalewicz": Michalewicz,
    "zakharov": Zakharov,
    "weierstrass": Weierstrass,
    "alpine": Alpine,
    "salomon": Salomon,
    "styblinski_tang": StyblinskiTang,
}
