"""
Sampling operators for population initialisation.

This module provides strategies for initialising populations
in the search space. Samplers are used by algorithms to create
the initial population.

Available samplers:
    - UniformSampling: Uniform random sampling (default)
    - LatinHypercubeSampling: Better space coverage via LHS
    - NormalSampling: Gaussian sampling around center
    - LogUniformSampling: Log-scale uniform sampling

Example:
    >>> from evograd.operators import UniformSampling
    >>> from evograd.core import Problem
    >>> 
    >>> sampler = UniformSampling()
    >>> problem = Problem(n_var=10, xl=-5.0, xu=5.0)
    >>> 
    >>> # Sample 100 individuals
    >>> population = sampler(100, problem)
    >>> print(population.shape)  # torch.Size([100, 10])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = [
    "Sampling",
    "UniformSampling",
    "LatinHypercubeSampling",
    "NormalSampling",
    "LogUniformSampling",
]


# =============================================================================
# Base Sampling Class
# =============================================================================

class Sampling(nn.Module, ABC):
    """
    Abstract base class for population sampling strategies.
    
    Subclasses must implement:
        - _sample(): Generate samples within [0, 1]^d
    
    The base class handles:
        - Scaling samples to problem bounds
        - Device/dtype management
        - Seeding for reproducibility
    
    Args:
        seed: Random seed for reproducibility.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self.seed = seed
        self._generator: Optional[torch.Generator] = None
    
    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Get or create random generator for reproducibility."""
        if self._generator is None and self.seed is not None:
            # Handle MPS device - use CPU generator since MPS has limited generator support
            # Also handle the 'mps' vs 'mps:0' issue
            if device.type == 'mps':
                # MPS generators have issues - use CPU generator instead
                # The results will still be on MPS, just seeded from CPU
                gen_device = torch.device('cpu')
            else:
                gen_device = device
            
            self._generator = torch.Generator(device=gen_device)
            self._generator.manual_seed(self.seed)
        return self._generator
    
    @abstractmethod
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Generate samples in [0, 1]^d.
        
        Args:
            n_samples: Number of samples to generate.
            n_var: Number of variables (dimensions).
            device: Target device.
            dtype: Target dtype.
        
        Returns:
            Tensor of shape [n_samples, n_var] with values in [0, 1].
        """
        pass
    
    def forward(
        self,
        n_samples: int,
        problem: Problem,
    ) -> Tensor:
        """
        Sample population for a problem.
        
        Args:
            n_samples: Number of individuals to sample.
            problem: Problem instance with bounds.
        
        Returns:
            Population tensor of shape [n_samples, n_var].
        """
        # Get bounds from problem
        xl = problem.xl  # [n_var] or scalar
        xu = problem.xu  # [n_var] or scalar
        device = xl.device
        dtype = xl.dtype
        n_var = problem.n_var
        
        # Sample in [0, 1]^d
        samples = self._sample(n_samples, n_var, device, dtype)
        
        # Scale to problem bounds: x = xl + samples * (xu - xl)
        population = xl + samples * (xu - xl)
        
        return population
    
    def __call__(
        self,
        n_samples: int,
        problem: Problem,
    ) -> Tensor:
        """Sample population (alias for forward)."""
        return self.forward(n_samples, problem)
    
    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"{self.__class__.__name__}({seed_str})"


# =============================================================================
# Uniform Random Sampling
# =============================================================================

class UniformSampling(Sampling):
    """
    Uniform random sampling in the search space.
    
    The simplest and most common initialisation strategy.
    Samples are drawn uniformly from [xl, xu] for each variable.
    
    Args:
        seed: Random seed for reproducibility.
    
    Example:
        >>> sampler = UniformSampling()
        >>> population = sampler(100, problem)
    """
    
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        generator = self._get_generator(device)
        
        if generator is not None:
            # Generator may be on CPU for MPS devices
            gen_device = generator.device
            samples = torch.rand(
                n_samples, n_var,
                device=gen_device,
                dtype=dtype,
                generator=generator,
            )
            # Move to target device if needed
            if gen_device != device:
                samples = samples.to(device)
            return samples
        else:
            return torch.rand(n_samples, n_var, device=device, dtype=dtype)


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

class LatinHypercubeSampling(Sampling):
    """
    Latin Hypercube Sampling (LHS) for better space coverage.
    
    LHS ensures that samples are well-distributed across each
    dimension by dividing each dimension into n equal intervals
    and sampling exactly once from each interval.
    
    This provides better coverage of the search space compared
    to pure random sampling, especially for small sample sizes.
    
    Args:
        smooth: If True, add jitter within each stratum (default).
            If False, sample at stratum centers.
        seed: Random seed for reproducibility.
    
    Example:
        >>> sampler = LatinHypercubeSampling()
        >>> population = sampler(100, problem)
    
    Note:
        LHS is particularly useful for:
        - Small population sizes
        - High-dimensional problems
        - When initial coverage is important
    """
    
    def __init__(
        self,
        smooth: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.smooth = smooth
    
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        generator = self._get_generator(device)
        
        # Determine generator device (may be CPU for MPS)
        gen_device = generator.device if generator is not None else device
        
        # Create intervals: [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1)
        # Sample one point in each interval for each dimension
        
        # Base positions (left edge of each stratum)
        indices = torch.arange(n_samples, device=device, dtype=dtype)
        
        # Randomly permute indices for each dimension
        samples = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
        
        for j in range(n_var):
            # Random permutation (generate on generator device, then move if needed)
            if generator is not None:
                perm = torch.randperm(n_samples, device=gen_device, generator=generator)
                if gen_device != device:
                    perm = perm.to(device)
            else:
                perm = torch.randperm(n_samples, device=device)
            
            if self.smooth:
                # Add random jitter within each stratum
                if generator is not None:
                    jitter = torch.rand(n_samples, device=gen_device, dtype=dtype, generator=generator)
                    if gen_device != device:
                        jitter = jitter.to(device)
                else:
                    jitter = torch.rand(n_samples, device=device, dtype=dtype)
                samples[:, j] = (perm.to(dtype) + jitter) / n_samples
            else:
                # Sample at stratum centers
                samples[:, j] = (perm.to(dtype) + 0.5) / n_samples
        
        return samples
    
    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"LatinHypercubeSampling(smooth={self.smooth}{seed_str})"


# =============================================================================
# Normal (Gaussian) Sampling
# =============================================================================

class NormalSampling(Sampling):
    """
    Gaussian sampling around the center of the search space.
    
    Samples are drawn from a normal distribution centered at
    the midpoint of the bounds, with standard deviation scaled
    to fit within the bounds.
    
    This is useful when:
        - Good solutions are expected near the center
        - A focused initial search is desired
    
    Args:
        sigma_factor: Standard deviation as fraction of range.
            Default 1/3 means 99.7% of samples within bounds
            (before clipping).
        clip_to_bounds: If True, clip samples to [xl, xu].
        seed: Random seed for reproducibility.
    
    Example:
        >>> sampler = NormalSampling(sigma_factor=0.25)
        >>> population = sampler(100, problem)
    """
    
    def __init__(
        self,
        sigma_factor: float = 1.0 / 3.0,
        clip_to_bounds: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.sigma_factor = sigma_factor
        self.clip_to_bounds = clip_to_bounds
    
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        generator = self._get_generator(device)
        
        # Sample from standard normal
        if generator is not None:
            # Generator may be on CPU for MPS devices
            gen_device = generator.device
            z = torch.randn(
                n_samples, n_var,
                device=gen_device,
                dtype=dtype,
                generator=generator,
            )
            if gen_device != device:
                z = z.to(device)
        else:
            z = torch.randn(n_samples, n_var, device=device, dtype=dtype)
        
        # Transform to [0, 1] centered at 0.5 with scaled std
        # mean=0.5, std=sigma_factor * 0.5
        samples = 0.5 + self.sigma_factor * 0.5 * z
        
        if self.clip_to_bounds:
            samples = torch.clamp(samples, 0.0, 1.0)
        
        return samples
    
    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"NormalSampling(sigma_factor={self.sigma_factor}{seed_str})"


# =============================================================================
# Log-Uniform Sampling
# =============================================================================

class LogUniformSampling(Sampling):
    """
    Log-uniform sampling for problems with log-scale parameters.
    
    Samples are uniformly distributed in log-space, useful for
    parameters that span multiple orders of magnitude (e.g.,
    learning rates, regularisation coefficients).
    
    Note: Bounds must be strictly positive!
    
    Args:
        base: Deprecated, kept for backwards compatibility. 
            The implementation now uses natural log internally.
        seed: Random seed for reproducibility.
    
    Example:
        >>> # Sample learning rates from [1e-5, 1e-1]
        >>> problem = Problem(n_var=1, xl=1e-5, xu=1e-1)
        >>> sampler = LogUniformSampling()
        >>> population = sampler(100, problem)
    
    Warning:
        Both xl and xu must be > 0 for all variables.
    """
    
    def __init__(
        self,
        base: float = 10.0,  # Kept for backwards compatibility, not used
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.base = base  # Kept for repr, not used in computation
    
    def forward(
        self,
        n_samples: int,
        problem: Problem,
    ) -> Tensor:
        """
        Sample in log-space and transform back.
        
        Overrides base forward to handle log transformation.
        """
        xl = problem.xl
        xu = problem.xu
        device = xl.device
        dtype = xl.dtype
        n_var = problem.n_var
        
        # Validate positive bounds
        if (xl <= 0).any() or (xu <= 0).any():
            raise ValueError(
                "LogUniformSampling requires strictly positive bounds. "
                f"Got xl={xl}, xu={xu}"
            )
        
        # Work in natural log space (avoids torch.pow(scalar, tensor) which isn't supported on MPS)
        # Mathematically equivalent: uniform in log-space => log-uniform in original space
        log_xl = torch.log(xl)
        log_xu = torch.log(xu)
        
        # Sample uniformly in [0, 1]
        samples = self._sample(n_samples, n_var, device, dtype)
        
        # Scale to log-space bounds (natural log)
        log_samples = log_xl + samples * (log_xu - log_xl)
        
        # Transform back to original space using exp (works on all devices)
        population = torch.exp(log_samples)
        
        return population
    
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        generator = self._get_generator(device)
        
        if generator is not None:
            # Generator may be on CPU for MPS devices
            gen_device = generator.device
            samples = torch.rand(
                n_samples, n_var,
                device=gen_device,
                dtype=dtype,
                generator=generator,
            )
            if gen_device != device:
                samples = samples.to(device)
            return samples
        else:
            return torch.rand(n_samples, n_var, device=device, dtype=dtype)
    
    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"LogUniformSampling(base={self.base}{seed_str})"


# =============================================================================
# Halton Sequence Sampling (Quasi-Random)
# =============================================================================

class HaltonSampling(Sampling):
    """
    Halton sequence quasi-random sampling.
    
    Generates low-discrepancy sequences that fill the space
    more uniformly than random sampling. Each dimension uses
    a different prime base.
    
    Particularly useful for:
        - Integration/Monte Carlo methods
        - Surrogate model initialisation
        - When uniform coverage is critical
    
    Args:
        scramble: If True, apply random scrambling to reduce
            correlation in high dimensions.
        seed: Random seed for scrambling.
    
    Example:
        >>> sampler = HaltonSampling()
        >>> population = sampler(100, problem)
    """
    
    # First 100 primes for high-dimensional problems
    _PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
        199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
        383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    ]
    
    def __init__(
        self,
        scramble: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.scramble = scramble
    
    def _halton_sequence(
        self,
        n_samples: int,
        base: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate Halton sequence for a single dimension."""
        result = torch.zeros(n_samples, device=device, dtype=dtype)
        
        for i in range(n_samples):
            f = 1.0
            r = 0.0
            idx = i + 1  # Start from 1
            
            while idx > 0:
                f = f / base
                r = r + f * (idx % base)
                idx = idx // base
            
            result[i] = r
        
        return result
    
    def _sample(
        self,
        n_samples: int,
        n_var: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if n_var > len(self._PRIMES):
            raise ValueError(
                f"HaltonSampling supports up to {len(self._PRIMES)} dimensions, "
                f"got n_var={n_var}"
            )
        
        samples = torch.zeros(n_samples, n_var, device=device, dtype=dtype)
        
        for j in range(n_var):
            base = self._PRIMES[j]
            samples[:, j] = self._halton_sequence(n_samples, base, device, dtype)
        
        # Optional scrambling
        if self.scramble:
            generator = self._get_generator(device)
            
            for j in range(n_var):
                if generator is not None:
                    # Generator may be on CPU for MPS devices
                    gen_device = generator.device
                    shift = torch.rand(1, device=gen_device, dtype=dtype, generator=generator)
                    if gen_device != device:
                        shift = shift.to(device)
                else:
                    shift = torch.rand(1, device=device, dtype=dtype)
                
                samples[:, j] = (samples[:, j] + shift) % 1.0
        
        return samples
    
    def __repr__(self) -> str:
        seed_str = f", seed={self.seed}" if self.seed is not None else ""
        return f"HaltonSampling(scramble={self.scramble}{seed_str})"
