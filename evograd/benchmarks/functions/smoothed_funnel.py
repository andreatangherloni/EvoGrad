"""
Smoothed Multi-Funnel Benchmark Functions

These benchmarks create landscapes with multiple attraction basins where:
- One basin is wide but suboptimal (gradient trap)
- One basin is narrow but optimal (requires exploration to find)
- The optimal basin contains an ill-conditioned valley (benefits from gradients)

This design specifically targets the scenario where:
- Pure gradient methods: Fall into the wide basin, converge to suboptimal
- Pure EAs: Can find the narrow basin, but waste evaluations on fine convergence  
- Differentiable EAs: Population finds basins + gradients accelerate valley convergence
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    from .base import BenchmarkFunction
except ImportError:
    from base import BenchmarkFunction


def log_sum_exp_min(f_values: Tensor, tau: float = 1.0) -> Tensor:
    """
    Smooth approximation to min using log-sum-exp.
    
    f(x) = -τ * log(Σ exp(-f_i/τ))
    
    As τ → 0, this approaches min(f_values).
    Larger τ gives smoother transitions between basins.
    
    Args:
        f_values: (..., K) tensor of K function values
        tau: Temperature parameter (smaller = sharper min)
        
    Returns:
        (...,) tensor of smoothed minimum values
    """
    # Use logsumexp for numerical stability
    return -tau * torch.logsumexp(-f_values / tau, dim=-1)


def random_orthogonal_matrix(n: int, seed: int = 0) -> Tensor:
    """Generate a random orthogonal matrix via QR decomposition."""
    g = torch.Generator()
    g.manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q


class SmoothedMultiFunnel(BenchmarkFunction):
    """
    Smoothed multi-funnel benchmark with asymmetric basins.
    
    Creates two funnels:
    - Funnel A (distractor): Wide, easy to find, suboptimal
    - Funnel B (global): Narrow, hard to find, optimal with ill-conditioned valley
    
    The funnels are combined using log-sum-exp smoothing to maintain
    differentiability while preserving distinct attraction basins.
    
    Parameters
    ----------
    n_var : int
        Dimensionality of the problem.
    tau : float
        Temperature for log-sum-exp smoothing. Smaller values create sharper
        basin boundaries (harder). Default 1.0.
    delta : float
        Offset added to the distractor basin. Controls how "deceptive" it is.
        Larger values make the distractor less attractive. Default 10.0.
    distractor_center : Optional[Tensor]
        Center of the distractor basin. If None, placed at (-2, -2, ..., -2).
    rotate : bool
        Whether to apply random rotation for non-separability. Default True.
    condition : float
        Condition number for the distractor ellipsoid. Default 1.0 (sphere).
    seed : int
        Random seed for rotation matrix generation. Default 0.
    xl : float
        Lower bound. Default -5.0.
    xu : float
        Upper bound. Default 5.0.
    """
    
    @staticmethod
    def default_bounds() -> Tuple[float, float]:
        """Default bounds for SmoothedMultiFunnel."""
        return (-5.0, 5.0)
    
    def __init__(
        self,
        n_var: int = 10,
        tau: float = 1.0,
        delta: float = 10.0,
        distractor_center: Optional[Tensor] = None,
        rotate: bool = True,
        condition: float = 1.0,
        seed: int = 0,
        xl: float = -5.0,
        xu: float = 5.0,
    ):
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        
        if n_var < 2:
            raise ValueError("SmoothedMultiFunnel requires n_var >= 2")
        
        self.name = "SmoothedMultiFunnel"
        self.tau = tau
        self.delta = delta
        self.condition = condition
        self.seed = seed
        
        # Distractor center (default: away from Rosenbrock optimum at (1,1,...,1))
        if distractor_center is None:
            self._distractor_center = torch.full((n_var,), -2.0)
        else:
            self._distractor_center = distractor_center.clone()
        
        # Rotation matrix for non-separability
        if rotate:
            self._Q = random_orthogonal_matrix(n_var, seed)
        else:
            self._Q = torch.eye(n_var)
        
        # Condition scaling for distractor ellipsoid
        # Creates eigenvalues from 1 to condition
        self._scales = torch.logspace(0, torch.log10(torch.tensor(condition)), n_var)
        
        # Global optimum is at (1, 1, ..., 1) in rotated coordinates
        # In original coordinates: x* = Q^T @ [1, 1, ..., 1]
        ones = torch.ones(n_var)
        self._optimal_x = self._Q.T @ ones
        self._optimal_value = 0.0  # Rosenbrock minimum
    
    @property
    def optimal_x(self) -> Tensor:
        """Global optimum location."""
        return self._optimal_x
    
    @property
    def optimal_value(self) -> float:
        """Global optimum value."""
        return self._optimal_value
    
    def _rosenbrock(self, x: Tensor) -> Tensor:
        """Standard Rosenbrock function."""
        x_i = x[..., :-1]
        x_ip1 = x[..., 1:]
        return (100.0 * (x_ip1 - x_i ** 2) ** 2 + (1.0 - x_i) ** 2).sum(dim=-1)
    
    def _funnel_global(self, x: Tensor) -> Tensor:
        """
        Global optimum funnel: Rotated Rosenbrock.
        
        Narrow attraction basin with ill-conditioned curved valley.
        Optimum at Q^T @ [1, 1, ..., 1] with value 0.
        """
        # Rotate to make non-separable
        Q = self._Q.to(x.device)
        y = x @ Q  # Rotated coordinates
        return self._rosenbrock(y)
    
    def _funnel_distractor(self, x: Tensor) -> Tensor:
        """
        Distractor funnel: Ellipsoid centered away from global optimum.
        
        Wide attraction basin, easy to find via gradients.
        Suboptimal by delta.
        """
        center = self._distractor_center.to(x.device)
        scales = self._scales.to(x.device)
        
        diff = x - center
        # Weighted squared norm (ellipsoid)
        return (scales * diff ** 2).sum(dim=-1) + self.delta
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the smoothed multi-funnel function.
        
        Args:
            x: (..., n_var) input tensor
            
        Returns:
            (...,) function values
        """
        f_global = self._funnel_global(x)
        f_distractor = self._funnel_distractor(x)
        
        # Stack and compute smooth min
        f_all = torch.stack([f_global, f_distractor], dim=-1)
        return log_sum_exp_min(f_all, self.tau)


class MultiBasinRosenbrock(BenchmarkFunction):
    """
    Multiple Rosenbrock basins with smooth transitions.
    
    Creates K funnels, each a shifted/rotated Rosenbrock with different
    biases. Uses log-sum-exp for smooth differentiable combination.
    
    Unlike the hard-min version, this maintains gradients across basin
    boundaries, enabling gradient-based methods to potentially escape
    suboptimal basins.
    
    Parameters
    ----------
    n_var : int
        Dimensionality.
    n_funnels : int
        Number of funnels. Default 4.
    tau : float
        Temperature for smoothing. Default 1.0.
    shift_scale : float
        Scale of random shifts. Default 2.0.
    bias_scale : float
        Scale of random biases (funnel depth differences). Default 50.0.
    rotate_funnels : bool
        Whether each funnel gets a random rotation. Default True.
    seed : int
        Random seed. Default 0.
    xl : float
        Lower bound. Default -5.0.
    xu : float
        Upper bound. Default 5.0.
    """
    
    @staticmethod
    def default_bounds() -> Tuple[float, float]:
        """Default bounds for MultiBasinRosenbrock."""
        return (-5.0, 5.0)
    
    def __init__(
        self,
        n_var: int = 10,
        n_funnels: int = 4,
        tau: float = 1.0,
        shift_scale: float = 2.0,
        bias_scale: float = 50.0,
        rotate_funnels: bool = True,
        seed: int = 0,
        xl: float = -5.0,
        xu: float = 5.0,
    ):
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        
        if n_var < 2:
            raise ValueError("MultiBasinRosenbrock requires n_var >= 2")
        if n_funnels < 1:
            raise ValueError("n_funnels must be >= 1")
        
        self.name = "MultiBasinRosenbrock"
        self.n_funnels = n_funnels
        self.tau = tau
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Generate shifts (funnel centers offset from optimum)
        # Optimum of shifted Rosenbrock(y) is at y = 1, so x* = center + 1
        centers = torch.randn(n_funnels, n_var, generator=g) * shift_scale
        centers = centers.clamp(xl - 1.0, xu - 1.0)  # Keep optima in bounds
        self._centers = centers
        
        # Generate biases (funnel 0 is global best)
        if n_funnels == 1:
            biases = torch.zeros(1)
        else:
            biases = torch.zeros(n_funnels)
            biases[1:] = torch.rand(n_funnels - 1, generator=g) * bias_scale
        self._biases = biases
        
        # Generate per-funnel rotations
        if rotate_funnels:
            self._rotations = torch.stack([
                random_orthogonal_matrix(n_var, seed + k)
                for k in range(n_funnels)
            ])
        else:
            self._rotations = torch.eye(n_var).unsqueeze(0).expand(n_funnels, -1, -1)
        
        # Global optimum
        self._optimal_x = (self._rotations[0].T @ (self._centers[0] + 1.0))
        self._optimal_value = float(self._biases[0].item())
    
    @property
    def optimal_x(self) -> Tensor:
        """Global optimum location."""
        return self._optimal_x
    
    @property
    def optimal_value(self) -> float:
        """Global optimum value."""
        return self._optimal_value
    
    def _rosenbrock(self, x: Tensor) -> Tensor:
        """Standard Rosenbrock function."""
        x_i = x[..., :-1]
        x_ip1 = x[..., 1:]
        return (100.0 * (x_ip1 - x_i ** 2) ** 2 + (1.0 - x_i) ** 2).sum(dim=-1)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the multi-basin function.
        
        Args:
            x: (..., n_var) input tensor
            
        Returns:
            (...,) function values
        """
        centers = self._centers.to(x.device)
        biases = self._biases.to(x.device)
        rotations = self._rotations.to(x.device)
        
        # Compute all funnel values
        # x: (..., D), centers: (K, D) -> shifted: (..., K, D)
        shifted = x.unsqueeze(-2) - centers
        
        # Apply per-funnel rotations: (..., K, D) @ (K, D, D) -> (..., K, D)
        # Use einsum for batched matrix multiply
        rotated = torch.einsum('...kd,kde->...ke', shifted, rotations)
        
        # Rosenbrock on each funnel
        y_i = rotated[..., :-1]
        y_ip1 = rotated[..., 1:]
        rosen = (100.0 * (y_ip1 - y_i ** 2) ** 2 + (1.0 - y_i) ** 2).sum(dim=-1)
        
        # Add biases: (..., K)
        f_all = rosen + biases
        
        # Smooth min over funnels
        return log_sum_exp_min(f_all, self.tau)


class DeceptiveLandscape(BenchmarkFunction):
    """
    Highly deceptive landscape with controllable difficulty.
    
    Combines multiple elements designed to challenge different optimization
    approaches:
    
    1. A wide, smooth distractor basin (traps gradient methods)
    2. A narrow global basin with ill-conditioning (needs exploration + exploitation)
    3. Optional saddle points and ridges
    4. Non-separable structure via rotation
    
    Parameters
    ----------
    n_var : int
        Dimensionality.
    tau : float
        Smoothing temperature. Default 0.5 (fairly sharp).
    n_distractors : int
        Number of distractor basins. Default 2.
    distractor_depth : float
        How close distractors are to global optimum. Default 5.0.
    global_conditioning : float
        Condition number of global basin valley. Default 100.0.
    rotate : bool
        Apply random rotation. Default True.
    seed : int
        Random seed. Default 0.
    xl : float
        Lower bound. Default -5.0.
    xu : float
        Upper bound. Default 5.0.
    """
    
    @staticmethod
    def default_bounds() -> Tuple[float, float]:
        """Default bounds for DeceptiveLandscape."""
        return (-5.0, 5.0)
    
    def __init__(
        self,
        n_var: int = 10,
        tau: float = 0.5,
        n_distractors: int = 2,
        distractor_depth: float = 5.0,
        global_conditioning: float = 100.0,
        rotate: bool = True,
        seed: int = 0,
        xl: float = -5.0,
        xu: float = 5.0,
    ):
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        
        if n_var < 2:
            raise ValueError("DeceptiveLandscape requires n_var >= 2")
        
        self.name = "DeceptiveLandscape"
        self.tau = tau
        self.n_distractors = n_distractors
        self.distractor_depth = distractor_depth
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Rotation for non-separability
        if rotate:
            self._Q = random_orthogonal_matrix(n_var, seed)
        else:
            self._Q = torch.eye(n_var)
        
        # Generate distractor centers (spread around the space)
        self._distractor_centers = torch.randn(n_distractors, n_var, generator=g) * 2.0
        
        # Distractor widths (wider = easier to fall into)
        self._distractor_widths = torch.rand(n_distractors, generator=g) * 0.5 + 0.5
        
        # Condition scaling for global basin (ill-conditioned valley)
        self._global_scales = torch.logspace(
            0, torch.log10(torch.tensor(global_conditioning)), n_var
        )
        
        # Global optimum at origin in rotated coordinates
        self._optimal_x = torch.zeros(n_var)
        self._optimal_value = 0.0
    
    @property
    def optimal_x(self) -> Tensor:
        """Global optimum location."""
        return self._optimal_x
    
    @property
    def optimal_value(self) -> float:
        """Global optimum value."""
        return self._optimal_value
    
    def _global_basin(self, x: Tensor) -> Tensor:
        """
        Global optimum: Ill-conditioned ellipsoid in rotated coordinates.
        
        This creates a narrow valley that benefits from gradient-based
        fine-tuning once the basin is found.
        """
        Q = self._Q.to(x.device)
        scales = self._global_scales.to(x.device)
        
        y = x @ Q  # Rotated
        return (scales * y ** 2).sum(dim=-1)
    
    def _distractor_basins(self, x: Tensor) -> Tensor:
        """
        Distractor basins: Wide spheres offset from origin.
        
        Returns tensor of shape (..., n_distractors) with value for each distractor.
        """
        centers = self._distractor_centers.to(x.device)
        widths = self._distractor_widths.to(x.device)
        
        # (..., D) - (K, D) -> (..., K, D)
        diff = x.unsqueeze(-2) - centers
        
        # Scaled squared distance + offset
        dist_sq = (diff ** 2).sum(dim=-1)  # (..., K)
        return dist_sq * widths + self.distractor_depth
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the deceptive landscape.
        
        Args:
            x: (..., n_var) input tensor
            
        Returns:
            (...,) function values
        """
        f_global = self._global_basin(x)  # (...,)
        f_distractors = self._distractor_basins(x)  # (..., K)
        
        # Combine all basins
        f_all = torch.cat([f_global.unsqueeze(-1), f_distractors], dim=-1)
        
        return log_sum_exp_min(f_all, self.tau)


# Registry
SMOOTHED_FUNNEL_FUNCTIONS = {
    "smoothedmultifunnel": SmoothedMultiFunnel,
    "multibasinrosenbrock": MultiBasinRosenbrock,
    "deceptivelandscape": DeceptiveLandscape,
}
