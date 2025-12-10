"""
Crossover operators for recombination.

This module provides crossover (recombination) operators that
combine genetic information from parent individuals to create
offspring. All operators support both classical and differentiable
modes.

Available crossovers:
    - SBXCrossover: Simulated Binary Crossover (GA)
    - BlendCrossover: BLX-alpha crossover (GA)
    - BinomialCrossover: DE-style binomial crossover
    - ExponentialCrossover: DE-style exponential crossover
    - UniformCrossover: Simple uniform crossover
    - ArithmeticCrossover: Weighted average of parents

Differentiable Mode:
    When `differentiable=True`, crossover masks use Binary-Concrete
    (Gumbel-Sigmoid) relaxation with straight-through estimator,
    allowing gradients to flow through crossover decisions.

Example:
    >>> from evograd.operators import SBXCrossover
    >>> 
    >>> # Classical mode
    >>> crossover = SBXCrossover(eta=15, prob=0.9)
    >>> offspring = crossover(parent1, parent2)
    >>> 
    >>> # Differentiable mode
    >>> crossover = SBXCrossover(
    ...     eta=15,
    ...     prob=0.9,
    ...     differentiable=True,
    ...     learn_eta=True,
    ... )
    >>> offspring = crossover(parent1, parent2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Crossover",
    "SBXCrossover",
    "BlendCrossover",
    "BinomialCrossover",
    "ExponentialCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
]


# =============================================================================
# Base Crossover Class
# =============================================================================

class Crossover(nn.Module, ABC):
    """
    Abstract base class for crossover operators.
    
    Subclasses must implement:
        - _crossover(): Perform crossover between parents
    
    Args:
        prob: Crossover probability (per individual or per gene).
        differentiable: If True, use Binary-Concrete for soft masks.
        temperature: Temperature for Binary-Concrete.
        learn_temperature: If True, temperature is learnable.
        learn_prob: If True, crossover probability is learnable.
        n_var: Number of variables (for per-gene probability).
    """
    
    def __init__(
        self,
        prob: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.differentiable = differentiable
        self.n_var = n_var
        
        # Temperature parameter (log for positivity)
        if learn_temperature and differentiable:
            self._log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "_log_temperature",
                torch.tensor(temperature).log()
            )
        
        # Crossover probability as logits
        prob_logit = self._prob_to_logit(prob)
        if learn_prob and differentiable:
            if n_var is not None:
                # Per-gene probability
                self.prob_logits = nn.Parameter(
                    torch.full((n_var,), prob_logit)
                )
            else:
                # Scalar probability (will be expanded later)
                self.prob_logits = nn.Parameter(torch.tensor(prob_logit))
        else:
            if n_var is not None:
                self.register_buffer(
                    "prob_logits",
                    torch.full((n_var,), prob_logit)
                )
            else:
                self.register_buffer(
                    "prob_logits",
                    torch.tensor(prob_logit)
                )
    
    @staticmethod
    def _prob_to_logit(p: float, eps: float = 1e-7) -> float:
        """Convert probability to logit."""
        p = max(min(p, 1 - eps), eps)
        return torch.logit(torch.tensor(p)).item()
    
    @property
    def temperature(self) -> Tensor:
        """Current temperature value."""
        return self._log_temperature.exp()
    
    @property
    def prob(self) -> Tensor:
        """Current crossover probability."""
        return torch.sigmoid(self.prob_logits)
    
    def _get_prob_logits(self, n_var: int, device: torch.device) -> Tensor:
        """Get probability logits, expanding if necessary."""
        logits = self.prob_logits.to(device)
        if logits.dim() == 0:
            # Scalar -> expand to n_var
            return logits.expand(n_var)
        return logits
    
    def _binary_concrete(
        self,
        logits: Tensor,
        hard: bool = True,
        eps: float = 1e-10,
    ) -> Tensor:
        """
        Binary-Concrete (Gumbel-Sigmoid) with straight-through.
        
        Args:
            logits: Unnormalised log-odds.
            hard: If True, use straight-through estimator.
            eps: Small constant for numerical stability.
        
        Returns:
            Soft or hard binary mask.
        """
        # Sample uniform noise
        u = torch.rand_like(logits)
        u = torch.clamp(u, eps, 1.0 - eps)
        
        # Gumbel-Sigmoid
        noise = torch.log(u) - torch.log(1 - u)
        y_soft = torch.sigmoid((logits + noise) / self.temperature)
        
        if hard:
            # Straight-through estimator
            y_hard = (y_soft > 0.5).float()
            return (y_hard - y_soft).detach() + y_soft
        
        return y_soft
    
    @abstractmethod
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        """
        Perform crossover between parent pairs.
        
        Args:
            parent1: First parents [n_pairs, n_var].
            parent2: Second parents [n_pairs, n_var].
        
        Returns:
            Offspring [n_pairs, n_var].
        """
        pass
    
    def forward(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        """
        Apply crossover to parent pairs.
        
        Args:
            parent1: First parents [n_pairs, n_var].
            parent2: Second parents [n_pairs, n_var].
        
        Returns:
            Offspring [n_pairs, n_var].
        """
        return self._crossover(parent1, parent2)
    
    def __call__(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        """Apply crossover (alias for forward)."""
        return self.forward(parent1, parent2)


# =============================================================================
# Simulated Binary Crossover (SBX)
# =============================================================================

class SBXCrossover(Crossover):
    """
    Simulated Binary Crossover (SBX).
    
    SBX simulates single-point crossover for real-valued variables.
    It creates offspring that are distributed around the parents
    with spread controlled by the distribution index eta.
    
    Higher eta values produce offspring closer to parents (more
    exploitation), while lower values produce more spread (more
    exploration).
    
    Args:
        eta: Distribution index (higher = tighter spread).
        prob: Crossover probability per gene.
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_eta: If True, eta is learnable.
        learn_prob: If True, crossover probability is learnable.
        n_var: Number of variables (for per-gene probability).
    
    Example:
        >>> crossover = SBXCrossover(eta=15, prob=0.9)
        >>> offspring = crossover(parent1, parent2)
    
    Reference:
        Deb & Agrawal (1995). Simulated Binary Crossover for
        Continuous Search Space.
    """
    
    def __init__(
        self,
        eta: float = 15.0,
        prob: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_eta: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        # Eta parameter (log for positivity)
        if learn_eta and differentiable:
            self._log_eta = nn.Parameter(torch.tensor(eta).log())
        else:
            self.register_buffer("_log_eta", torch.tensor(eta).log())
    
    @property
    def eta(self) -> Tensor:
        """Current eta value."""
        return self._log_eta.exp()
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Get crossover mask (which genes to cross)
        prob_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            # Binary-Concrete mask per gene
            logits = prob_logits.unsqueeze(0).expand(n_pairs, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            # Hard Bernoulli mask
            mask = (torch.rand(n_pairs, n_var, device=device) < self.prob).float()
        
        # Compute SBX spread factor beta
        u = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        
        eta = self.eta
        beta = torch.where(
            u <= 0.5,
            (2 * u).pow(1.0 / (eta + 1)),
            (2 * (1 - u)).pow(-1.0 / (eta + 1))
        )
        
        # Apply mask: beta=1 means no crossover (offspring = parent)
        beta = mask * beta + (1.0 - mask) * 1.0
        
        # Generate offspring
        offspring = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        
        return offspring
    
    def __repr__(self) -> str:
        return (
            f"SBXCrossover("
            f"eta={self.eta.item():.2f}, "
            f"prob={self.prob.mean().item():.3f}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Blend Crossover (BLX-alpha)
# =============================================================================

class BlendCrossover(Crossover):
    """
    Blend Crossover (BLX-alpha).
    
    Creates offspring by sampling uniformly from an extended
    interval around the parents. The interval is extended by
    alpha * (parent_max - parent_min) on each side.
    
    With alpha=0, offspring are sampled between parents.
    With alpha=0.5 (default), the interval is extended by 50%.
    
    Args:
        alpha: Extension factor for the interval.
        prob: Crossover probability (per individual).
        differentiable: If True, use soft interpolation.
        learn_alpha: If True, alpha is learnable.
    
    Example:
        >>> crossover = BlendCrossover(alpha=0.5)
        >>> offspring = crossover(parent1, parent2)
    
    Reference:
        Eshelman & Schaffer (1993). Real-Coded Genetic Algorithms
        and Interval-Schemata.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        prob: float = 0.9,
        differentiable: bool = False,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        # Alpha parameter
        if learn_alpha and differentiable:
            # Use sigmoid to keep alpha in reasonable range
            self._alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(alpha / 2.0))  # Map [0,2] -> sigmoid
            )
        else:
            self.register_buffer(
                "_alpha_logit",
                torch.logit(torch.tensor(alpha / 2.0))
            )
    
    @property
    def alpha(self) -> Tensor:
        """Current alpha value."""
        return 2.0 * torch.sigmoid(self._alpha_logit)
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Determine interval bounds
        p_min = torch.minimum(parent1, parent2)
        p_max = torch.maximum(parent1, parent2)
        diff = p_max - p_min
        
        alpha = self.alpha
        
        # Extended interval
        lower = p_min - alpha * diff
        upper = p_max + alpha * diff
        
        # Sample uniformly from interval
        u = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        offspring = lower + u * (upper - lower)
        
        # Apply crossover probability (per individual)
        if not self.differentiable:
            do_cross = (torch.rand(n_pairs, 1, device=device) < self.prob).float()
            offspring = do_cross * offspring + (1 - do_cross) * parent1
        
        return offspring
    
    def __repr__(self) -> str:
        return (
            f"BlendCrossover("
            f"alpha={self.alpha.item():.3f}, "
            f"prob={self.prob.item():.3f})"
        )


# =============================================================================
# Binomial Crossover (DE-style)
# =============================================================================

class BinomialCrossover(Crossover):
    """
    Binomial (uniform) crossover for Differential Evolution.
    
    Each gene is independently selected from either the target
    or donor vector based on the crossover rate. At least one
    gene is always taken from the donor (j_rand).
    
    Args:
        cr: Crossover rate (probability of taking donor gene).
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_cr: If True, crossover rate is learnable.
        n_var: Number of variables (for per-gene CR).
    
    Example:
        >>> # target = current individual, donor = mutant vector
        >>> crossover = BinomialCrossover(cr=0.9)
        >>> trial = crossover(target, donor)
    
    Note:
        In DE terminology:
        - parent1 = target vector (current individual)
        - parent2 = donor vector (mutant)
        - output = trial vector
    """
    
    def __init__(
        self,
        cr: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_cr: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=cr,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_cr,
            n_var=n_var,
        )
    
    @property
    def cr(self) -> Tensor:
        """Current crossover rate."""
        return self.prob
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        """
        Binomial crossover.
        
        Args:
            parent1: Target vectors [n_pairs, n_var].
            parent2: Donor vectors [n_pairs, n_var].
        
        Returns:
            Trial vectors [n_pairs, n_var].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        
        # Get CR logits
        cr_logits = self._get_prob_logits(n_var, device)
        
        if self.differentiable:
            # Binary-Concrete mask
            logits = cr_logits.unsqueeze(0).expand(n_pairs, -1)
            mask = self._binary_concrete(logits, hard=True)
        else:
            # Hard Bernoulli mask
            mask = (torch.rand(n_pairs, n_var, device=device) < self.cr).float()
        
        # Ensure at least one gene from donor (j_rand)
        j_rand = torch.randint(0, n_var, (n_pairs,), device=device)
        mask[torch.arange(n_pairs, device=device), j_rand] = 1.0
        
        # Trial vector: mask=1 -> donor, mask=0 -> target
        trial = mask * parent2 + (1.0 - mask) * parent1
        
        return trial
    
    def __repr__(self) -> str:
        return (
            f"BinomialCrossover("
            f"cr={self.cr.mean().item():.3f}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Exponential Crossover (DE-style)
# =============================================================================

class ExponentialCrossover(Crossover):
    """
    Exponential crossover for Differential Evolution.
    
    Copies a contiguous segment of genes from the donor vector,
    starting at a random position. The segment length follows
    a geometric distribution with parameter CR.
    
    Args:
        cr: Crossover rate (probability of extending segment).
        differentiable: If True, use soft approximation.
        temperature: Temperature for soft crossover.
        learn_cr: If True, crossover rate is learnable.
    
    Example:
        >>> crossover = ExponentialCrossover(cr=0.9)
        >>> trial = crossover(target, donor)
    
    Note:
        Exponential crossover tends to preserve more structure
        from the target vector compared to binomial crossover.
    """
    
    def __init__(
        self,
        cr: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
        learn_cr: bool = True,
    ) -> None:
        super().__init__(
            prob=cr,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_cr,
            n_var=None,
        )
    
    @property
    def cr(self) -> Tensor:
        """Current crossover rate."""
        return self.prob
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        """
        Exponential crossover.
        
        Args:
            parent1: Target vectors [n_pairs, n_var].
            parent2: Donor vectors [n_pairs, n_var].
        
        Returns:
            Trial vectors [n_pairs, n_var].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        cr = self.cr
        
        # Random start position for each individual
        j_rand = torch.randint(0, n_var, (n_pairs,), device=device)
        
        # Random numbers to determine segment length
        u = torch.rand(n_pairs, n_var, device=device)
        
        # Roll so column 0 is the starting position
        cols = torch.arange(n_var, device=device).unsqueeze(0)
        indices = (cols - j_rand.unsqueeze(1)) % n_var
        u_rolled = u.gather(1, indices)
        
        # Continuation mask: 1 while u < CR
        cont = (u_rolled < cr).float()
        cont[:, 0] = 1.0  # Always take at least one gene
        
        # Segment mask: 1 until first 0
        segment = torch.cumprod(cont, dim=1)
        
        # Roll back to original gene order
        mask = torch.zeros_like(segment)
        mask.scatter_(1, indices, segment)
        
        if not self.differentiable:
            # Hard mask
            mask = mask.detach()
        
        # Trial vector
        trial = mask * parent2 + (1.0 - mask) * parent1
        
        return trial
    
    def __repr__(self) -> str:
        return (
            f"ExponentialCrossover("
            f"cr={self.cr.item():.3f}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Uniform Crossover
# =============================================================================

class UniformCrossover(Crossover):
    """
    Uniform crossover.
    
    Each gene is independently selected from either parent
    with equal probability (0.5). Simpler than binomial
    crossover as there's no CR parameter.
    
    Args:
        prob: Probability of crossover occurring per individual.
        differentiable: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
    
    Example:
        >>> crossover = UniformCrossover()
        >>> offspring = crossover(parent1, parent2)
    """
    
    def __init__(
        self,
        prob: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=False,
            n_var=None,
        )
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        n_pairs, n_var = parent1.shape
        device = parent1.device
        
        # 50-50 mask for each gene
        if self.differentiable:
            # Binary-Concrete with logits=0 (p=0.5)
            logits = torch.zeros(n_pairs, n_var, device=device)
            mask = self._binary_concrete(logits, hard=True)
        else:
            mask = (torch.rand(n_pairs, n_var, device=device) < 0.5).float()
        
        # Create offspring
        offspring = mask * parent1 + (1.0 - mask) * parent2
        
        # Apply per-individual crossover probability
        if not self.differentiable:
            do_cross = (torch.rand(n_pairs, 1, device=device) < self.prob).float()
            offspring = do_cross * offspring + (1 - do_cross) * parent1
        
        return offspring
    
    def __repr__(self) -> str:
        return f"UniformCrossover(prob={self.prob.item():.3f})"


# =============================================================================
# Arithmetic Crossover
# =============================================================================

class ArithmeticCrossover(Crossover):
    """
    Arithmetic (intermediate) crossover.
    
    Creates offspring as a weighted average of parents:
        offspring = alpha * parent1 + (1 - alpha) * parent2
    
    Args:
        alpha: Weighting factor. If None, sampled randomly
            from [0, 1] for each crossover.
        whole: If True, same alpha for all genes. If False,
            different alpha per gene.
        differentiable: If True, alpha is learnable.
        learn_alpha: If True, alpha is a learnable parameter.
    
    Example:
        >>> # Fixed alpha
        >>> crossover = ArithmeticCrossover(alpha=0.5)
        >>> 
        >>> # Random alpha per crossover
        >>> crossover = ArithmeticCrossover(alpha=None)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        whole: bool = True,
        differentiable: bool = False,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__(
            prob=1.0,
            differentiable=differentiable,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        self.whole = whole
        self._fixed_alpha = alpha is not None
        
        if alpha is not None:
            # Fixed alpha
            if learn_alpha and differentiable:
                self._alpha_logit = nn.Parameter(
                    torch.logit(torch.tensor(alpha))
                )
            else:
                self.register_buffer(
                    "_alpha_logit",
                    torch.logit(torch.tensor(alpha))
                )
        else:
            self.register_buffer("_alpha_logit", torch.tensor(0.0))
    
    @property
    def alpha(self) -> Optional[Tensor]:
        """Current alpha value (None if random)."""
        if self._fixed_alpha:
            return torch.sigmoid(self._alpha_logit)
        return None
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        if self._fixed_alpha:
            alpha = torch.sigmoid(self._alpha_logit)
        else:
            # Random alpha
            if self.whole:
                alpha = torch.rand(n_pairs, 1, device=device, dtype=dtype)
            else:
                alpha = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        
        offspring = alpha * parent1 + (1 - alpha) * parent2
        
        return offspring
    
    def __repr__(self) -> str:
        alpha_str = f"{self.alpha.item():.3f}" if self._fixed_alpha else "random"
        return f"ArithmeticCrossover(alpha={alpha_str}, whole={self.whole})"


# =============================================================================
# Multi-Point Crossover
# =============================================================================

class NPointCrossover(Crossover):
    """
    N-point crossover.
    
    Selects N random crossover points and alternates between
    parents at each point. Classic crossover operator for
    binary and real-coded GAs.
    
    Args:
        n_points: Number of crossover points (1 for single-point,
            2 for two-point, etc.).
        prob: Crossover probability per individual.
        differentiable: If True, use soft masks.
        temperature: Temperature for soft crossover.
    
    Example:
        >>> # Single-point crossover
        >>> crossover = NPointCrossover(n_points=1)
        >>> 
        >>> # Two-point crossover
        >>> crossover = NPointCrossover(n_points=2)
    """
    
    def __init__(
        self,
        n_points: int = 1,
        prob: float = 0.9,
        differentiable: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            prob=prob,
            differentiable=differentiable,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=False,
            n_var=None,
        )
        
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        
        self.n_points = n_points
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
    ) -> Tensor:
        n_pairs, n_var = parent1.shape
        device = parent1.device
        
        # Generate random crossover points
        # For each individual, select n_points positions
        points = torch.sort(
            torch.randint(1, n_var, (n_pairs, self.n_points), device=device),
            dim=1
        ).values
        
        # Create mask based on crossover points
        positions = torch.arange(n_var, device=device).unsqueeze(0)
        
        # Count how many crossover points are before each position
        # Even count -> parent1, odd count -> parent2
        count_before = (positions.unsqueeze(-1) >= points.unsqueeze(1)).sum(dim=-1)
        mask = (count_before % 2 == 0).float()
        
        offspring = mask * parent1 + (1.0 - mask) * parent2
        
        # Apply per-individual crossover probability
        if not self.differentiable:
            do_cross = (torch.rand(n_pairs, 1, device=device) < self.prob).float()
            offspring = do_cross * offspring + (1 - do_cross) * parent1
        
        return offspring
    
    def __repr__(self) -> str:
        return (
            f"NPointCrossover("
            f"n_points={self.n_points}, "
            f"prob={self.prob.item():.3f})"
        )
