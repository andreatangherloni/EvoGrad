"""
Crossover operators for recombination.

This module provides crossover (recombination) operators that
combine genetic information from parent individuals to create
offspring. All operators support both classical and differentiable
(i.e., adaptive) modes.

Available crossovers:
    - SBXCrossover: Simulated Binary Crossover (GA)
    - BlendCrossover: BLX-alpha crossover (GA)
    - BinomialCrossover: DE-style binomial crossover
    - ExponentialCrossover: DE-style exponential crossover
    - UniformCrossover: Simple uniform crossover
    - ArithmeticCrossover: Weighted average of parents
    - NPointCrossover: N-point crossover

Differentiable Mode:
    When `adaptive=True`, crossover masks use Binary-Concrete
    (Gumbel-Sigmoid) relaxation with straight-through estimator,
    allowing gradients to flow through crossover decisions.

Per-Individual/Per-Gene Parameters:
    All crossover operators support four parameter configurations via
    optional runtime overrides in forward(). This is essential for
    self-adaptive algorithms like SHADE, jDE, or self-adaptive GAs.
    
    Configurations:
        - Fixed (scalar): Same value for all individuals and genes
        - Per-gene [D]: Different value per gene, same across individuals
        - Per-individual [N]: Different value per individual, same across genes
        - Per-gene + Per-individual [N, D]: Full matrix, different for each
    
    Example:
        >>> # SHADE-style per-individual CR
        >>> cr_per_ind = torch.rand(pop_size)  # [N]
        >>> trial = crossover(target, donor, cr=cr_per_ind)
        >>> 
        >>> # Per-gene CR
        >>> cr_per_gene = torch.rand(n_var)  # [D]
        >>> trial = crossover(target, donor, cr=cr_per_gene)
        >>> 
        >>> # Full matrix
        >>> cr_matrix = torch.rand(pop_size, n_var)  # [N, D]
        >>> trial = crossover(target, donor, cr=cr_matrix)

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
    ...     adaptive=True,
    ...     learn_eta=True,
    ... )
    >>> offspring = crossover(parent1, parent2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from torch import Tensor

from evograd.operators.relaxations import binary_concrete, expand_param

__all__ = [
    "Crossover",
    "SBXCrossover",
    "BlendCrossover",
    "BinomialCrossover",
    "ExponentialCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
    "NPointCrossover",
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
        adaptive: If True, use Binary-Concrete for soft masks.
        temperature: Temperature for Binary-Concrete.
        learn_temperature: If True, temperature is learnable.
        learn_prob: If True, crossover probability is learnable.
        n_var: Number of variables (for per-gene probability).
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional parameter overrides that
        support four configurations:
        
        - scalar: Fixed value for all individuals and genes
        - [D] tensor: Per-gene values (same across individuals)
        - [N] tensor: Per-individual values (same across genes)
        - [N, D] tensor: Full matrix (different for each individual and gene)
        
        When an override is provided, it takes precedence over the stored
        parameter. This enables self-adaptive algorithms like SHADE.
    """
    
    def __init__(
        self,
        prob: float = 0.9,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_temperature: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self._MIN_TEMPERATURE = 0.05
        self._MAX_TEMPERATURE = 10.0
        
        self.adaptive = adaptive
        self.n_var = n_var
        
        # Temperature parameter (log for positivity)
        if learn_temperature and adaptive:
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
        if learn_prob and adaptive:
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
    
    def _clamp_temperature(self):
        if hasattr(self, "_log_temperature") and self._log_temperature is not None:
            with torch.no_grad():
                self._log_temperature.clamp_(
                    math.log(self._MIN_TEMPERATURE),
                    math.log(self._MAX_TEMPERATURE),
                )
            
    @abstractmethod
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Perform crossover between parent pairs.
        
        Args:
            parent1: First parents [n_pairs, n_var].
            parent2: Second parents [n_pairs, n_var].
            **kwargs: Optional per-individual/per-gene parameter overrides.
        
        Returns:
            Offspring [n_pairs, n_var].
        """
        pass
    
    def forward(
        self,
        parent1: Tensor,
        parent2: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Apply crossover to parent pairs.
        
        Args:
            parent1: First parents [n_pairs, n_var].
            parent2: Second parents [n_pairs, n_var].
            **kwargs: Optional parameter overrides for per-individual or
                per-gene operation. Supported kwargs depend on the specific
                crossover operator (e.g., cr, eta, prob, alpha).
                
                Each parameter can be:
                - scalar: Fixed value for all
                - [D] tensor: Per-gene values
                - [N] tensor: Per-individual values  
                - [N, D] tensor: Full matrix
        
        Returns:
            Offspring [n_pairs, n_var].
        
        Example:
            >>> # Standard call (uses stored parameters)
            >>> offspring = crossover(parent1, parent2)
            >>> 
            >>> # Per-individual CR override (for SHADE)
            >>> offspring = crossover(parent1, parent2, cr=cr_per_individual)
            >>> 
            >>> # Per-gene eta override
            >>> offspring = crossover(parent1, parent2, eta=eta_per_gene)
        """
        
        self._clamp_temperature()
        return self._crossover(parent1, parent2, **kwargs)
    
    def __call__(
        self,
        parent1: Tensor,
        parent2: Tensor,
        **kwargs,
    ) -> Tensor:
        """Apply crossover (alias for forward)."""
        return self.forward(parent1, parent2, **kwargs)


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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_eta: If True, eta is learnable.
        learn_prob: If True, crossover probability is learnable.
        n_var: Number of variables (for per-gene probability).
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - eta: Distribution index [scalar, D, N, or N×D]
        - prob: Crossover probability [scalar, D, N, or N×D]
    
    Example:
        >>> crossover = SBXCrossover(eta=15, prob=0.9)
        >>> offspring = crossover(parent1, parent2)
        >>> 
        >>> # Per-individual eta (for self-adaptive GA)
        >>> eta_per_ind = torch.rand(pop_size) * 20 + 5  # [N]
        >>> offspring = crossover(parent1, parent2, eta=eta_per_ind)
    
    Reference:
        Deb & Agrawal (1995). Simulated Binary Crossover for
        Continuous Search Space.
    """
    
    def __init__(
        self,
        eta: float = 15.0,
        prob: float = 0.9,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_eta: bool = True,
        learn_prob: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=learn_prob,
            n_var=n_var,
        )
        
        # Eta parameter (log for positivity)
        if learn_eta and adaptive:
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
        eta: Optional[Tensor] = None,
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Perform SBX crossover.
        
        Args:
            parent1: First parents [N, D].
            parent2: Second parents [N, D].
            eta: Optional distribution index override [scalar, D, N, or N×D].
            prob: Optional crossover probability override [scalar, D, N, or N×D].
        
        Returns:
            Offspring [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Expand eta to [N, D]
        eta_expanded = expand_param(eta, self.eta, n_pairs, n_var, device, dtype)
        
        # Expand prob to [N, D]
        prob_expanded = expand_param(prob, self.prob, n_pairs, n_var, device, dtype)
        
        # Get crossover mask (which genes to cross)
        if self.adaptive:
            # Convert prob to logits for Binary-Concrete
            prob_logits = torch.logit(prob_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                prob_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            # Hard Bernoulli mask
            mask = (torch.rand(n_pairs, n_var, device=device) < prob_expanded).float()
        
        # Compute SBX spread factor beta
        u = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        
        beta = torch.where(
            u <= 0.5,
            (2 * u).pow(1.0 / (eta_expanded + 1)),
            (2 * (1 - u)).pow(-1.0 / (eta_expanded + 1))
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
            f"adaptive={self.adaptive})"
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
        adaptive: If True, use soft interpolation.
        learn_alpha: If True, alpha is learnable.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - alpha: Extension factor [scalar, D, N, or N×D]
        - prob: Crossover probability [scalar, D, N, or N×D]
    
    Example:
        >>> crossover = BlendCrossover(alpha=0.5)
        >>> offspring = crossover(parent1, parent2)
        >>> 
        >>> # Per-individual alpha
        >>> alpha_per_ind = torch.rand(pop_size)  # [N]
        >>> offspring = crossover(parent1, parent2, alpha=alpha_per_ind)
    
    Reference:
        Eshelman & Schaffer (1993). Real-Coded Genetic Algorithms
        and Interval-Schemata.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        prob: float = 0.9,
        adaptive: bool = False,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        # Alpha as sigmoid(logit) to keep in [0, 2]
        # alpha = 2 * sigmoid(logit), so logit = logit(alpha/2)
        alpha_logit = torch.logit(torch.tensor(alpha / 2.0).clamp(1e-7, 1 - 1e-7))
        
        if learn_alpha and adaptive:
            self._alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("_alpha_logit", alpha_logit)
    
    @property
    def alpha(self) -> Tensor:
        """Current alpha value in [0, 2]."""
        return 2.0 * torch.sigmoid(self._alpha_logit)
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
        alpha: Optional[Tensor] = None,
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Perform blend crossover.
        
        Args:
            parent1: First parents [N, D].
            parent2: Second parents [N, D].
            alpha: Optional extension factor override [scalar, D, N, or N×D].
            prob: Optional crossover probability override [scalar, D, N, or N×D].
        
        Returns:
            Offspring [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Expand alpha to [N, D]
        alpha_expanded = expand_param(alpha, self.alpha, n_pairs, n_var, device, dtype)
        
        # Expand prob to [N, D] (but we use per-individual for blend)
        prob_expanded = expand_param(prob, self.prob, n_pairs, n_var, device, dtype)
        
        # Determine interval bounds
        p_min = torch.minimum(parent1, parent2)
        p_max = torch.maximum(parent1, parent2)
        diff = p_max - p_min
        
        # Extended interval
        lower = p_min - alpha_expanded * diff
        upper = p_max + alpha_expanded * diff
        
        # Sample uniformly from interval
        u = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        offspring = lower + u * (upper - lower)
        
        # Apply crossover probability (per individual, use first column)
        if not self.adaptive:
            do_cross = (torch.rand(n_pairs, 1, device=device) < prob_expanded[:, :1]).float()
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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
        learn_cr: If True, crossover rate is learnable.
        n_var: Number of variables (for per-gene CR).
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - cr: Crossover rate [scalar, D, N, or N×D]
        
        This is essential for SHADE/L-SHADE where each individual
        has its own CR value sampled from the historical memory.
    
    Example:
        >>> # target = current individual, donor = mutant vector
        >>> crossover = BinomialCrossover(cr=0.9)
        >>> trial = crossover(target, donor)
        >>> 
        >>> # SHADE-style per-individual CR
        >>> cr_per_ind = torch.rand(pop_size)  # [N]
        >>> trial = crossover(target, donor, cr=cr_per_ind)
    
    Note:
        In DE terminology:
        - parent1 = target vector (current individual)
        - parent2 = donor vector (mutant)
        - output = trial vector
    """
    
    def __init__(
        self,
        cr: float = 0.9,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_cr: bool = True,
        n_var: Optional[int] = None,
    ) -> None:
        super().__init__(
            prob=cr,
            adaptive=adaptive,
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
        cr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Binomial crossover.
        
        Args:
            parent1: Target vectors [N, D].
            parent2: Donor vectors [N, D].
            cr: Optional crossover rate override [scalar, D, N, or N×D].
        
        Returns:
            Trial vectors [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Expand CR to [N, D]
        cr_expanded = expand_param(cr, self.cr, n_pairs, n_var, device, dtype)
        
        if self.adaptive:
            # Convert CR to logits for Binary-Concrete
            cr_logits = torch.logit(cr_expanded.clamp(1e-7, 1 - 1e-7))
            mask = binary_concrete(
                cr_logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            # Hard Bernoulli mask
            mask = (torch.rand(n_pairs, n_var, device=device) < cr_expanded).float()
        
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
            f"adaptive={self.adaptive})"
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
        adaptive: If True, use soft approximation.
        temperature: Temperature for soft crossover.
        learn_cr: If True, crossover rate is learnable.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - cr: Crossover rate [scalar or N] (per-gene not supported
              for exponential due to contiguous segment nature)
    
    Example:
        >>> crossover = ExponentialCrossover(cr=0.9)
        >>> trial = crossover(target, donor)
        >>> 
        >>> # Per-individual CR
        >>> cr_per_ind = torch.rand(pop_size)  # [N]
        >>> trial = crossover(target, donor, cr=cr_per_ind)
    
    Note:
        Exponential crossover tends to preserve more structure
        from the target vector compared to binomial crossover.
    """
    
    def __init__(
        self,
        cr: float = 0.9,
        adaptive: bool = False,
        temperature: float = 1.0,
        learn_cr: bool = True,
    ) -> None:
        super().__init__(
            prob=cr,
            adaptive=adaptive,
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
        cr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Exponential crossover.
        
        Args:
            parent1: Target vectors [N, D].
            parent2: Donor vectors [N, D].
            cr: Optional crossover rate override [scalar or N].
                Note: Per-gene CR not supported for exponential crossover.
        
        Returns:
            Trial vectors [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Get CR value (scalar or per-individual [N])
        if cr is None:
            cr_val = self.cr
        else:
            cr_val = cr if isinstance(cr, Tensor) else torch.tensor(cr)

        # Ensure on correct device
        cr_val = cr_val.to(device=device, dtype=dtype)
        
        # Expand to [N] if scalar
        if cr_val.dim() == 0:
            cr_val = cr_val.expand(n_pairs)
        elif cr_val.dim() == 1 and cr_val.shape[0] != n_pairs:
            raise ValueError(f"CR must be scalar or [N={n_pairs}], got [{cr_val.shape[0]}]")
        elif cr_val.dim() == 2:
            # For exponential, use mean across genes if [N, D] provided
            cr_val = cr_val.mean(dim=1)
        
        # Random start position for each individual
        j_rand = torch.randint(0, n_var, (n_pairs,), device=device)
        
        # Random numbers to determine segment length
        u = torch.rand(n_pairs, n_var, device=device)
        
        # Roll so column 0 is the starting position
        cols = torch.arange(n_var, device=device).unsqueeze(0)
        indices = (cols - j_rand.unsqueeze(1)) % n_var
        u_rolled = u.gather(1, indices)
        
        # Continuation mask: 1 while u < CR (per-individual CR)
        cr_expanded = cr_val.unsqueeze(1)  # [N, 1]
        cont = (u_rolled < cr_expanded).float()
        cont[:, 0] = 1.0  # Always take at least one gene
        
        # Segment mask: 1 until first 0
        segment = torch.cumprod(cont, dim=1)
        
        # Roll back to original gene order
        mask = torch.zeros_like(segment)
        mask.scatter_(1, indices, segment)
        
        if not self.adaptive:
            # Hard mask
            mask = mask.detach()
        
        # Trial vector
        trial = mask * parent2 + (1.0 - mask) * parent1
        
        return trial
    
    def __repr__(self) -> str:
        return (
            f"ExponentialCrossover("
            f"cr={self.cr.item():.3f}, "
            f"adaptive={self.adaptive})"
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
        adaptive: If True, use Binary-Concrete masks.
        temperature: Temperature for Binary-Concrete.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - prob: Crossover probability [scalar, D, N, or N×D]
    
    Example:
        >>> crossover = UniformCrossover()
        >>> offspring = crossover(parent1, parent2)
    """
    
    def __init__(
        self,
        prob: float = 0.9,
        adaptive: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
            temperature=temperature,
            learn_temperature=True,
            learn_prob=False,
            n_var=None,
        )
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Uniform crossover.
        
        Args:
            parent1: First parents [N, D].
            parent2: Second parents [N, D].
            prob: Optional crossover probability override [scalar, D, N, or N×D].
        
        Returns:
            Offspring [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # 50-50 mask for each gene
        if self.adaptive:
            # Binary-Concrete with logits=0 (p=0.5)
            logits = torch.zeros(n_pairs, n_var, device=device)
            mask = binary_concrete(
                logits, 
                temperature=self.temperature  # Pass temperature
            )
        else:
            mask = (torch.rand(n_pairs, n_var, device=device) < 0.5).float()
        
        # Create offspring
        offspring = mask * parent1 + (1.0 - mask) * parent2
        
        # Apply per-individual crossover probability
        if not self.adaptive:
            # Expand prob to [N, D]
            prob_expanded = expand_param(prob, self.prob, n_pairs, n_var, device, dtype)
            do_cross = (torch.rand(n_pairs, 1, device=device) < prob_expanded[:, :1]).float()
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
        adaptive: If True, alpha is learnable.
        learn_alpha: If True, alpha is a learnable parameter.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - alpha: Weighting factor [scalar, D, N, or N×D]
    
    Example:
        >>> # Fixed alpha
        >>> crossover = ArithmeticCrossover(alpha=0.5, whole=True)
        >>> offspring = crossover(parent1, parent2)
        >>> 
        >>> # Per-individual alpha
        >>> alpha_per_ind = torch.rand(pop_size)  # [N]
        >>> offspring = crossover(parent1, parent2, alpha=alpha_per_ind)
    """
    
    def __init__(
        self,
        alpha: Optional[float] = 0.5,
        whole: bool = True,
        adaptive: bool = False,
        learn_alpha: bool = True,
    ) -> None:
        super().__init__(
            prob=1.0,
            adaptive=adaptive,
            temperature=1.0,
            learn_temperature=False,
            learn_prob=False,
            n_var=None,
        )
        
        self.whole = whole
        self._random_alpha = alpha is None
        
        if alpha is not None:
            # Alpha as sigmoid(logit) to keep in [0, 1]
            alpha_logit = torch.logit(torch.tensor(alpha).clamp(1e-7, 1 - 1e-7))
            
            if learn_alpha and adaptive:
                self._alpha_logit = nn.Parameter(alpha_logit)
            else:
                self.register_buffer("_alpha_logit", alpha_logit)
        else:
            self.register_buffer("_alpha_logit", torch.tensor(0.0))
    
    @property
    def alpha(self) -> Optional[Tensor]:
        """Current alpha value (None if random)."""
        if self._random_alpha:
            return None
        return torch.sigmoid(self._alpha_logit)
    
    def _crossover(
        self,
        parent1: Tensor,
        parent2: Tensor,
        alpha: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Arithmetic crossover.
        
        Args:
            parent1: First parents [N, D].
            parent2: Second parents [N, D].
            alpha: Optional weighting factor override [scalar, D, N, or N×D].
        
        Returns:
            Offspring [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
        # Determine alpha value
        if alpha is not None:
            # Use provided override
            alpha_val = alpha
        elif self._random_alpha:
            # Sample random alpha
            if self.whole:
                alpha_val = torch.rand(n_pairs, 1, device=device, dtype=dtype)
            else:
                alpha_val = torch.rand(n_pairs, n_var, device=device, dtype=dtype)
        else:
            # Use stored alpha
            alpha_val = self.alpha
        
        # Expand alpha to [N, D]
        alpha_expanded = expand_param(alpha_val, torch.tensor(0.5), n_pairs, n_var, device, dtype)
        
        # Weighted average
        offspring = alpha_expanded * parent1 + (1.0 - alpha_expanded) * parent2
        
        return offspring
    
    def __repr__(self) -> str:
        if self._random_alpha:
            return f"ArithmeticCrossover(alpha=random, whole={self.whole})"
        return f"ArithmeticCrossover(alpha={self.alpha.item():.3f}, whole={self.whole})"


# =============================================================================
# N-Point Crossover
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
        adaptive: If True, use soft masks.
        temperature: Temperature for soft crossover.
    
    Per-Individual/Per-Gene Parameters:
        The forward() method accepts optional overrides:
        - prob: Crossover probability [scalar, D, N, or N×D]
        
        Note: n_points cannot be overridden per-individual.
    
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
        adaptive: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            prob=prob,
            adaptive=adaptive,
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
        prob: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        N-point crossover.
        
        Args:
            parent1: First parents [N, D].
            parent2: Second parents [N, D].
            prob: Optional crossover probability override [scalar, D, N, or N×D].
        
        Returns:
            Offspring [N, D].
        """
        n_pairs, n_var = parent1.shape
        device = parent1.device
        dtype = parent1.dtype
        
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
        if not self.adaptive:
            # Expand prob to [N, D]
            prob_expanded = expand_param(prob, self.prob, n_pairs, n_var, device, dtype)
            do_cross = (torch.rand(n_pairs, 1, device=device) < prob_expanded[:, :1]).float()
            offspring = do_cross * offspring + (1 - do_cross) * parent1
        
        return offspring
    
    def __repr__(self) -> str:
        return (
            f"NPointCrossover("
            f"n_points={self.n_points}, "
            f"prob={self.prob.item():.3f})"
        )
