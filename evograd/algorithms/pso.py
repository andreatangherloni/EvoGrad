"""
Particle Swarm Optimisation (PSO) implementation for EvoGrad.

This module provides a fully differentiable Particle Swarm Optimisation
algorithm that supports both classical and gradient-enabled optimisation modes.

PSO evolves a swarm of particles through:
    1. Velocity update: Combine inertia, cognitive, and social components
    2. Position update: Move particles according to velocity
    3. Personal best update: Track each particle's best position
    4. Global best update: Track the swarm's best position

Modes:
    - adaptive=False, differentiable=False: Classical PSO
    - adaptive=True, differentiable=False: Hyperparameters (inertia, c1, c2)
        are learnable via backpropagation
    - adaptive=False, differentiable=True: Particle positions are learnable
        via backpropagation
    - adaptive=True, differentiable=True: Both hyperparameters and positions
        are learnable

Example:
    >>> from evograd.algorithms import PSO
    >>> from evograd.core import Problem, minimize
    >>> 
    >>> problem = Problem(
    ...     objective=lambda x: (x**2).sum(dim=-1),
    ...     n_var=30,
    ...     xl=-100.0,
    ...     xu=100.0,
    ... )
    >>> 
    >>> # Classical PSO
    >>> pso = PSO(pop_size=100, inertia=0.7, c1=1.5, c2=1.5)
    >>> result = minimize(problem, pso, max_evals=10000)
    >>> 
    >>> # Adaptive PSO with learnable hyperparameters
    >>> pso = PSO(pop_size=100, adaptive=True)
    >>> result = minimize(problem, pso, max_evals=10000)

Reference:
    Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization.
    Proceedings of ICNN'95.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from evograd.core.algorithm import Algorithm

if TYPE_CHECKING:
    from evograd.core.problem import Problem

__all__ = ["PSO", "pso_default", "pso_constriction"]


class PSO(Algorithm):
    """
    Particle Swarm Optimisation (PSO) for continuous optimisation.
    
    PSO simulates a swarm of particles moving through the search space,
    influenced by their own best known position and the swarm's best
    known position.
    
    The velocity update equation is:
        v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
    
    where:
        - w: inertia weight (controls momentum)
        - c1: cognitive coefficient (attraction to personal best)
        - c2: social coefficient (attraction to global best)
        - r1, r2: random vectors in [0, 1]
    
    Args:
        pop_size: Swarm size (number of particles).
        inertia: Inertia weight w. Default: 0.7.
        c1: Cognitive coefficient. Default: 1.5.
        c2: Social coefficient. Default: 1.5.
        v_max_ratio: Maximum velocity as ratio of search space range.
            Default: 0.2 (20% of range).
        sampling: Operator for initial population generation.
        repair: Repair operator for constraint handling.
        adaptive: If True, hyperparameters (inertia, c1, c2) are
            learnable via backpropagation.
        differentiable: If True, particle positions are learnable
            via backpropagation.
        per_particle_coeffs: If True and adaptive=True, each particle
            has its own inertia, c1, c2 values. Default: False.
        dtype: Tensor dtype.
    
    Attributes:
        inertia: Current inertia weight.
        c1: Current cognitive coefficient.
        c2: Current social coefficient.
        velocity: Current velocity vectors [pop_size, n_var].
        p_best: Personal best positions [pop_size, n_var].
        p_best_fitness: Personal best fitness values [pop_size].
    
    Example:
        >>> # Classical PSO
        >>> pso = PSO(pop_size=50, inertia=0.7, c1=1.5, c2=1.5)
        >>> 
        >>> # Adaptive PSO with learnable coefficients
        >>> pso = PSO(pop_size=50, adaptive=True)
        >>> 
        >>> # Differentiable particle positions
        >>> pso = PSO(pop_size=50, differentiable=True)
        >>> 
        >>> # Fully differentiable
        >>> pso = PSO(pop_size=50, adaptive=True, differentiable=True)
        >>> 
        >>> # Per-particle adaptive coefficients
        >>> pso = PSO(pop_size=50, adaptive=True, per_particle_coeffs=True)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_ratio: float = 0.2,
        sampling: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        adaptive: bool = False,
        differentiable: bool = False,
        per_particle_coeffs: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.adaptive = adaptive
        self.per_particle_coeffs = per_particle_coeffs
        self._init_inertia = w
        self._init_c1 = c1
        self._init_c2 = c2
        self._v_max_ratio = v_max_ratio
        
        # PSO doesn't use standard EA operators (selection, crossover, mutation)
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=None,
            crossover=None,
            mutation=None,
            survival=None,
            repair=repair,
            eliminate_duplicates=False,
            n_offsprings=pop_size,
            differentiable=differentiable,
            adaptive=adaptive,
            dtype=dtype,
        )
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    def _setup(self) -> None:
        """PSO-specific setup after initialization."""
        n_var = self.problem.n_var
        N = self.pop_size
        
        # Compute velocity bounds
        search_range = self.xu - self.xl
        v_max = self._v_max_ratio * search_range
        v_min = -v_max
        
        # Register velocity bounds
        self.register_buffer("_v_max", v_max)
        self.register_buffer("_v_min", v_min)
        
        # Initialize velocities to zero
        self.register_buffer(
            "_velocity",
            torch.zeros(N, n_var, device=self.device, dtype=self.dtype)
        )
        
        # Initialize personal bests
        self.register_buffer(
            "_p_best",
            self._population.clone().detach()
        )
        self.register_buffer(
            "_p_best_fitness",
            self.state.fitness.clone().detach()
        )
        
        # Setup hyperparameters
        self._setup_hyperparameters(N, n_var)
    
    def _setup_hyperparameters(self, N: int, n_var: int) -> None:
        """Setup inertia, c1, c2 as learnable or fixed parameters."""
        if self.adaptive:
            if self.per_particle_coeffs:
                # Per-particle coefficients [N, 1] for broadcasting
                self._inertia = nn.Parameter(
                    torch.full((N, 1), self._init_inertia, 
                               device=self.device, dtype=self.dtype)
                )
                self._c1 = nn.Parameter(
                    torch.full((N, 1), self._init_c1,
                               device=self.device, dtype=self.dtype)
                )
                self._c2 = nn.Parameter(
                    torch.full((N, 1), self._init_c2,
                               device=self.device, dtype=self.dtype)
                )
            else:
                # Scalar coefficients (shared by all particles)
                self._inertia = nn.Parameter(
                    torch.tensor(self._init_inertia,
                                 device=self.device, dtype=self.dtype)
                )
                self._c1 = nn.Parameter(
                    torch.tensor(self._init_c1,
                                 device=self.device, dtype=self.dtype)
                )
                self._c2 = nn.Parameter(
                    torch.tensor(self._init_c2,
                                 device=self.device, dtype=self.dtype)
                )
        else:
            # Fixed coefficients (buffers)
            self.register_buffer(
                "_inertia",
                torch.tensor(self._init_inertia, device=self.device, dtype=self.dtype)
            )
            self.register_buffer(
                "_c1",
                torch.tensor(self._init_c1, device=self.device, dtype=self.dtype)
            )
            self.register_buffer(
                "_c2",
                torch.tensor(self._init_c2, device=self.device, dtype=self.dtype)
            )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def population(self) -> Tensor:
        """Current particle positions."""
        return self._population
    
    @property
    def fitness(self) -> Tensor:
        """Current fitness values."""
        return self.state.fitness
    
    @property
    def velocity(self) -> Tensor:
        """Current velocity vectors."""
        return self._velocity
    
    @property
    def p_best(self) -> Tensor:
        """Personal best positions."""
        return self._p_best
    
    @property
    def p_best_fitness(self) -> Tensor:
        """Personal best fitness values."""
        return self._p_best_fitness
    
    @property
    def inertia(self) -> Tensor:
        """Current inertia weight."""
        return self._inertia
    
    @property
    def c1(self) -> Tensor:
        """Current cognitive coefficient."""
        return self._c1
    
    @property
    def c2(self) -> Tensor:
        """Current social coefficient."""
        return self._c2
    
    # =========================================================================
    # Core PSO Methods
    # =========================================================================
    
    def _update_velocity(self) -> Tensor:
        """
        Compute new velocities using the PSO velocity update equation.
        
        v_new = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        
        Returns:
            New velocity vectors [pop_size, n_var].
        """
        N, D = self.pop_size, self.n_var
        
        # Random vectors
        r1 = torch.rand(N, D, device=self.device, dtype=self.dtype)
        r2 = torch.rand(N, D, device=self.device, dtype=self.dtype)
        
        # Get global best
        g_best = self.state.best_solution
        if g_best is None:
            best_idx = torch.argmin(self._p_best_fitness)
            g_best = self._p_best[best_idx]
        
        # Velocity components
        inertia_term = self.inertia * self._velocity
        cognitive_term = self.c1 * r1 * (self._p_best - self.population)
        social_term = self.c2 * r2 * (g_best.unsqueeze(0) - self.population)
        
        # New velocity
        v_new = inertia_term + cognitive_term + social_term
        
        # Clamp velocity
        v_new = torch.clamp(v_new, self._v_min, self._v_max)
        
        return v_new
    
    def _update_position(self, velocity: Tensor) -> Tensor:
        """
        Update particle positions based on velocity.
        
        Args:
            velocity: Velocity vectors [pop_size, n_var].
        
        Returns:
            New positions [pop_size, n_var].
        """
        new_pos = self.population + velocity
        return new_pos
    
    def _reflect_bounds(self, position: Tensor, velocity: Tensor) -> tuple:
        """
        Handle boundary violations by reflection.
        
        When a particle hits a boundary, it bounces back and its
        velocity component is reversed.
        
        Args:
            position: Particle positions [pop_size, n_var].
            velocity: Particle velocities [pop_size, n_var].
        
        Returns:
            Tuple of (repaired_position, repaired_velocity).
        """
        # Detect violations
        below = position < self.xl
        above = position > self.xu
        
        # Reflect positions
        pos_repaired = position.clone()
        pos_repaired = torch.where(below, 2 * self.xl - position, pos_repaired)
        pos_repaired = torch.where(above, 2 * self.xu - position, pos_repaired)
        
        # Clamp to ensure within bounds (in case of large violations)
        pos_repaired = torch.clamp(pos_repaired, self.xl, self.xu)
        
        # Reverse velocity at boundaries
        vel_repaired = velocity.clone()
        vel_repaired = torch.where(below | above, -velocity, vel_repaired)
        
        return pos_repaired, vel_repaired
    
    def _update_personal_best(
        self,
        new_pos: Tensor,
        new_fitness: Tensor,
    ) -> tuple:
        """
        Update personal best positions where improved.
        
        Args:
            new_pos: New particle positions [pop_size, n_var].
            new_fitness: Fitness at new positions [pop_size].
        
        Returns:
            Tuple of (new_p_best, new_p_best_fitness).
        """
        improved = new_fitness < self._p_best_fitness
        
        new_p_best = torch.where(
            improved.unsqueeze(-1),
            new_pos,
            self._p_best
        )
        new_p_best_fitness = torch.where(
            improved,
            new_fitness,
            self._p_best_fitness
        )
        
        return new_p_best, new_p_best_fitness
    
    def _infill(self) -> Tensor:
        """
        Generate new particle positions through velocity update.
        
        Returns:
            New positions [pop_size, n_var].
        """
        # 1. Update velocities
        new_velocity = self._update_velocity()
        
        # 2. Update positions
        new_pos = self._update_position(new_velocity)
        
        # 3. Handle boundary violations
        if self.repair is not None:
            new_pos = self.repair(new_pos, self.xl, self.xu)
            # Recompute velocity to match repaired position
            new_velocity = new_pos - self.population
        else:
            new_pos, new_velocity = self._reflect_bounds(new_pos, new_velocity)
        
        # Store velocity for state update
        self._pending_velocity = new_velocity
        
        return new_pos
    
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Update swarm state with new positions.
        
        Args:
            offspring: New particle positions [pop_size, n_var].
            offspring_fitness: Fitness at new positions [pop_size].
        """
        # Update personal bests
        new_p_best, new_p_best_fitness = self._update_personal_best(
            offspring, offspring_fitness
        )
        
        # Update state
        self._update_state(
            new_pos=offspring,
            new_fitness=offspring_fitness,
            new_velocity=self._pending_velocity,
            new_p_best=new_p_best,
            new_p_best_fitness=new_p_best_fitness,
        )
        
        # Update global best
        self.state.update_best(self.population, self.state.fitness)
        
        # Cleanup
        if hasattr(self, '_pending_velocity'):
            del self._pending_velocity
    
    def _update_state(
        self,
        new_pos: Tensor,
        new_fitness: Tensor,
        new_velocity: Tensor,
        new_p_best: Tensor,
        new_p_best_fitness: Tensor,
    ) -> None:
        """Update all PSO state tensors."""
        with torch.no_grad():
            self._population.copy_(new_pos)
            self._velocity.copy_(new_velocity)
            self._p_best.copy_(new_p_best)
            self._p_best_fitness.copy_(new_p_best_fitness)
        
        self.state.fitness = new_fitness
        self.state.population = self._population
    
    # =========================================================================
    # Hyperparameter Management
    # =========================================================================
    
    @torch.no_grad()
    def _clamp_hyperparams(self) -> None:
        """Clamp learnable hyperparameters to valid ranges."""
        if self.adaptive:
            # Inertia in [0, 1.5]
            self._inertia.clamp_(min=0.0, max=1.5)
            # c1, c2 in [0, 4]
            self._c1.clamp_(min=0.0, max=4.0)
            self._c2.clamp_(min=0.0, max=4.0)
    
    def update_state(self) -> None:
        """Commit pending changes and clamp hyperparameters."""
        super().update_state()
        self._clamp_hyperparams()
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameter values."""
        def _to_float(x: Tensor) -> float:
            if x.numel() == 1:
                return float(x.item())
            return float(x.mean().item())
        
        return {
            'pop_size': self.pop_size,
            'inertia': _to_float(self.inertia),
            'c1': _to_float(self.c1),
            'c2': _to_float(self.c2),
            'v_max_ratio': self._v_max_ratio,
            'adaptive': self.adaptive,
            'differentiable': self.differentiable,
            'per_particle_coeffs': self.per_particle_coeffs,
        }
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        def _fmt(x: Tensor) -> str:
            if x.numel() == 1:
                return f"{float(x.item()):.3f}"
            return f"{float(x.mean().item()):.3f}"
        
        return (
            f"PSO(pop_size={self.pop_size}, "
            f"w={_fmt(self.inertia)}, "
            f"c1={_fmt(self.c1)}, "
            f"c2={_fmt(self.c2)}, "
            f"adaptive={self.adaptive}, "
            f"differentiable={self.differentiable})"
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def pso_default(
    pop_size: int = 100,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> PSO:
    """
    Create PSO with default settings.
    
    Uses standard coefficients:
        - inertia = 0.7
        - c1 = c2 = 1.5
    
    Args:
        pop_size: Swarm size.
        adaptive: If True, hyperparameters are learnable.
        differentiable: If True, positions are learnable.
        **kwargs: Additional arguments passed to PSO.
    
    Returns:
        Configured PSO instance.
    """
    return PSO(
        pop_size=pop_size,
        w=0.7,
        c1=1.5,
        c2=1.5,
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def pso_constriction(
    pop_size: int = 100,
    adaptive: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> PSO:
    """
    Create PSO with constriction coefficients.
    
    Uses Clerc's constriction factor approach:
        - inertia = 0.7298
        - c1 = c2 = 1.4962
    
    This configuration provides guaranteed convergence.
    
    Args:
        pop_size: Swarm size.
        adaptive: If True, hyperparameters are learnable.
        differentiable: If True, positions are learnable.
        **kwargs: Additional arguments passed to PSO.
    
    Returns:
        Configured PSO instance.
    
    Reference:
        Clerc, M. & Kennedy, J. (2002). The particle swarm - explosion,
        stability, and convergence in a multidimensional complex space.
    """
    # Constriction coefficients
    phi = 4.1
    chi = 2.0 / abs(2.0 - phi - (phi**2 - 4*phi)**0.5)
    
    return PSO(
        pop_size=pop_size,
        w=chi,  # ~0.7298
        c1=chi * 2.05,  # ~1.4962
        c2=chi * 2.05,  # ~1.4962
        adaptive=adaptive,
        differentiable=differentiable,
        **kwargs,
    )


def pso_adaptive(
    pop_size: int = 100,
    per_particle: bool = False,
    differentiable: bool = False,
    **kwargs,
) -> PSO:
    """
    Create PSO with adaptive (learnable) hyperparameters.
    
    Args:
        pop_size: Swarm size.
        per_particle: If True, each particle has its own coefficients.
        differentiable: If True, positions are also learnable.
        **kwargs: Additional arguments passed to PSO.
    
    Returns:
        Configured PSO instance with adaptive=True.
    """
    return PSO(
        pop_size=pop_size,
        adaptive=True,
        differentiable=differentiable,
        per_particle_coeffs=per_particle,
        **kwargs,
    )