"""
Abstract base class for all EvoGrad algorithms.

This module provides the foundation for population-based optimisers,
supporting both differentiable (gradient-enabled) and classical modes.

The design follows an infill/advance pattern inspired by pymoo:
    - _infill(): Generate offspring using evolutionary operators
    - _advance(): Update population state based on offspring fitness

Algorithms receive operators via dependency injection (pymoo-style):
    - sampling: Initial population generation
    - selection: Parent selection (optional)
    - crossover: Recombination operator (optional)
    - mutation: Perturbation operator (optional)
    - survival: Survivor selection (optional)
    - repair: Constraint handling (optional)

For differentiable mode, the entire generation is a differentiable
computation graph, enabling gradient-based hyperparameter learning.

Example:
    >>> from evograd.core import Problem
    >>> from evograd.operators import FloatRandomSampling, TournamentSelection
    >>> 
    >>> problem = Problem(objective=ackley, n_var=30, xl=-100.0, xu=100.0)
    >>> 
    >>> ga = GA(
    ...     pop_size=100,
    ...     sampling=FloatRandomSampling(),
    ...     selection=TournamentSelection(tournament_size=3),
    ...     crossover=SBXCrossover(eta=15, prob=0.9),
    ...     mutation=PolynomialMutation(eta=20),
    ...     differentiable=True,
    ... )
    >>> 
    >>> ga.initialize(problem)
    >>> result = minimize(ga, problem, max_evals=10000)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import torch
import torch.nn as nn

from evograd.utils.device import set_seed
from evograd.utils.duplicates import DuplicateEliminator, DuplicateMethod
from evograd.operators.sampling import UniformSampling

if TYPE_CHECKING:
    from torch import Tensor
    from evograd.core.problem import Problem

__all__ = [
    "Algorithm",
    "AlgorithmState",
]


# =============================================================================
# Algorithm State Container
# =============================================================================

class AlgorithmState:
    """
    Container for algorithm state that can be saved/loaded.
    
    This separates the persistent state from the algorithm logic,
    making it easier to checkpoint and resume optimisation.
    
    Attributes:
        generation: Current generation number.
        n_evals: Total fitness evaluations performed.
        population: Current population tensor.
        fitness: Current fitness values.
        best_fitness: Best fitness found so far.
        best_solution: Best solution found so far.
        hyperparams: Dictionary of current hyperparameter values.
        extra: Dictionary for algorithm-specific state.
    """
    
    def __init__(self) -> None:
        self.generation: int = 0
        self.n_evals: int = 0
        self.population: Optional[Tensor] = None
        self.fitness: Optional[Tensor] = None
        self.best_fitness: float = float('inf')
        self.best_solution: Optional[Tensor] = None
        self.hyperparams: Dict[str, Any] = {}
        self.extra: Dict[str, Any] = {}
    
    def update_best(self, population: Tensor, fitness: Tensor) -> None:
        """Update best solution if improved."""
        best_idx = torch.argmin(fitness)
        best_val = float(fitness[best_idx].detach())
        
        if best_val < self.best_fitness:
            self.best_fitness = best_val
            self.best_solution = population[best_idx].detach().clone()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialisation."""
        return {
            "generation": self.generation,
            "n_evals": self.n_evals,
            "population": self.population.detach().cpu() if self.population is not None else None,
            "fitness": self.fitness.detach().cpu() if self.fitness is not None else None,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution.detach().cpu() if self.best_solution is not None else None,
            "hyperparams": {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.hyperparams.items()
            },
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> "AlgorithmState":
        """Restore state from dictionary."""
        state = cls()
        state.generation = data["generation"]
        state.n_evals = data["n_evals"]
        state.best_fitness = data["best_fitness"]
        
        if data["population"] is not None:
            state.population = data["population"].to(device)
        if data["fitness"] is not None:
            state.fitness = data["fitness"].to(device)
        if data["best_solution"] is not None:
            state.best_solution = data["best_solution"].to(device)
        
        state.hyperparams = data.get("hyperparams", {})
        state.extra = data.get("extra", {})
        
        return state


# =============================================================================
# Abstract Algorithm Base Class
# =============================================================================

class Algorithm(nn.Module, ABC):
    """
    Abstract base class for all EvoGrad optimisation algorithms.
    
    This class provides the common infrastructure for population-based
    optimisers following the pymoo-style dependency injection pattern.
    Operators (selection, crossover, mutation, etc.) are passed to the
    constructor and used during the optimisation loop.
    
    Subclasses must implement:
        - _infill(): Generate offspring population
        - _advance(): Update state based on offspring evaluation
    
    Optionally override:
        - _setup(): One-time setup after initialisation
        - _get_hyperparams(): Return current hyperparameter values
    
    Args:
        pop_size: Population size.
        sampling: Operator for initial population generation.
        selection: Parent selection operator (optional, algorithm-specific).
        crossover: Crossover/recombination operator (optional).
        mutation: Mutation operator (optional).
        survival: Survivor selection operator (optional).
        repair: Repair operator for constraint handling (optional).
        eliminate_duplicates: Duplicate handling strategy:
            - True: Use default epsilon-based elimination
            - False: No duplicate elimination
            - DuplicateEliminator instance: Custom eliminator
        n_offsprings: Number of offspring per generation (default: pop_size).
        differentiable: Enable gradient flow through operations.
        dtype: Tensor dtype (default: torch.float32). Use ``torch.float64``
            when the objective requires higher numerical precision (e.g.,
            parameter estimation with stiff ODE solvers). The dtype should
            match the Problem's dtype to avoid silent precision loss in
            operator computations and log/exp hyperparameter transforms.

    Attributes:
        pop_size: Population size.
        n_offsprings: Number of offspring per generation.
        differentiable: Whether gradients are enabled.
        dtype: Tensor data type.
        problem: The Problem instance (set after initialize()).
        state: AlgorithmState containing current optimisation state.
    
    Example:
        >>> ga = GA(
        ...     pop_size=100,
        ...     sampling=FloatRandomSampling(),
        ...     selection=TournamentSelection(tournament_size=3),
        ...     crossover=SBXCrossover(eta=15, prob=0.9),
        ...     mutation=PolynomialMutation(eta=20),
        ...     repair=BoundsRepair(method='reflect'),
        ...     eliminate_duplicates=True,
        ...     differentiable=True,
        ... )
        >>> 
        >>> problem = Problem(objective=ackley, n_var=30, xl=-100, xu=100)
        >>> ga.initialize(problem)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        sampling: Optional[nn.Module] = None,
        selection: Optional[nn.Module] = None,
        crossover: Optional[nn.Module] = None,
        mutation: Optional[nn.Module] = None,
        survival: Optional[nn.Module] = None,
        repair: Optional[nn.Module] = None,
        eliminate_duplicates: Union[bool, DuplicateEliminator] = True,
        n_offsprings: Optional[int] = None,
        differentiable: bool = True,
        adaptive: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Validate inputs
        if pop_size < 1:
            raise ValueError(f"pop_size must be >= 1, got {pop_size}")

        # Device and dtype
        self.dtype = dtype
        
        # Population parameters
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings if n_offsprings is not None else pop_size
        self.differentiable = differentiable
        self.adaptive = adaptive
        
        # Store operators (can be None for algorithms that don't use them)
        if sampling is None:
            sampling = UniformSampling()
        self.sampling = sampling
        
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival
        self.repair = repair
        
        # Register operators as submodules if they are nn.Module
        # This ensures their parameters are included in algorithm.parameters()
        self._register_operator("sampling", sampling)
        self._register_operator("selection", selection)
        self._register_operator("crossover", crossover)
        self._register_operator("mutation", mutation)
        self._register_operator("survival", survival)
        self._register_operator("repair", repair)
        
        # Set up duplicate elimination
        if eliminate_duplicates is True:
            self.duplicate_eliminator = DuplicateEliminator(
                method=DuplicateMethod.EPSILON_L2,
                epsilon=1e-8,
            )
        elif eliminate_duplicates is False:
            self.duplicate_eliminator = None
        else:
            self.duplicate_eliminator = eliminate_duplicates
        
        # Problem reference (set in initialize())
        self.problem: Optional[Problem] = None
        
        # Algorithm state
        self.state = AlgorithmState()
        
        # Internal flags
        self._is_initialized = False
    
    def _register_operator(self, name: str, operator: Optional[nn.Module]) -> None:
        """Register an operator as a submodule if it's an nn.Module."""
        if operator is not None and isinstance(operator, nn.Module):
            self.add_module(f"_op_{name}", operator)
    
    # =========================================================================
    # Core Interface (to be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _infill(self) -> Tensor:
        """
        Generate offspring population.
        
        This method implements the core evolutionary operators:
        selection, crossover, mutation, etc. Subclasses use the
        operators passed to __init__ as needed.
        
        Returns:
            Offspring population tensor of shape (n_offsprings, n_var).
        """
        pass
    
    @abstractmethod
    def _advance(self, offspring: Tensor, offspring_fitness: Tensor) -> None:
        """
        Update algorithm state based on offspring evaluation.
        
        This method implements survivor selection and state updates.
        Should update self.state.population, self.state.fitness, and
        call self.state.update_best().
        
        Args:
            offspring: Offspring population tensor.
            offspring_fitness: Fitness values of offspring.
        """
        pass
    
    # =========================================================================
    # Optional Hooks (can be overridden)
    # =========================================================================
    
    def _setup_pop_size(self) -> None:
        """
        Resolve the population size before the initial population is sampled.

        Override for algorithms that derive pop_size from the problem
        (e.g. CMA-ES default lambda = 4 + floor(3*ln(n))). Called near the
        start of initialize(), after the problem is attached but before
        sampling, so the population is allocated at the final size.
        """
        pass

    def _setup(self) -> None:
        """
        One-time setup after initialisation.

        Override to perform algorithm-specific setup that requires
        the problem and population to be initialized.
        Called at the end of initialize().
        """
        pass
    
    def _get_hyperparams(self) -> Dict[str, Any]:
        """
        Return current hyperparameter values.
        
        Override to include algorithm-specific hyperparameters.
        These are passed to callbacks and stored in history.
        
        Returns:
            Dictionary of hyperparameter names to values.
        """
        return {}
    
    # =========================================================================
    # Initialisation
    # =========================================================================
    
    def initialize(self, problem: Problem) -> "Algorithm":
        """
        Initialize the algorithm with a problem.
        
        Creates initial population, evaluates fitness, and sets up
        internal state. Must be called before step() or forward().
        
        Args:
            problem: Problem instance defining objective, bounds, etc.
        
        Returns:
            Self for method chaining.
        """
        if self._is_initialized:
            return self
        
        # Store problem reference
        self.problem = problem
        self.device = problem.device
        
        # Move problem bounds to device
        self.register_buffer(
            "xl",
            problem.xl.to(device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "xu",
            problem.xu.to(device=self.device, dtype=self.dtype)
        )

        # Resolve final population size (e.g. CMA-ES auto lambda) before sampling
        self._setup_pop_size()

        # Create initial population using sampling operator
        population = self.sampling(self.pop_size, problem)
        
        # Apply repair if provided
        if self.repair is not None:
            population = self.repair(population, self.xl, self.xu)
        
        # Eliminate duplicates
        if self.duplicate_eliminator is not None:
            population = self.duplicate_eliminator(population, self.xl, self.xu)
        
        # Store population
        if self.differentiable:
            self._population = nn.Parameter(population)
        else:
            self.register_buffer("_population", population)
        
        # Evaluate initial population
        fitness = self._evaluate(population)
        
        # Initialize state
        # Always reference the registered _population (nn.Parameter or buffer)
        # to ensure gradient flow in differentiable mode and a single source of truth.
        self.state.population = self._population
        self.state.fitness = fitness
        self.state.generation = 0
        self.state.n_evals = self.pop_size
        self.state.update_best(self._population, fitness)
        
        # Algorithm-specific setup
        self._setup()
        
        self._is_initialized = True
        return self
    
    def reset(self, seed: Optional[int] = None) -> "Algorithm":
        """
        Reset algorithm to initial state.
        
        Args:
            seed: Optional new random seed.
        
        Returns:
            Self for method chaining.
        
        Raises:
            RuntimeError: If problem not set (never initialized).
        """
        if self.problem is None:
            raise RuntimeError(
                "Cannot reset: algorithm was never initialized with a problem."
            )
        
        if seed is not None:
            set_seed(seed)
            self._seed = seed
        
        self._is_initialized = False
        self.state = AlgorithmState()
        
        return self.initialize(self.problem)
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def _evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate fitness of population.
        
        Args:
            x: Population tensor of shape (N, n_var).
        
        Returns:
            Fitness tensor of shape (N,).
        """
        fitness = self.problem.evaluate(x)
        
        # Ensure correct shape and type
        if fitness.dim() == 0:
            fitness = fitness.unsqueeze(0)
        
        return fitness.to(device=self.device, dtype=self.dtype)
    
    # =========================================================================
    # Main Evolution Methods
    # =========================================================================
    
    def step(self) -> float:
        """
        Perform one generation of evolution (classical mode).
        
        This is the main entry point for advancing the algorithm
        without gradient tracking. Handles the complete cycle:
        infill -> repair -> eliminate_duplicates -> evaluate -> advance.
        
        Returns:
            Best fitness after this generation.
        
        Raises:
            RuntimeError: If algorithm not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize(problem) first."
            )
        
        # Generate offspring
        offspring = self._infill()
        
        # Apply repair if provided
        if self.repair is not None:
            offspring = self.repair(offspring, self.xl, self.xu)
        
        # Eliminate duplicates
        if self.duplicate_eliminator is not None:
            offspring = self.duplicate_eliminator(offspring, self.xl, self.xu)
        
        # Evaluate offspring
        offspring_fitness = self._evaluate(offspring)
        self.state.n_evals += offspring.shape[0]
        
        # Update state (implemented by subclass)
        self._advance(offspring, offspring_fitness)
        self.state.generation += 1
        
        # Update hyperparams in state
        self.state.hyperparams = self._get_hyperparams()
        
        return self.state.best_fitness
    
    def forward(self) -> Tensor:
        """
        PyTorch forward pass for differentiable optimisation.
        
        In differentiable mode, this builds a computation graph
        through the entire generation, returning the best fitness
        as a differentiable scalar loss. Call update_state() after
        loss.backward() and optimizer.step() to commit changes.
        
        Returns:
            Best fitness as a scalar tensor (for backprop).
        
        Raises:
            RuntimeError: If algorithm not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Algorithm not initialized. Call initialize(problem) first."
            )
        
        # Generate offspring (differentiable)
        offspring = self._infill()
        
        # Apply repair if provided (should be differentiable)
        if self.repair is not None:
            offspring = self.repair(offspring, self.xl, self.xu)
        
        # Note: duplicate elimination is typically not differentiable
        # Skip in forward pass, apply in update_state if needed
        
        # Evaluate offspring (differentiable if objective supports it)
        offspring_fitness = self._evaluate(offspring)
        self.state.n_evals += offspring.shape[0]
        
        # Store for update_state() to commit later
        self._pending_offspring = offspring
        self._pending_fitness = offspring_fitness
        
        # Return best fitness as loss
        return offspring_fitness.min()
    
    @torch.no_grad()
    def update_state(self) -> None:
        """
        Commit pending changes after backward pass.
        
        In differentiable mode, call this after loss.backward()
        and optimizer.step() to update the algorithm state.
        """
        if not hasattr(self, "_pending_offspring"):
            return
        
        offspring = self._pending_offspring
        offspring_fitness = self._pending_fitness
        
        # Now apply duplicate elimination (non-differentiable)
        if self.duplicate_eliminator is not None:
            # Duplicate elimination is non-differentiable. We can avoid
            # a full re-evaluation by only re-evaluating individuals that
            # were actually resampled.
            offspring, changed_indices = self.duplicate_eliminator(
                offspring, self.xl, self.xu, return_indices=True
            )

            if changed_indices.numel() > 0:
                # Ensure we can assign into the fitness tensor safely.
                offspring_fitness = offspring_fitness.clone()

                changed_fitness = self._evaluate(offspring[changed_indices])
                offspring_fitness[changed_indices] = changed_fitness
                self.state.n_evals += int(changed_indices.numel())
        
        # Advance with pending offspring
        self._advance(offspring, offspring_fitness)
        self.state.generation += 1
        self.state.hyperparams = self._get_hyperparams()
        
        # Clean up
        del self._pending_offspring
        del self._pending_fitness
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def population(self) -> Tensor:
        """Current population tensor."""
        return self._population
    
    @property
    def fitness(self) -> Optional[Tensor]:
        """Current fitness values."""
        return self.state.fitness
    
    @property
    def best_fitness(self) -> float:
        """Best fitness found so far."""
        return self.state.best_fitness
    
    @property
    def best_solution(self) -> Optional[Tensor]:
        """Best solution found so far."""
        return self.state.best_solution
    
    @property
    def n_evals(self) -> int:
        """Total number of fitness evaluations."""
        return self.state.n_evals
    
    @property
    def generation(self) -> int:
        """Current generation number."""
        return self.state.generation
    
    @property
    def n_var(self) -> Optional[int]:
        """Number of variables (from problem)."""
        return self.problem.n_var if self.problem is not None else None
    
    # =========================================================================
    # Serialisation
    # =========================================================================
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get complete state dictionary for checkpointing.
        
        Returns:
            Dictionary containing all state for serialisation.
        """
        state = {
            "algorithm_state": self.state.to_dict(),
            "model_state": super().state_dict(),
            "config": {
                "pop_size": self.pop_size,
                "n_offsprings": self.n_offsprings,
                "differentiable": self.differentiable,
                "adaptive": self.adaptive,
            },
            "is_initialized": self._is_initialized,
        }
        return state
    
    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Load state from dictionary.
        
        Note: The problem must be set separately via initialize() 
        or by setting self.problem before calling this method.
        
        Args:
            state_dict: State dictionary from state_dict().
            strict: Whether to require exact key matching.
        """
        # Load model parameters (population, operator params, etc.)
        super().load_state_dict(state_dict["model_state"], strict=strict)
        
        # Load algorithm state
        self.state = AlgorithmState.from_dict(
            state_dict["algorithm_state"], self.device
        )
        
        self._is_initialized = state_dict.get("is_initialized", True)
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        mode = "differentiable" if self.differentiable else "classical"
        status = "initialized" if self._is_initialized else "not initialized"
        n_var = self.n_var if self.problem else "?"
        # device is only assigned in initialize(); fall back before then.
        device = getattr(self, "device", "?")
        return (
            f"{self.__class__.__name__}("
            f"pop_size={self.pop_size}, "
            f"n_var={n_var}, "
            f"mode={mode}, "
            f"status={status}, "
            f"device={device})"
        )
    
    def summary(self) -> str:
        """Return a detailed summary of the algorithm configuration."""
        lines = [
            f"{'=' * 60}",
            f"Algorithm: {self.__class__.__name__}",
            f"{'=' * 60}",
            f"  Population size: {self.pop_size}",
            f"  Offspring size: {self.n_offsprings}",
            f"  Mode: {'Differentiable' if self.differentiable else 'Classical'}",
            f"  Adaptive: {self.adaptive}",
            f"  Device: {self.device}",
            f"  Initialized: {self._is_initialized}",
        ]
        
        # Problem info
        if self.problem is not None:
            lines.extend([
                f"",
                f"Problem:",
                f"  Variables: {self.problem.n_var}",
                f"  Bounds: [{float(self.xl.min()):.2g}, {float(self.xu.max()):.2g}]",
            ])
        
        # Operators
        lines.append(f"")
        lines.append("Operators:")
        operators = [
            ("Sampling", self.sampling),
            ("Selection", self.selection),
            ("Crossover", self.crossover),
            ("Mutation", self.mutation),
            ("Survival", self.survival),
            ("Repair", self.repair),
        ]
        for name, op in operators:
            if op is not None:
                lines.append(f"  {name}: {op.__class__.__name__}")
            else:
                lines.append(f"  {name}: None")
        
        # Duplicate elimination
        if self.duplicate_eliminator is not None:
            lines.append(f"  Duplicates: {self.duplicate_eliminator.method.name}")
        else:
            lines.append(f"  Duplicates: Disabled")
        
        # State info
        if self._is_initialized:
            lines.extend([
                f"",
                f"State:",
                f"  Generation: {self.state.generation}",
                f"  Evaluations: {self.state.n_evals}",
                f"  Best fitness: {self.state.best_fitness:.6g}",
            ])
        
        # Hyperparameters
        hp = self._get_hyperparams()
        if hp:
            lines.append(f"")
            lines.append("Hyperparameters:")
            for name, value in hp.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean()) if value.numel() > 1 else float(value)
                if isinstance(value, float):
                    lines.append(f"  {name}: {value:.4g}")
                else:
                    lines.append(f"  {name}: {value}")
        
        # Parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.extend([
            f"",
            f"Parameters:",
            f"  Total: {n_params:,}",
            f"  Trainable: {n_trainable:,}",
            f"{'=' * 60}",
        ])
        
        return "\n".join(lines)