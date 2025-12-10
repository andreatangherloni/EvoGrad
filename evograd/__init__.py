"""
EvoGrad: Metaheuristics in a Differentiable Wonderland.

EvoGrad is a PyTorch-based framework for differentiable evolutionary
computation and swarm intelligence. It provides:

- **Differentiable Algorithms**: GA, DE, PSO, CMA-ES with gradient flow
- **Classical Algorithms**: GPU-accelerated versions without backpropagation
- **Modular Operators**: Pluggable selection, crossover, mutation operators
- **Flexible API**: pymoo-inspired interface with dependency injection

Quick Start
-----------
>>> from evograd.core.problem import Problem
>>> from evograd.core.minimize import minimize
>>> from evograd.core.termination import MaxEvaluations
>>> from evograd.algorithms import GA
>>> from evograd.operators import TournamentSelection, SBX, PolynomialMutation
>>>
>>> # Define a problem
>>> problem = Problem(
...     objective=lambda x: (x ** 2).sum(dim=-1),  # Sphere function
...     n_var=10,
...     xl=-5.0,
...     xu=5.0,
... )
>>>
>>> # Create algorithm with custom operators
>>> algorithm = GA(
...     pop_size=100,
...     selection=TournamentSelection(tournament_size=3),
...     crossover=SBX(eta=15, prob=0.9),
...     mutation=PolynomialMutation(eta=20),
...     differentiable=True,  # Enable gradient-based hyperparameter learning
... )
>>>
>>> # Run optimization
>>> result = minimize(problem, algorithm, termination=MaxEvaluations(10000), seed=42)
>>> print(f"Best fitness: {result.best_fitness:.6f}")

Architecture
------------
evograd/
├── core/           # Problem, Algorithm base, Result, Termination, minimize/maximize
├── algorithms/     # GA, DE, PSO, CMA-ES implementations
├── operators/      # Selection, Crossover, Mutation, Repair
├── benchmarks/     # Classic functions, CEC2017
└── utils/          # Device, Callbacks, Duplicates

License
-------
MIT License - See LICENSE file for details.

Authors
-------
Andrea Tangherloni <andrea.tangherloni@unibocconi.it>
"""

__version__ = "0.1.0"
__author__ = "Andrea Tangherloni"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    ]