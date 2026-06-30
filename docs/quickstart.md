# Quickstart

Define a {class}`~evograd.core.problem.Problem`, pick an algorithm, and call
{func}`~evograd.core.minimize.minimize`.

```python
import torch
from evograd.core import Problem, minimize, MaxEvaluations
from evograd.algorithms import GA, DE, PSO, CMAES

# Define an optimisation problem (Sphere function)
problem = Problem(
    objective=lambda x: (x ** 2).sum(dim=-1),
    n_var=30,
    xl=-100.0,
    xu=100.0,
)

# Genetic Algorithm
ga = GA(pop_size=100, differentiable=True)
result = minimize(problem, ga, termination=MaxEvaluations(10000), seed=42)
print(f"GA best: {result.best_fitness:.6f}")

# Differential Evolution (adaptive hyperparameters)
de = DE(pop_size=100, variant="DE/rand/1/bin", adaptive=True)
result = minimize(problem, de, termination=MaxEvaluations(10000), seed=42)
print(f"DE best: {result.best_fitness:.6f}")

# CMA-ES
cmaes = CMAES(pop_size=50, sigma=0.5, adaptive=True)
result = minimize(problem, cmaes, termination=MaxEvaluations(10000), seed=42)
print(f"CMA-ES best: {result.best_fitness:.6f}")
```

The objective is any callable mapping a batch `(N, n_var)` to fitness `(N,)`; it may be
non-differentiable (classical mode) or differentiable (gradient-enabled modes).

See the [User guide](guide.md) for the operating modes and per-algorithm options.
