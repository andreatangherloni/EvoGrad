# EvoGrad: Metaheuristics in a Differentiable Wonderland

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg" alt="License: Apache-2.0">
 
</p>

**EvoGrad** is a PyTorch-based framework for differentiable Evolutionary Computation and Swarm Intelligence. It bridges classical population-based optimisation with modern differentiable programming by enabling gradient flow through evolutionary operators.

## 🌟 Key Features

- **Fully Differentiable**: All operators support gradient computation via reparameterisation tricks (Gumbel-Softmax, Binary-Concrete, pathwise gradients)
- **GPU Accelerated**: Native PyTorch implementation for seamless CPU/GPU/MPS execution
- **Modular Design**: Dependency injection pattern inspired by [pymoo](https://pymoo.org/) for flexible operator composition
- **Learnable Hyperparameters**: Automatically tune algorithm parameters via backpropagation
- **Four Algorithms**: GA, DE, PSO, and CMA-ES with multiple variants

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/evograd.git
cd evograd

# Install dependencies
pip install torch numpy

# Install in development mode
pip install -e .
```

## 🚀 Quick Start

```python
import torch
from evograd.core import Problem, minimize
from evograd.algorithms import GA, DE, PSO, CMAES

# Define an optimisation problem
problem = Problem(
    objective=lambda x: (x**2).sum(dim=-1),  # Sphere function
    n_var=30,
    xl=-100.0,
    xu=100.0,
)

# Run with Genetic Algorithm
ga = GA(pop_size=100, differentiable=True)
result = minimize(problem, ga, max_evals=10000, seed=42)
print(f"GA Best: {result.best_fitness:.6f}")

# Run with Differential Evolution
de = DE(pop_size=100, variant="DE/rand/1/bin", adaptive=True)
result = minimize(problem, de, max_evals=10000, seed=42)
print(f"DE Best: {result.best_fitness:.6f}")

# Run with Particle Swarm Optimisation
pso = PSO(pop_size=100, adaptive=True, differentiable=True)
result = minimize(problem, pso, max_evals=10000, seed=42)
print(f"PSO Best: {result.best_fitness:.6f}")

# Run with CMA-ES
cmaes = CMAES(sigma=0.5, adaptive=True)
result = minimize(problem, cmaes, max_evals=10000, seed=42)
print(f"CMA-ES Best: {result.best_fitness:.6f}")
```

## 🔧 Algorithms and Operating Modes

### Genetic Algorithm (GA)

The GA uses **operator-level differentiability**. Each operator (selection, crossover, mutation, survival) can independently be set to differentiable mode:

```python
from evograd.algorithms import GA
from evograd.operators import (
    RouletteSelection,
    SBXCrossover,
    PolynomialMutation,
    MergeSurvival,
)

# Classical GA (no gradients)
ga = GA(pop_size=100, differentiable=False)

# Fully differentiable GA with custom operators
ga = GA(
    pop_size=100,
    selection=RouletteSelection(differentiable=True, learn_temperature=True),
    crossover=SBXCrossover(differentiable=True, learn_eta=True, learn_prob=True),
    mutation=PolynomialMutation(differentiable=True, learn_eta=True, learn_prob=True),
    survival=MergeSurvival(selection=RouletteSelection(differentiable=True)),
    differentiable=True,  # Makes population learnable
)
```

| Parameter | Effect |
|-----------|--------|
| `differentiable=False` | Classical GA with discrete operators |
| `differentiable=True` | Population is an `nn.Parameter` (learnable via backprop) |
| Operator `differentiable=True` | Operator uses Gumbel-Softmax/Binary-Concrete for gradient flow |
| Operator `learn_*=True` | Operator hyperparameters become learnable `nn.Parameter` |

### Differential Evolution (DE)

DE uses **algorithm-level flags** for adaptive hyperparameters and differentiable population:

```python
from evograd.algorithms import DE, de_rand_1_bin, de_best_1_bin

# Classical DE
de = DE(pop_size=100, variant="DE/rand/1/bin", F=0.5, CR=0.9)

# Adaptive DE (learnable F, CR, selection temperature)
de = DE(pop_size=100, variant="DE/best/1/bin", adaptive=True)

# Differentiable population
de = DE(pop_size=100, variant="DE/rand/1/bin", differentiable=True)

# Both adaptive and differentiable
de = DE(pop_size=100, variant="DE/current-to-best/1/bin", adaptive=True, differentiable=True)
```

| `adaptive` | `differentiable` | Effect |
|------------|------------------|--------|
| False | False | Classical DE |
| True | False | F, CR, temperatures learnable via backprop |
| False | True | Population learnable via backprop |
| True | True | Both hyperparameters and population learnable |

**Supported Variants:**
- `DE/rand/1/bin`, `DE/rand/1/exp`, `DE/rand/2/bin`, `DE/rand/2/exp`
- `DE/best/1/bin`, `DE/best/1/exp`, `DE/best/2/bin`, `DE/best/2/exp`
- `DE/current-to-best/1/bin`, `DE/current-to-best/1/exp`
- `DE/current-to-rand/1`

### Particle Swarm Optimisation (PSO)

PSO uses the same **algorithm-level flags** as DE:

```python
from evograd.algorithms import PSO, pso_constriction, pso_adaptive

# Classical PSO
pso = PSO(pop_size=100, inertia=0.7, c1=1.5, c2=1.5)

# Adaptive PSO (learnable inertia, c1, c2)
pso = PSO(pop_size=100, adaptive=True)

# Per-particle adaptive coefficients
pso = PSO(pop_size=100, adaptive=True, per_particle_coeffs=True)

# Constriction factor PSO
pso = pso_constriction(pop_size=100)

# Fully differentiable
pso = PSO(pop_size=100, adaptive=True, differentiable=True)
```

| `adaptive` | `differentiable` | Effect |
|------------|------------------|--------|
| False | False | Classical PSO |
| True | False | Inertia, c1, c2 learnable via backprop |
| False | True | Particle positions learnable via backprop |
| True | True | Both coefficients and positions learnable |

### CMA-ES

CMA-ES supports **adaptive coefficients** and **restart strategies** (IPOP/BIPOP):

```python
from evograd.algorithms import CMAES, cmaes_ipop, cmaes_bipop

# Classical CMA-ES
cmaes = CMAES(pop_size=50, sigma=0.5)

# Adaptive CMA-ES (learnable cc, cs, c1, cmu, damps)
cmaes = CMAES(pop_size=50, sigma=0.5, adaptive=True)

# Differentiable mean
cmaes = CMAES(pop_size=50, sigma=0.5, differentiable=True)

# IPOP-CMA-ES (increasing population restarts)
cmaes = cmaes_ipop(restarts=9, incpopsize=2)

# BIPOP-CMA-ES (alternating small/large populations)
cmaes = cmaes_bipop(restarts=9)
```

| `adaptive` | `differentiable` | Effect |
|------------|------------------|--------|
| False | False | Classical CMA-ES |
| True | False | Adaptation coefficients learnable via backprop |
| False | True | Distribution mean μ learnable via backprop |
| True | True | Both coefficients and mean learnable |

**Restart Strategies:**
- **IPOP**: Restart with doubled population after convergence
- **BIPOP**: Alternate between small (focused) and large (exploratory) populations

## 📚 Operators Library

EvoGrad provides a comprehensive library of evolutionary operators:

### Selection
| Operator | Description | Differentiable |
|----------|-------------|----------------|
| `RandomSelection` | Uniform random selection | ✗ |
| `RouletteSelection` | Fitness-proportionate (Gumbel-Softmax) | ✓ |
| `TournamentSelection` | Tournament with soft winner (Gumbel-Softmax) | ✓ |
| `RankSelection` | Rank-based probabilities | ✓ |
| `StochasticUniversalSampling` | SUS with soft selection | ✓ |

### Crossover
| Operator | Description | Differentiable |
|----------|-------------|----------------|
| `SBXCrossover` | Simulated Binary Crossover | ✓ |
| `BinomialCrossover` | DE-style binomial | ✓ |
| `ExponentialCrossover` | DE-style exponential | ✓ |
| `BlendCrossover` | BLX-α crossover | ✓ |
| `ArithmeticCrossover` | Weighted average | ✓ |
| `UniformCrossover` | Gene-wise uniform swap | ✓ |
| `SimulatedBinaryCrossover` | Alias for SBX | ✓ |

### Mutation
| Operator | Description | Differentiable |
|----------|-------------|----------------|
| `PolynomialMutation` | Polynomial bounded mutation | ✓ |
| `GaussianMutation` | Additive Gaussian noise | ✓ |
| `UniformMutation` | Uniform random replacement | ✓ |
| `AdaptiveMutation` | Self-adaptive mutation rates | ✓ |

### Survival
| Operator | Description |
|----------|-------------|
| `MergeSurvival` | (μ+λ) with optional elitism |
| `ReplacementSurvival` | (μ,λ) generational replacement |
| `AgingSurvival` | Age-based replacement |
| `FitnessSurvival` | Pure fitness-based truncation |

### Repair
| Operator | Description |
|----------|-------------|
| `BoundsRepair` | Clamp to bounds |
| `ReflectionRepair` | Bounce off boundaries |
| `WrapRepair` | Toroidal wrap-around |
| `RandomRepair` | Random resampling |

## 🎯 Advanced Usage

### Training Neural Networks with EvoGrad

```python
import torch
import torch.nn as nn
from evograd.algorithms import CMAES
from evograd.core import Problem
from evograd.core.termination import MaxEvaluations


# Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.net(x)

# Flatten parameters for optimisation
model = MLP()
n_params = sum(p.numel() for p in model.parameters())

def loss_fn(params):
    # Reshape flat params back to model
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(params[idx:idx+numel].view(p.shape))
        idx += numel
    
    # Compute loss on dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    pred = model(x)
    return ((pred - y)**2).mean()

# Batch evaluation
def batch_loss(pop):
    return torch.stack([loss_fn(p) for p in pop])

problem = Problem(
    objective=batch_loss,
    n_var=n_params,
    xl=-10.0,
    xu=10.0,
)

cmaes = CMAES(pop_size=50, sigma=1.0, adaptive=True)
result = minimize(problem, cmaes, MaxEvaluations(10000))
print(f"Final loss: {result.best_fitness:.6f}")
```

### Callbacks for Logging

```python
from evograd.core import minimize
from evograd.utils import HistoryCallback, PrintCallback

callbacks = [
    PrintCallback(every=10),  # Print progress every 10 generations
    HistoryCallback(),        # Record full history
]

result = minimize(problem, algorithm, termination=MaxEvaluations(10000), callback=callbacks)

# Access history
print(f"Fitness over time: {result.history['best_fitness']}")
```

## 🏗️ Architecture

```
evograd/
├── algorithms/
│   └── cmaes.py          # CMA-ES with IPOP/BIPOP
│   ├── de.py             # Differential Evolution
│   ├── ga.py             # Genetic Algorithm
│   ├── pso.py            # Particle Swarm Optimisation
├── core/
│   ├── algorithm.py      # Base Algorithm class
│   ├── maximize.py       # Optimisation loop (maximisation)
│   ├── minimize.py       # Optimisation loop (minimisation)
│   ├── problem.py        # Problem definition
│   ├── result.py         # Result container
│   └── termination.py    # Stopping criteria
├── operators/
│   ├── crossover.py      # Crossover operators
│   ├── mutation.py       # Mutation operators
│   ├── sampling.py       # Sampling operators
│   ├── selection.py      # Selection operators
│   ├── survival.py       # Survival/replacement
│   └── repair.py         # Constraint handling
└── utils/
    ├── callbacks.py      # Logging utilities
    ├── device.py         # Device management
    └── duplicates.py     # Duplicate elimination
```

## 🔬 How It Works

EvoGrad makes evolutionary algorithms differentiable through:

1. **Reparameterisation Trick**: Convert random sampling into deterministic transformations of parameter-free noise:
   ```
   x = g_θ(ε), ε ~ p(ε)  →  ∇_θ L ≈ ∇_θ f(g_θ(ε))
   ```

2. **Gumbel-Softmax**: Differentiable approximation for categorical selection:
   ```python
   # Soft selection (differentiable)
   probs = softmax((log_probs + gumbel_noise) / temperature)
   selected = probs @ population  # Weighted combination
   ```

3. **Binary-Concrete**: Differentiable approximation for binary masks (mutation/crossover):
   ```python
   # Soft mask (differentiable)
   mask = sigmoid((log(u) - log(1-u) + logits) / temperature)
   # Straight-through estimator for hard decisions
   hard_mask = (mask > 0.5).float() - mask.detach() + mask
   ```

4. **Pathwise Gradients**: For continuous distributions (Gaussian sampling in CMA-ES):
   ```python
   # x = μ + σ * L @ z, z ~ N(0, I)
   z = torch.randn(pop_size, n_var)
   x = mean + sigma * (L @ z.T).T  # Fully differentiable
   ```

## 📊 Benchmarks

TODO

## 📖 Citation

TBA

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- Inspired by [pymoo](https://pymoo.org/) for API design
- Built with [PyTorch](https://pytorch.org/) for automatic differentiation