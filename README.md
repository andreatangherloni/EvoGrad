# EvoGrad: Accelerated Metaheuristics in a Differentiable Wonderland

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/IEEE%20CEC-2026-success.svg" alt="IEEE CEC 2026">
 
</p>

> 🎉 **EvoGrad has been accepted at [IEEE CEC 2026](#-citation)!**

**EvoGrad** is a PyTorch-based framework for differentiable Evolutionary Computation and Swarm Intelligence. It bridges classical population-based optimisation with modern differentiable programming by enabling gradient flow through evolutionary operators.

## 🌟 Key Features

- **Fully Differentiable**: All operators support gradient computation via reparameterisation tricks (Gumbel-Softmax, Binary-Concrete, pathwise gradients)
- **GPU Accelerated**: Native PyTorch implementation for seamless CPU/GPU/MPS execution
- **Modular Design**: Dependency injection pattern inspired by [pymoo](https://pymoo.org/) for flexible operator composition
- **Learnable Hyperparameters**: Automatically tune algorithm parameters via backpropagation
- **Four Algorithms**: GA, DE, PSO, and CMA-ES with multiple variants

## 📦 Installation

```bash
# From PyPI (the import name is `evograd`)
pip install evograd-diff
```

Or install directly from the repository:

```bash
pip install "git+https://github.com/andreatangherloni/EvoGrad.git"
```

For local development:

```bash
git clone https://github.com/andreatangherloni/EvoGrad.git
cd EvoGrad
pip install -e .
```

## 🚀 Quick Start

```python
import torch
from evograd.core import Problem, minimize, MaxEvaluations
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
result = minimize(problem, ga, termination=MaxEvaluations(10000), seed=42,
                  lr_pop=-1, lr_hyper=-1)
print(f"GA Best: {result.best_fitness:.6f}")

# Run with Differential Evolution
de = DE(pop_size=100, variant="DE/rand/1/bin", adaptive=True)
result = minimize(problem, de, termination=MaxEvaluations(10000), seed=42,
                  lr_hyper=-1)
print(f"DE Best: {result.best_fitness:.6f}")

# Run with Particle Swarm Optimisation
pso = PSO(pop_size=100, adaptive=True, differentiable=True)
result = minimize(problem, pso, termination=MaxEvaluations(10000), seed=42,
                  lr_pop=-1, lr_hyper=-1)
print(f"PSO Best: {result.best_fitness:.6f}")

# Run with CMA-ES
cmaes = CMAES(sigma=0.5, adaptive=True)
result = minimize(problem, cmaes, termination=MaxEvaluations(10000), seed=42,
                  lr_hyper=-1)
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
    selection=RouletteSelection(adaptive=True, learn_temperature=True),
    crossover=SBXCrossover(adaptive=True, learn_eta=True, learn_prob=True),
    mutation=PolynomialMutation(adaptive=True, learn_eta=True, learn_prob=True),
    survival=MergeSurvival(elitism=True, adaptive=True),
    differentiable=True,  # Makes population learnable
)
```

| Parameter | Effect |
|-----------|--------|
| `differentiable=False` | Classical GA with discrete operators |
| `differentiable=True` | Population is an `nn.Parameter` (learnable via backprop) |
| Operator `adaptive=True` | Operator uses Gumbel-Softmax/Binary-Concrete for gradient flow |
| Operator `learn_*=True` | Operator hyperparameters become learnable `nn.Parameter` |

Gradient optimizers are opt-in at the `minimize()` call. `lr_pop=None` and
`lr_hyper=None` (the defaults) perform no backpropagation; use `-1` to select
EvoGrad's algorithm-specific learning rate or pass a numeric value. For strict
classical behavior, also construct the algorithm with `differentiable=False`
and `adaptive=False`.

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
from evograd.algorithms import PSO, pso_constriction, pso_default

# Classical PSO
pso = PSO(pop_size=100, w=0.7, c1=1.5, c2=1.5)

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
| `NPointCrossover` | N-point crossover | ✓ |

### Mutation
| Operator | Description | Differentiable |
|----------|-------------|----------------|
| `PolynomialMutation` | Polynomial bounded mutation | ✓ |
| `GaussianMutation` | Additive Gaussian noise | ✓ |
| `UniformMutation` | Uniform random replacement | ✓ |
| `NonUniformMutation` | Annealed mutation strength | ✓ |

### Survival
| Operator | Description |
|----------|-------------|
| `MergeSurvival` | (μ+λ) with optional elitism |
| `CommaSurvival` | (μ,λ) generational replacement |
| `ReplaceWorstSurvival` | Steady-state worst replacement |
| `AgeSurvival` | Age-based replacement |
| `FitnessSurvival` | Pure fitness-based truncation |

### Repair
| Operator | Description |
|----------|-------------|
| `BoundsRepair` | Clamp to bounds |
| `ReflectRepair` | Bounce off boundaries |
| `WrapRepair` | Toroidal wrap-around |
| `RandomRepair` | Random resampling |

## 🎯 Advanced Usage

### Training Neural Networks with EvoGrad

```python
import torch
import torch.nn as nn
from evograd.algorithms import CMAES
from evograd.core import Problem, minimize
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
from evograd.core import minimize, MaxEvaluations
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

EvoGrad ships a self-contained, **PyTorch-native benchmark suite** (`evograd.benchmarks`) together with a parallel runner that evaluates every algorithm in its four operating modes against two reference baselines.

### Function library

All functions share a common `BenchmarkFunction` interface (`f(x)` on an `(N, n_var)` batch, plus `.bounds` and the known optimum) and run on CPU/GPU/MPS.

| Category | Functions |
|----------|-----------|
| **Classical — unimodal** | Sphere, Ellipsoid, SumOfDifferentPowers, Schwefel 2.22, Cigar, Discus, BentCigar, Rosenbrock, DixonPrice, Powell, Trid |
| **Classical — multimodal** | Rastrigin, Ackley, Griewank, Schwefel, Levy, Michalewicz, Zakharov, Weierstrass, Alpine, Salomon, Styblinski–Tang |
| **CEC 2017** (`F1`–`F30`) | Simple/unimodal (F1–F10), Hybrid (F11–F20), Composition (F21–F30) — the full competition suite, **rewritten from scratch in PyTorch** |
| **Multi-Basin / Smoothed-Funnel** | `MultiBasinRastrigin`, `MultiBasinRosenbrock`, `DeceptiveLandscape` — designed for differentiable EAs |
| **Transforms** | Shifted / Rotated / Scaled / Asymmetric / Oscillated / Biased wrappers for building custom variants |

```python
import torch
from evograd.benchmarks.functions import Sphere, Rastrigin, get_cec2017_function, MultiBasinRastrigin

f = get_cec2017_function(14, n_var=30)   # CEC 2017 F14 in 30D
y = f(torch.randn(100, 30))              # batch evaluation -> shape [100]
```

The **Multi-Basin** functions aggregate `K` basins (each a full Rastrigin/Rosenbrock landscape) with a smooth *log-sum-exp* minimum, so the surface stays differentiable everywhere while still trapping pure gradient descent in distractor basins — exactly the setting where population search combined with gradient refinement pays off.

### Running the benchmarks

The runner evaluates the four EvoGrad modes — **Classical**, **Differentiable**, **Adaptive**, **Full** — and, by default, the **pymoo** and **Adam** (multi-start) baselines:

```bash
# Install the manuscript's pinned cross-library baseline
pip install -r requirements-benchmarks.txt

# 30 runs of DE on the full CEC 2017 suite in 30D (vs pymoo + Adam)
python -m evograd.benchmarks.run_benchmark_functions -a DE -s cec2017 -D 30 -r 30

# CMA-ES on the multi-basin functions, on GPU
python -m evograd.benchmarks.run_benchmark_functions -a CMAES -s funnel -D 30 --device cuda

# List every available function and suite
python -m evograd.benchmarks.run_benchmark_functions --list_functions
```

Key flags: `-a {DE,SHADE,PSO,GA,CMAES,ADAM}`, `-s` suite (`classical`, `standard`, `cec2017[_simple|_hybrid|_composition]`, `funnel`, …), `-D` dimensionality, `-r` runs, `-p` population size, `--no_pymoo` / `--no_adam` to drop baselines. Plotting utilities live in `plot_benchmarks.py`.

### Results

The three differentiable variants are compared against the **Classical** baseline and pymoo:

- **Adaptive** — learnable hyperparameters, purely stochastic variation (no gradient through the population).
- **Diff** (Differentiable) — fixed hyperparameters, gradients refine the population.
- **Full** — both: learnable hyperparameters *and* gradient-based population refinement.

**CEC 2017 (30D & 100D).** 29 functions (F2 excluded, per the competition), search space `[-100, 100]^D`, 100 individuals, `10000·D` evaluations, 30 independent paired runs, one-sided Wilcoxon signed-rank test with Benjamini–Hochberg correction. Highlights:

- Differentiable variants are **statistically significantly better than the classical baseline in ~31% of all comparisons**, and **never substantially worse** — gradient refinement can be added to EAs safely.
- Gains concentrate where local refinement helps most: **GA (70.1%)** and **DE (46.0%)** of comparisons improved, versus **PSO (6.9%)** and **CMA-ES (1.1%)**, which already include strong built-in adaptation.
- Across variants, **Full (41.4%) > Adaptive (35.3%) > Diff (16.4%)** — combining hyperparameter learning with population refinement helps the most, increasingly so at 100D.
- CMA-ES is the strongest method overall (especially on hybrid/composition functions), and EvoGrad runs ~**3× faster** than the pymoo baselines on CPU *despite* the added gradient computation.

**Multi-Basin Rastrigin** (`D=30`, bounds `[-5, 5]^D`, 150,000 evaluations, 30 runs). Every CMA-ES variant locates the global basin (best fitness `0.00`); a multi-start **Adam** baseline (100 parallel solutions) stays trapped in distractor basins:

| Configuration | Best | Mean | Std | Time (s) |
|---|---|---|---|---|
| CMA-ES Classical | 0.00 | 2.22 | 3.04 | 25.66 |
| CMA-ES Differentiable | 0.00 | 1.49 | 2.16 | 9.77 |
| CMA-ES Adaptive | 0.00 | **0.99** | **1.36** | 45.24 |
| CMA-ES Full | 0.00 | 1.29 | 2.12 | **7.94** |
| Adam (multi-start, pop-based) | 116.41 | 153.77 | 13.98 | 3.88 |

The **Adaptive** variant reaches the lowest mean/variance, while **Full** matches it closely at the **fastest** runtime — gradient flow yields large speed-ups while population search secures the global basin. Adam alone is **>2 orders of magnitude worse**, confirming that pure gradient descent cannot escape distractor basins.

> Full experimental details are in the paper (see [Citation](#-citation)).

## 📖 Citation

EvoGrad was accepted at the **IEEE Congress on Evolutionary Computation (CEC) 2026**. If you use EvoGrad in your research, please cite:

```bibtex
@inproceedings{citterio2026evograd,
  title     = {{EvoGrad}: Accelerated Metaheuristics in a Differentiable Wonderland},
  author    = {Citterio, Beatrice F. R. and Papetti, Daniele M. and Dimitri, Giovanna Maria and Tangherloni, Andrea},
  booktitle = {Proceedings of the IEEE Congress on Evolutionary Computation (CEC)},
  year      = {2026},
}
```

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
