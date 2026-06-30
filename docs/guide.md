# User guide

## Operating modes

EvoGrad algorithms expose two independent flags that select four modes:

| `adaptive` | `differentiable` | Mode | Behaviour |
|:---:|:---:|---|---|
| `False` | `False` | **Classical** | Standard, gradient-free metaheuristic |
| `True` | `False` | **Adaptive** | Hyperparameters (F, CR, inertia, σ-coeffs, temperatures) are learnable `nn.Parameter`s, tuned by backprop |
| `False` | `True` | **Differentiable (Diff)** | The population is an `nn.Parameter`, refined by gradients of the objective |
| `True` | `True` | **Full** | Both of the above, co-adapted in one autodiff graph |

In differentiable mode, {func}`~evograd.core.minimize.minimize` builds a per-generation
computation graph and updates learnable tensors with a gradient optimiser, then commits
the evolutionary state. The scalar loss reduction is configurable
(`reduction="mean"` by default; also `"sum"`/`"min"`), and `live_selection=True` lets the
Gumbel-Softmax selection gradient reach the population.

## Differentiability mechanisms

- **Gumbel-Softmax** — differentiable categorical selection.
- **Binary-Concrete + straight-through** — differentiable masks for crossover/mutation.
- **Pathwise (reparameterisation) gradients** — Gaussian sampling (e.g. CMA-ES, Gaussian mutation).

## Algorithms

- **GA** — operator-level differentiability; compose {mod}`selection <evograd.operators.selection>`,
  {mod}`crossover <evograd.operators.crossover>`, {mod}`mutation <evograd.operators.mutation>`,
  and {mod}`survival <evograd.operators.survival>` operators.
- **DE** — variants like `DE/rand/1/bin`, `DE/best/1/bin`, `DE/current-to-best/1/bin`; learnable F, CR.
- **PSO** — learnable inertia and cognitive/social coefficients (optionally per-particle).
- **CMA-ES** — learnable adaptation coefficients and step size; IPOP/BIPOP restarts.
- **SHADE / L-SHADE** — success-history adaptive DE (reuses the binomial crossover operator).

```python
from evograd.algorithms import GA
from evograd.operators import RouletteSelection, SBXCrossover, PolynomialMutation, MergeSurvival

ga = GA(
    pop_size=100,
    selection=RouletteSelection(adaptive=True, learn_temperature=True),
    crossover=SBXCrossover(adaptive=True, learn_eta=True),
    mutation=PolynomialMutation(adaptive=True, learn_eta=True),
    survival=MergeSurvival(elitism=True, adaptive=True),
    differentiable=True,
)
```

See the [API reference](api.md) for the full set of classes and parameters.
