# Benchmarks

EvoGrad ships a PyTorch-native benchmark suite (`evograd.benchmarks`) and a parallel runner.

## Function library

- **Classical** — Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel, Levy, … (unimodal and multimodal).
- **CEC 2017** — the full `F1`–`F30` competition suite, rewritten in PyTorch.
- **Multi-Basin / Smoothed-Funnel** — `MultiBasinRastrigin`, `MultiBasinRosenbrock`, `DeceptiveLandscape`,
  designed for differentiable EAs (log-sum-exp aggregation of multiple basins).
- **Transforms** — shifted / rotated / scaled wrappers for building variants.

```python
import torch
from evograd.benchmarks.functions import get_cec2017_function, MultiBasinRastrigin

f = get_cec2017_function(14, n_var=30)   # CEC 2017 F14 in 30D
y = f(torch.randn(100, 30))              # batch evaluation -> shape [100]
```

## Running

```bash
# 30 runs of DE on the full CEC 2017 suite in 30D (vs pymoo + Adam baselines)
python -m evograd.benchmarks.run_benchmark_functions -a DE -s cec2017 -D 30 -r 30

# List every available function and suite
python -m evograd.benchmarks.run_benchmark_functions --list_functions
```

See the project README for reported results (CEC 2017 and Multi-Basin Rastrigin).
