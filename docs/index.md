# EvoGrad

**EvoGrad** is a PyTorch-based framework for **differentiable** Evolutionary Computation
and Swarm Intelligence. It bridges classical population-based optimisation with modern
differentiable programming by enabling gradient flow through evolutionary operators.

- **Fully differentiable** operators (Gumbel-Softmax selection, Binary-Concrete /
  straight-through masks, pathwise Gaussian sampling).
- **GPU/MPS accelerated** — native PyTorch, runs on CPU/CUDA/MPS.
- **Four algorithms** — GA, DE, PSO, CMA-ES (plus SHADE/L-SHADE), each with classical,
  *adaptive*, *differentiable*, and *full* modes.
- **Learnable hyperparameters** tuned via backpropagation.

```bash
pip install evograd-diff   # import name: evograd
```

EvoGrad was accepted at the **IEEE Congress on Evolutionary Computation (CEC) 2026**
(see [Citation](citation.md)).

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
guide
benchmarks
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
citation
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
