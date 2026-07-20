# Changelog

## 0.3.0

### Fixed

- Keep `best_solution` paired with the coordinates at which `best_fitness` was evaluated.
- Keep the differentiable PSO population consistent with its fitness: the committed position (`P₀ + v − lr·∇`) is now re-evaluated so `population`, `fitness`, and the personal/global bests all agree (previously `fitness` was that of the pre-gradient position `P₀ + v`). Benchmarks show no change in optimisation quality; the extra objective pass is uncounted in `n_evals`.
- Avoid autograd/backward work and stale gradient accumulation when both learning rates are disabled.
- Enforce declared constraints through the configurable `Problem.constraint_penalty` and report final feasibility.
- Translate nested `TargetReached` criteria correctly in `maximize()`.
- Keep L-SHADE population sizes, offspring counts, and differentiable optimizer parameters synchronized during reduction.
- Use bounded Deb polynomial mutation and preserve fixed variables in reflect/wrap repair.
- Restore gradient-carrying adaptive Exponential, Uniform, and N-point crossover paths.
- N-point crossover now samples distinct cut points (previously allowed duplicates that could cancel to no crossover), supports `n_var == 1`, and raises when `n_points > n_var - 1`. This changes classical N-point outputs for a given seed (N-point is unused by the benchmarked algorithms).
- Reject ambiguous one-dimensional per-gene/per-individual parameters when `N == D > 1`; SHADE now passes explicit `[N, 1]` CR values. The unambiguous `N == D == 1` case is accepted.
- Construct all benchmark transform wrappers safely.
- Remove CEC composition-center NaNs, restore the cyclic Expanded Schaffer F6 term, and expose the true CEC F9 optimum location while retaining official F8/F9 numerics.
- Use scale-invariant softmax recombination in differentiable CMA-ES and honor `cmaes_large(pop_size_factor=...)`.
- Correct MultiBasinRosenbrock reference optima and keep them inside declared bounds.
- Match pymoo GA/CMA-ES benchmark configuration semantics and pin the manuscript baseline in `requirements-benchmarks.txt`.

### Clarified

- Gradient optimizers remain opt-in: `lr_pop=None` and `lr_hyper=None` preserve non-gradient behavior; `-1` selects algorithm defaults.
- Live parent graph-reconstruction objective passes remain excluded from `n_evals` by design and are now disclosed as real auxiliary calls.
