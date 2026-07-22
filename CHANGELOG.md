# Changelog

## 0.4.0

### Changed (breaking)

- **Learning-rate sentinels redesigned.** The `-1` sentinel is removed:
  `lr_pop=-1` / `lr_hyper=-1` (and the `grad_clip_*` equivalents) now raise
  `ValueError`. `None` (the default) no longer means "disable"; it means
  **auto**: if the channel's flag exposes learnable parameters
  (`differentiable=True` for the population, `adaptive=True` / differentiable
  operators for the hyperparameters) and the objective provides a gradient,
  the per-algorithm default learning rate is used. `0` explicitly disables a
  channel (with a warning). Consequence: a bare `minimize(problem, algo)` on a
  gradient-providing objective now runs in differentiable mode whenever the
  algorithm was constructed with `differentiable=True` / `adaptive=True`
  (GA's constructor defaults; DE, PSO, CMA-ES and SHADE default both flags to
  `False` and stay classical unless the flags are set). This reverses the
  0.3.0 clarification that `None` preserves non-gradient behavior.
- **Gradient clipping defaults now apply automatically.** `grad_clip_pop` /
  `grad_clip_hyper` `= None` (the default) now selects the per-algorithm
  clipping threshold whenever that channel is gradient-driven (previously
  `None` meant no clipping and only `-1` selected the defaults); pass `0` to
  disable clipping explicitly. Gradient-driven runs that previously omitted
  the clip arguments while passing `lr_pop=-1`-style learning rates will see
  (slightly) different trajectories; runs that passed all four `-1` sentinels
  — including every documented benchmark invocation — are verified
  float-identical under the new auto-resolution across GA/DE/PSO/CMA-ES.
- The benchmark CLI (`run_benchmark_functions.py`) defaults its
  `--lr_pop/--lr_hyper/--grad_clip_pop/--grad_clip_hyper` flags to omitted
  (auto) instead of `-1`; resolved values are unchanged, so benchmark
  configurations behave identically (verified float-identical trajectories
  vs. the previous `-1` invocation on seeded runs).

### Added

- **Objective-gradient probe.** Before enabling any gradient channel,
  `minimize()` evaluates the loss composite (objective plus, for constrained
  problems, the exterior constraint penalty — so differentiable constraints
  keep gradient mode available even for black-box objectives) once at the
  midpoint of the box bounds with a grad-requiring input and checks the
  output for a `grad_fn`. The probe is RNG-neutral for torch, NumPy and
  Python generators (states are snapshotted and restored, so internally
  stochastic objectives do not perturb seeded runs), excluded from `n_evals`,
  and retried with a population-sized batch when the objective rejects
  single rows (e.g. BatchNorm or fixed-batch objectives). Black-box
  objectives can no longer crash the differentiable path (`element 0 of
  tensors does not require grad`) or silently pretend to use gradients:
  every affected channel falls back to the classical update with an explicit
  warning — also when a positive learning rate was requested. If the probe
  itself cannot run (inconclusive), auto-resolution stays classical while
  explicitly requested learning rates / optimizers are honored unverified.
- **First-generation gradient diagnostic.** An optimizer whose parameters
  receive no gradient on the first differentiable generation is dropped with a
  warning (e.g. CMA-ES's population Parameter, which never enters the loss
  graph; its search center `_mean` is driven by `lr_hyper` instead). If every
  optimizer is dropped, the run continues classically.
- `result.extra` now reports `gradient_channels` (which channels ended up
  gradient-driven — computed from actual optimizer coverage and kept truthful
  after diagnostic drops), `lr_pop_effective` (after the `1/sqrt(n_var)`
  population scaling), and `lr_hyper_effective`.
- With a user-supplied `optimizer=`, `lr_pop`/`lr_hyper` are ignored with a
  warning (the optimizer's own learning rates apply), negative sentinels
  still raise, and channel reporting reflects the parameters the supplied
  optimizers actually cover.

### Notes

- Versions 0.2.0/0.2.1 were internal and never tagged; the public release
  sequence is 0.1.2 → 0.3.0 → 0.4.0.

## 0.3.0

### Fixed

- Keep `best_solution` paired with the coordinates at which `best_fitness` was evaluated.
- Keep the differentiable PSO population consistent with its fitness: the committed position (`P₀ + v − lr·∇`) is now re-evaluated so `population`, `fitness`, and the personal/global bests all agree (previously `fitness` was that of the pre-gradient position `P₀ + v`). Benchmarks show no change in optimisation quality; the extra objective pass is uncounted in `n_evals`.
- Avoid autograd/backward work and stale gradient accumulation when both learning rates are disabled.
- Enforce declared constraints through the configurable `Problem.constraint_penalty` and report final feasibility.
- Translate nested `TargetReached` criteria correctly in `maximize()`.
- Keep L-SHADE population sizes, offspring counts, and differentiable optimizer parameters synchronized during reduction.
- Use bounded Deb polynomial mutation and preserve fixed variables in reflect/wrap repair.
- Enforce box bounds uniformly, independent of operator and mode: GA now clamps offspring to `[xl, xu]` when no repair is configured (previously a non-clamping mutation such as Gaussian could evaluate and return out-of-bounds points), and differentiable runs project the gradient-updated population back into the box so DE/SHADE/GA never evaluate or return out-of-bounds solutions. Only the decision-variable population is clamped; hyperparameters are not.
- Expose `result.extra['best_raw_objective']` for constrained problems so the true (unpenalised) objective is recoverable when the returned best is infeasible (`best_fitness` folds in the constraint penalty).
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
