# Project Context: EvoGrad

## Overview
EvoGrad is a research-oriented Python library and accompanying academic manuscript focused on differentiable evolutionary algorithms, including CMA-ES and differentiable PSO. The project bridges complex mathematical optimization with algorithmic frameworks, enabling evolutionary strategies to operate reliably within differentiable computational graphs.

## Architecture Notes
- **`evograd/`**: The core repository containing the primary logic (e.g., `evograd/core/maximize.py`) and the rigorous benchmarking suites (e.g., `evograd/benchmarks/functions/cec2017/`).
- **`Paper/`**: Contains the LaTeX source files for the manuscript. The theoretical formulations and historical tables documented here are the source of truth; the Python code must flawlessly reflect these derivations.
- **`experiments/` & Root Scripts**: Directories and scripts (like `plot_benchmarks.py` and `Test_new_evograd.ipynb`) used for running analytical trials and generating visual evidence for the paper.
- **Environment**: Developed primarily in a macOS environment. The project strictly utilizes `uv` for package and dependency management (refer to `pyproject.toml` and `uv.lock`).

## Coding Conventions
- **Mathematical Rigor**: Implementations must be theoretically sound across all coordinate spaces. Agents must carefully evaluate edge cases in gradient derivations, paying special attention to boundary conditions and zero-expression mappings (always verify that derived formulas hold true when variables equal zero).
- **Reproducibility**: Maintaining exact API compatibility and numerical consistency is mandatory. Algorithmic updates must not invalidate the historical performance tables, break seeded reproducibility, or silently alter results on established benchmarks.
- **Version Control**: The project follows GitHub branch management and Pull Request protocols.
- **Testing Requirements**: Passing the full integrated test suite, including all audit regressions, is a strict prerequisite before any mathematical or architectural changes are merged.

## Current Focus
Finalizing the EvoGrad manuscript and conducting a comprehensive evidence-oriented review of the codebase. The immediate goal is ensuring recent codebase compatibility fixes—such as CMA soft selection stability and positional constructor order—are verified against the theoretical claims without introducing undocumented paper caveats.

## Known Issues & Watch-outs
- **Sampling Reproducibility**: Refactoring classical sampling methods (like NPoint) risks breaking seeded reproducibility or introducing new instantiation errors.
- **Floating-Point Drift**: Composition functions (F21-F30) are susceptible to epsilon shifts (e.g., variations at the 6th significant digit), which can disrupt exact historical data matching.
- **Boundary Vulnerabilities**: Hard clamps (e.g., clamping to `0.0` rather than `1e-12`) can draw `NaN`-poisons into the backward pass, leading to silent failures during long, boundary-hugging runs.