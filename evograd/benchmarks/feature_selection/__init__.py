"""
Feature Selection Benchmarks for EvoGrad.

This package provides static and dynamic feature selection problems
for evaluating differentiable evolutionary algorithms.

Problems:
    - FeatureSelectELMProblem: Static feature selection with ELM evaluation
    - DynamicFeatureSelectELMProblem: Time-varying ground truth with regime shifts

Runners:
    - run_ga: Genetic Algorithm with classic/adaptive/diff/full modes
    - run_adam: Adam gradient descent baseline (projected to [0,1])
    - run_random: Random search baseline

Key Features:
    - Differentiable objective (closed-form ridge regression)
    - All methods optimise continuous masks in [0, 1]
    - GA uses repair operators; Adam uses projection
    - Feature recovery metrics (Jaccard, F1, Precision, Recall)
    - Controlled overlap between regimes in dynamic setting

Example:
    >>> from feature_selection import FeatureSelectELMProblem
    >>> from feature_selection.common import make_synthetic_regression
    >>> 
    >>> X_tr, y_tr, X_va, y_va, true_idx, weights = make_synthetic_regression(
    ...     n_train=512, n_val=512, n_features=200, n_informative=20,
    ...     noise=0.1, seed=42, device=torch.device('cpu')
    ... )
    >>> problem = FeatureSelectELMProblem(
    ...     X_tr, y_tr, X_va, y_va,
    ...     hidden=128, lambda_sparsity=0.01,
    ... )
    >>> fitness = problem.evaluate(population)  # population in [0, 1]^D
"""

from .feature_selection import FeatureSelectELMProblem
from .dynamic_feature_selection import DynamicFeatureSelectELMProblem
from .common import (
    resolve_device,
    set_all_seeds,
    make_synthetic_regression,
    compute_feature_recovery_metrics,
    compute_mask_statistics,
    write_results_json,
)
from .ga_runner import run_ga
from .adam_runner import run_adam
from .random_runner import run_random

__all__ = [
    # Problems
    "FeatureSelectELMProblem",
    "DynamicFeatureSelectELMProblem",
    # Utilities
    "resolve_device",
    "set_all_seeds",
    "make_synthetic_regression",
    "compute_feature_recovery_metrics",
    "compute_mask_statistics",
    "write_results_json",
    # Runners
    "run_ga",
    "run_adam",
    "run_random",
]

__version__ = "0.2.1"
