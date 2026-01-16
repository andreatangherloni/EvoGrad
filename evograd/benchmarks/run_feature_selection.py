#!/usr/bin/env python3
"""
Run static feature-selection benchmark (FeatureSelectELM).

Outputs one JSON per algorithm:
  - <out>_GA.json
  - <out>_Adam.json
  - <out>_RandomSearch.json

Example:
    python run_feature_selection.py --out results/featureselect.json --with-random --with-adam
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Path setup for running from project root
SCRIPT_DIR = Path(__file__).resolve().parent
EVOGRAD_PARENT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))  # For 'feature_selection' subpackage
if str(EVOGRAD_PARENT) not in sys.path:
    sys.path.insert(0, str(EVOGRAD_PARENT))  # For 'evograd' package

# Import local modules
from feature_selection.common import (
    resolve_device,
    set_all_seeds,
    make_synthetic_regression,
    write_results_json,
    compute_feature_recovery_metrics,
    compute_mask_statistics,
)

from feature_selection.feature_selection import FeatureSelectELMProblem
from feature_selection.ga_runner import run_ga
from feature_selection.random_runner import run_random
from feature_selection.adam_runner import run_adam


def _split_and_write(
    out_path: Path,
    n_var: int,
    max_evals: int,
    n_runs: int,
    results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Split results by algorithm and write separate JSON files.
    
    Creates files named <out_stem>_<Algorithm>.json for each algorithm.
    """
    buckets = defaultdict(list)
    for r in results:
        algo = r.get("algorithm", "Unknown")
        buckets[algo].append(r)

    for algo, algo_results in buckets.items():
        out_algo = out_path.with_name(out_path.stem + f"_{algo}.json")
        write_results_json(
            out_path=out_algo,
            algorithm=algo,
            n_var=n_var,
            max_evals=max_evals,
            n_runs=n_runs,
            results=algo_results,
            metadata=metadata,
        )


def run_single_experiment(
    problem: FeatureSelectELMProblem,
    algorithm: str,
    config: str,
    pop: int,
    max_evals: int,
    seed: int,
    device,
    device_str: str,
    true_indices: np.ndarray,
    adam_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment and collect results with metrics.
    
    Args:
        problem: The feature selection problem instance.
        algorithm: Algorithm name ('GA', 'RandomSearch', 'Adam').
        config: Configuration string for GA modes.
        pop: Population size.
        max_evals: Maximum evaluations.
        seed: Random seed.
        device: Torch device object.
        device_str: Device string for GA.
        true_indices: Ground-truth informative feature indices.
        adam_params: Optional Adam hyperparameters.
        
    Returns:
        Result dictionary with fitness history and feature recovery metrics.
    """
    # Reset problem state (for dynamic compatibility)
    problem.reset()
    
    # Run algorithm
    if algorithm == "GA":
        best_f, hist, n_evals, best_solution = run_ga(
            problem=problem,
            config=config,
            pop=pop,
            max_evals=max_evals,
            seed=seed,
            device=device_str,
        )
    elif algorithm == "RandomSearch":
        best_f, hist, n_evals, best_solution = run_random(
            problem=problem,
            pop=pop,
            max_evals=max_evals,
            seed=seed,
            device=device,
        )
    elif algorithm == "Adam":
        best_f, hist, n_evals, best_solution = run_adam(
            problem=problem,
            pop=pop,
            max_evals=max_evals,
            seed=seed,
            device=device,
            lr=adam_params.get("lr", 0.05),
            b1=adam_params.get("b1", 0.9),
            b2=adam_params.get("b2", 0.999),
            wd=adam_params.get("wd", 0.0),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Compute feature recovery metrics if we have the best solution
    recovery_metrics = {}
    mask_stats = {}
    
    if best_solution is not None:
        # Get the effective mask (apply constraints if needed)
        if hasattr(problem, 'get_effective_mask'):
            binary_mask = problem.get_effective_mask(best_solution.unsqueeze(0))
            soft_mask = problem._constrain_mask(best_solution.unsqueeze(0)).squeeze(0)
        else:
            soft_mask = best_solution
            binary_mask = (best_solution > 0.5).float()
        
        recovery_metrics = compute_feature_recovery_metrics(
            predicted_mask=soft_mask,
            true_indices=true_indices,
            n_features=problem.n_var,
            threshold=0.5,
        )
        
        mask_stats = compute_mask_statistics(soft_mask)
    
    return {
        "algorithm": algorithm,
        "config": config,
        "seed": seed,
        "best_fitness_history": hist,
        "best_fitness": best_f,
        "n_evals": n_evals,
        "recovery_metrics": recovery_metrics,
        "mask_statistics": mask_stats,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run static feature selection benchmark with EvoGrad."
    )

    # Output
    ap.add_argument("--out", type=str, default="ga_featureselect.json",
                    help="Output JSON file path")
    
    # Experiment settings
    ap.add_argument("--runs", type=int, default=30,
                    help="Number of independent runs")
    ap.add_argument("--pop", type=int, default=100,
                    help="Population size")
    ap.add_argument("--max-evals", type=int, default=50000,
                    help="Maximum fitness evaluations per run")
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["classic", "differentiable", "full"],
                    choices=["classic", "adaptive", "differentiable", "full"],
                    help="GA configurations to test")
    
    # Problem settings
    ap.add_argument("--n-features", type=int, default=200,
                    help="Total number of features")
    ap.add_argument("--n-informative", type=int, default=20,
                    help="Number of informative features")
    ap.add_argument("--noise", type=float, default=0.1,
                    help="Target noise level")
    ap.add_argument("--hidden", type=int, default=128,
                    help="ELM hidden layer size")
    ap.add_argument("--ridge-alpha", type=float, default=1e-2,
                    help="Ridge regression regularisation")
    ap.add_argument("--lambda-sparsity", type=float, default=1e-2,
                    help="Sparsity penalty coefficient")
    
    # Hardware
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"],
                    help="Compute device")
    
    # Naming
    ap.add_argument("--function-name", type=str, default="FeatureSelectELM",
                    help="Function name for results")

    # Baseline toggles
    ap.add_argument("--with-random", action="store_true",
                    help="Include RandomSearch baseline")
    ap.add_argument("--with-adam", action="store_true",
                    help="Include Adam baseline")
    ap.add_argument("--skip-ga", action="store_true",
                    help="Skip GA runs (baselines only)")

    # Adam hyperparameters
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-weight-decay", type=float, default=0.0)

    args = ap.parse_args()

    device = resolve_device(args.device)
    device_str = args.device
    print(f"Device: {device}")

    adam_params = {
        "lr": args.adam_lr,
        "b1": args.adam_beta1,
        "b2": args.adam_beta2,
        "wd": args.adam_weight_decay,
    }

    results = []

    for run in range(args.runs):
        seed = 1234 + run
        set_all_seeds(seed)

        # Generate data for this run
        Xtr, ytr, Xva, yva, true_indices, true_weights = make_synthetic_regression(
            n_train=512,
            n_val=512,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            seed=seed,
            device=device,
        )

        # Create problem instance ONCE per run - shared across all configs!
        # This ensures fair comparison: same ELM weights W, b for all algorithms
        problem = FeatureSelectELMProblem(
            X_train=Xtr,
            y_train=ytr,
            X_val=Xva,
            y_val=yva,
            hidden=args.hidden,
            ridge_alpha=args.ridge_alpha,
            lambda_sparsity=args.lambda_sparsity,
            seed=seed,
            device=device,
            differentiable=False,  # Will be set per-algorithm
        )

        # Run GA configurations
        if not args.skip_ga:
            for cfg in args.configs:
                print(f"[run {run+1:02d}/{args.runs}] GA {cfg} (seed={seed})")
                
                result = run_single_experiment(
                    problem=problem,
                    algorithm="GA",
                    config=cfg,
                    pop=args.pop,
                    max_evals=args.max_evals,
                    seed=seed,
                    device=device,
                    device_str=device_str,
                    true_indices=true_indices,
                )
                result["function"] = args.function_name
                results.append(result)

        # Random baseline
        if args.with_random:
            print(f"[run {run+1:02d}/{args.runs}] RandomSearch (seed={seed})")
            
            result = run_single_experiment(
                problem=problem,
                algorithm="RandomSearch",
                config="random search",
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
                device_str=device_str,
                true_indices=true_indices,
            )
            result["function"] = args.function_name
            results.append(result)

        # Adam baseline
        if args.with_adam:
            print(f"[run {run+1:02d}/{args.runs}] Adam (seed={seed})")
            
            result = run_single_experiment(
                problem=problem,
                algorithm="Adam",
                config="adam",
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
                device_str=device_str,
                true_indices=true_indices,
                adam_params=adam_params,
            )
            result["function"] = args.function_name
            results.append(result)

    # Save results split by algorithm
    out_path = Path(args.out)
    metadata = {
        "n_features": args.n_features,
        "n_informative": args.n_informative,
        "noise": args.noise,
        "hidden": args.hidden,
        "ridge_alpha": args.ridge_alpha,
        "lambda_sparsity": args.lambda_sparsity,
        "pop_size": args.pop,
    }
    
    _split_and_write(
        out_path=out_path,
        n_var=args.n_features,
        max_evals=args.max_evals,
        n_runs=args.runs,
        results=results,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()