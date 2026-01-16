#!/usr/bin/env python3
"""
Run dynamic feature-selection benchmark with regime shifts.

Key improvements:
- Problem instance shared across configs within same run
- Overlap parameter properly implemented
- Regime-aware feature recovery metrics
- Tracks adaptation speed after regime shifts

Outputs one JSON per algorithm:
  - <out>_GA.json
  - <out>_Adam.json
  - <out>_RandomSearch.json

Example:
    python evograd/benchmarks/run_dynamic_feature_selection.py --out results/dynfeaturesel.json \
        --with-random --with-adam --n-regimes 6 --shift-every 5000 --overlap 0.25
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

import torch
import numpy as np

# Path setup for running from project root
SCRIPT_DIR = Path(__file__).resolve().parent
EVOGRAD_PARENT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EVOGRAD_PARENT) not in sys.path:
    sys.path.insert(0, str(EVOGRAD_PARENT))

# Import local modules
from feature_selection.common import (
    resolve_device,
    set_all_seeds,
    make_synthetic_regression,
    write_results_json,
    compute_feature_recovery_metrics,
    compute_mask_statistics,
)
from feature_selection.dynamic_feature_selection import DynamicFeatureSelectELMProblem
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
    problem: DynamicFeatureSelectELMProblem,
    algorithm: str,
    config: str,
    pop: int,
    max_evals: int,
    seed: int,
    device,
    device_str: str,
    adam_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment on dynamic problem with regime tracking.
    
    Args:
        problem: The dynamic feature selection problem instance.
        algorithm: Algorithm name ('GA', 'RandomSearch', 'Adam').
        config: Configuration string for GA modes.
        pop: Population size.
        max_evals: Maximum evaluations.
        seed: Random seed.
        device: Torch device object.
        device_str: Device string for GA.
        adam_params: Optional Adam hyperparameters.
        
    Returns:
        Result dictionary with fitness history, regime info, and recovery metrics.
    """
    # Reset problem state for this run
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
    
    # Get final regime (after all evaluations)
    final_regime = problem.current_regime
    
    # Compute feature recovery metrics against FINAL regime's ground truth
    recovery_metrics = {}
    mask_stats = {}
    
    if best_solution is not None:
        # Get effective mask
        if hasattr(problem, 'get_effective_mask'):
            soft_mask = problem._constrain_mask(best_solution.unsqueeze(0)).squeeze(0)
        else:
            soft_mask = best_solution
        
        # Recovery against final regime
        final_true_indices = problem.get_current_informative_features()
        recovery_metrics = compute_feature_recovery_metrics(
            predicted_mask=soft_mask,
            true_indices=final_true_indices,
            n_features=problem.n_var,
            threshold=0.5,
        )
        
        mask_stats = compute_mask_statistics(soft_mask)
    
    # Compute regime overlap statistics
    regime_overlaps = []
    for i in range(len(problem.weights) - 1):
        overlap = problem.get_regime_overlap(i, i + 1)
        regime_overlaps.append(overlap)
    
    return {
        "algorithm": algorithm,
        "config": config,
        "seed": seed,
        "best_fitness_history": hist,
        "best_fitness": best_f,
        "n_evals": n_evals,
        "final_regime": final_regime,
        "regime_history": problem._regime_history,
        "recovery_metrics": recovery_metrics,
        "mask_statistics": mask_stats,
        "regime_overlaps": regime_overlaps,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run dynamic feature selection benchmark with regime shifts."
    )

    # Output
    ap.add_argument("--out", type=str, default="ga_dynfeatureselect.json",
                    help="Output JSON file path")
    
    # Experiment settings
    ap.add_argument("--runs", type=int, default=30,
                    help="Number of independent runs")
    ap.add_argument("--pop", type=int, default=100,
                    help="Population size")
    ap.add_argument("--max-evals", type=int, default=50000,
                    help="Maximum fitness evaluations per run")
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["classic", "diff", "full"],
                    choices=["classic", "adaptive", "diff", "full"],
                    help="GA configurations to test")
    
    # Problem settings
    ap.add_argument("--n-features", type=int, default=200,
                    help="Total number of features")
    ap.add_argument("--n-informative", type=int, default=20,
                    help="Number of informative features per regime")
    ap.add_argument("--noise", type=float, default=0.1,
                    help="Target noise level")
    ap.add_argument("--hidden", type=int, default=128,
                    help="ELM hidden layer size")
    ap.add_argument("--ridge-alpha", type=float, default=1e-2,
                    help="Ridge regression regularisation")
    ap.add_argument("--lambda-sparsity", type=float, default=1e-2,
                    help="Sparsity penalty coefficient")
    
    # Dynamic settings
    ap.add_argument("--n-regimes", type=int, default=6,
                    help="Number of regimes (distinct weight vectors)")
    ap.add_argument("--shift-every", type=int, default=5000,
                    help="Regime shift period in evaluations")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="Fraction of informative features shared between consecutive regimes")
    ap.add_argument("--cycle-regimes", action="store_true",
                    help="Cycle through regimes instead of stopping at last")
    
    # Difficulty presets (override individual settings)
    ap.add_argument("--difficulty", type=str, default=None,
                    choices=["easy", "medium", "hard", "nightmare"],
                    help="Difficulty preset (overrides individual dynamic settings)")
    
    
    # Hardware
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"],
                    help="Compute device")
    
    # Naming
    ap.add_argument("--function-name", type=str, default="DynFeatureSelectELM",
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

    # Apply difficulty presets (override individual settings)
    if args.difficulty:
        presets = {
            "easy": {
                "n_features": 200,
                "n_informative": 20,
                "shift_every": 10000,
                "overlap": 0.5,
                "n_regimes": 4,
                "cycle_regimes": False,
                "lambda_sparsity": 0.01,
                "noise": 0.1,
            },
            "medium": {
                "n_features": 200,
                "n_informative": 20,
                "shift_every": 5000,
                "overlap": 0.25,
                "n_regimes": 6,
                "cycle_regimes": False,
                "lambda_sparsity": 0.01,
                "noise": 0.1,
            },
            "hard": {
                "n_features": 300,
                "n_informative": 15,
                "shift_every": 2000,
                "overlap": 0.1,
                "n_regimes": 8,
                "cycle_regimes": True,
                "lambda_sparsity": 0.03,
                "noise": 0.2,
            },
            "nightmare": {
                "n_features": 500,
                "n_informative": 10,
                "shift_every": 1000,
                "overlap": 0.0,
                "n_regimes": 10,
                "cycle_regimes": True,
                "lambda_sparsity": 0.05,
                "noise": 0.3,
            },
        }
        preset = presets[args.difficulty]
        print(f"\n{'='*60}")
        print(f"Applying '{args.difficulty}' difficulty preset:")
        print(f"{'='*60}")
        for key, value in preset.items():
            setattr(args, key, value)
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

    device = resolve_device(args.device)
    device_str = args.device
    print(f"Device: {device}")
    print(f"Overlap: {args.overlap:.0%} of features shared between regimes")

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

        # Generate features only (targets generated per regime in problem)
        Xtr, _, Xva, _, _, _ = make_synthetic_regression(
            n_train=512,
            n_val=512,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            seed=seed,
            device=device,
        )

        # Create problem instance ONCE per run - shared across all configs!
        problem = DynamicFeatureSelectELMProblem(
            X_train=Xtr,
            X_val=Xva,
            n_informative=args.n_informative,
            noise=args.noise,
            hidden=args.hidden,
            ridge_alpha=args.ridge_alpha,
            lambda_sparsity=args.lambda_sparsity,
            n_regimes=args.n_regimes,
            shift_every=args.shift_every,
            overlap=args.overlap,
            cycle_regimes=args.cycle_regimes,
            seed=seed,
            device=device,
        )
        
        # Log regime overlap for verification
        if run == 0:
            print(f"Regime overlaps (Jaccard): ", end="")
            for i in range(args.n_regimes - 1):
                overlap_val = problem.get_regime_overlap(i, i + 1)
                print(f"R{i}→R{i+1}: {overlap_val:.2f}  ", end="")
            print()

        # Run GA configurations
        if not args.skip_ga:
            for cfg in args.configs:
                print(f"[run {run+1:02d}/{args.runs:02d}] GA {cfg} (seed={seed})")
                
                result = run_single_experiment(
                    problem=problem,
                    algorithm="GA",
                    config=cfg,
                    pop=args.pop,
                    max_evals=args.max_evals,
                    seed=seed,
                    device=device,
                    device_str=device_str,
                )
                result["function"] = args.function_name
                results.append(result)

        # Random baseline
        if args.with_random:
            print(f"[run {run+1:02d}/{args.runs:02d}] RandomSearch (seed={seed})")
            
            result = run_single_experiment(
                problem=problem,
                algorithm="RandomSearch",
                config="RandomSearch",
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
                device_str=device_str,
            )
            result["function"] = args.function_name
            results.append(result)

        # Adam baseline
        if args.with_adam:
            print(f"[run {run+1:02d}/{args.runs:02d}] Adam (seed={seed})")
            
            result = run_single_experiment(
                problem=problem,
                algorithm="Adam",
                config="Adam",
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
                device_str=device_str,
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
        "n_regimes": args.n_regimes,
        "shift_every": args.shift_every,
        "overlap": args.overlap,
        "cycle_regimes": args.cycle_regimes,
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
