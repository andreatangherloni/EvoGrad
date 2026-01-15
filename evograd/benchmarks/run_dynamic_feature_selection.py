#!/usr/bin/env python3
"""
Run dynamic feature-selection benchmark (DynFeatureSelectELM) with regime shifts.

Outputs the same JSON structure as the rest of the EvoGrad benchmark tooling
expects (plot_benchmarks.py, ranking scripts, etc.).

Example:
  python benchmarks/run_dynamic_feature_selection.py --out results/ga_dynfeaturesel.json --with-random --with-adam
"""

import argparse
import sys
from pathlib import Path

# Directory containing this script (benchmarks/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent of benchmarks/ is evograd/, parent of evograd/ contains evograd package
EVOGRAD_PARENT = SCRIPT_DIR.parent.parent  # Go up two levels to find evograd package

# Add paths for imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))  # For 'functions' subpackage
if str(EVOGRAD_PARENT) not in sys.path:
    sys.path.insert(0, str(EVOGRAD_PARENT))  # For 'evograd' package

from feature_selection.common import (
    resolve_device,
    set_all_seeds,
    make_synthetic_regression,
    write_results_json,
)
from feature_selection.dynamic_feature_selection import DynamicFeatureSelectELMProblem
from feature_selection.ga_runner import run_ga
from feature_selection.random_runner import run_random
from feature_selection.adam_runner import run_adam


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", type=str, default="ga_featureselect.json")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=50000)
    ap.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["classic", "diff", "full"],
        choices=["classic", "adaptive", "diff", "full"],
    )
    ap.add_argument("--n-features", type=int, default=200)
    ap.add_argument("--n-informative", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ridge-alpha", type=float, default=1e-2)
    ap.add_argument("--lambda-sparsity", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--function-name", type=str, default="DynFeatureSelectELM")

    # ---- Dynamic extras ----
    ap.add_argument("--n-regimes", type=int, default=6, help="Number of regimes (ground-truth weight vectors).")
    ap.add_argument("--shift-every", type=int, default=5000, help="Regime shift period in *evaluations*.")
    ap.add_argument("--overlap", type=float, default=0.25, help="Fraction of informative features shared between consecutive regimes.")
    ap.add_argument("--cycle-regimes", action="store_true", help="Cycle regimes instead of stopping at the last one.")

    # ---- Baselines toggles ----
    ap.add_argument("--with-random", action="store_true", help="Also run RandomSearch baseline.")
    ap.add_argument("--with-adam", action="store_true", help="Also run Adam baseline.")
    ap.add_argument("--skip-ga", action="store_true", help="Skip GA runs (only baselines).")

    # ---- Adam hyperparams ----
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-weight-decay", type=float, default=0.0)

    args = ap.parse_args()

    device = resolve_device(args.device)
    device_str = args.device  # keep original string for GA factory usage
    print(f"Device: {device}")

    results = []

    for run in range(args.runs):
        seed = 1234 + run
        set_all_seeds(seed)

        # Data: in dynamic, we generate X only and let the problem generate y per regime
        Xtr, _, Xva, _, _ = make_synthetic_regression(
            n_train=512,
            n_val=512,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            seed=seed,
            device=device,
        )

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

        # ------------------------
        # GA configs
        # ------------------------
        if not args.skip_ga:
            for cfg in args.configs:
                print(f"[run {run+1:02d}/{args.runs}] GA {cfg} (seed={seed})")
                best_f, hist, n_evals = run_ga(
                    problem=problem,
                    config=cfg,
                    pop=args.pop,
                    max_evals=args.max_evals,
                    seed=seed,
                    device=device_str,
                )
                results.append({
                    "algorithm": "GA",
                    "config": cfg,
                    "function": args.function_name,
                    "seed": seed,
                    "best_fitness_history": hist,
                    "best_fitness": best_f,
                    "n_evals": n_evals,
                })

        # ------------------------
        # Random baseline
        # ------------------------
        if args.with_random:
            print(f"[run {run+1:02d}/{args.runs}] RandomSearch baseline (seed={seed})")
            best_f, hist, n_evals = run_random(
                problem=problem,
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
            )
            results.append({
                "algorithm": "RandomSearch",
                "config": "baseline",
                "function": args.function_name,
                "seed": seed,
                "best_fitness_history": hist,
                "best_fitness": best_f,
                "n_evals": n_evals,
            })

        # ------------------------
        # Adam baseline
        # ------------------------
        if args.with_adam:
            print(f"[run {run+1:02d}/{args.runs}] Adam baseline (seed={seed})")
            best_f, hist, n_evals = run_adam(
                problem=problem,
                pop=args.pop,
                max_evals=args.max_evals,
                seed=seed,
                device=device,
                lr=args.adam_lr,
                b1=args.adam_beta1,
                b2=args.adam_beta2,
                wd=args.adam_weight_decay,
            )
            results.append({
                "algorithm": "Adam",
                "config": "baseline",
                "function": args.function_name,
                "seed": seed,
                "best_fitness_history": hist,
                "best_fitness": best_f,
                "n_evals": n_evals,
            })

    out_path = Path(args.out)
    write_results_json(
        out_path=out_path,
        algorithm="GA",
        n_var=args.n_features,
        max_evals=args.max_evals,
        n_runs=args.runs,
        results=results,
    )


if __name__ == "__main__":
    main()