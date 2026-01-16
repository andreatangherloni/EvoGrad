#!/usr/bin/env python3
"""
Run static feature-selection benchmark (FeatureSelectELM).

It outputs one JSON per algorithm:
  - <out>_GA.json
  - <out>_Adam.json
  - <out>_RandomSearch.json

Example:
  python benchmarks/run_feature_selection.py --out results/ga_featureselect.json --with-random --with-adam
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
EVOGRAD_PARENT = SCRIPT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EVOGRAD_PARENT) not in sys.path:
    sys.path.insert(0, str(EVOGRAD_PARENT))

from feature_selection.common import (
    resolve_device,
    set_all_seeds,
    make_synthetic_regression,
    write_results_json,
)
from feature_selection.feature_selection import FeatureSelectELMProblem
from feature_selection.ga_runner import run_ga
from feature_selection.random_runner import run_random
from feature_selection.adam_runner import run_adam


def _split_and_write(out_path: Path, n_var: int, max_evals: int, n_runs: int, results: list):
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
        )


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
    ap.add_argument("--function-name", type=str, default="FeatureSelectELM")

    ap.add_argument("--with-random", action="store_true")
    ap.add_argument("--with-adam", action="store_true")
    ap.add_argument("--skip-ga", action="store_true")

    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-weight-decay", type=float, default=0.0)

    args = ap.parse_args()

    device = resolve_device(args.device)
    device_str = args.device
    print(f"Device: {device}")

    results = []

    for run in range(args.runs):
        seed = 1234 + run
        set_all_seeds(seed)

        Xtr, ytr, Xva, yva, _ = make_synthetic_regression(
            n_train=512,
            n_val=512,
            n_features=args.n_features,
            n_informative=args.n_informative,
            noise=args.noise,
            seed=seed,
            device=device,
        )

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
        )

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
    _split_and_write(
        out_path=out_path,
        n_var=args.n_features,
        max_evals=args.max_evals,
        n_runs=args.runs,
        results=results,
    )


if __name__ == "__main__":
    main()