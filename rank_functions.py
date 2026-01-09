#!/usr/bin/env python3
"""
Rank benchmark functions by where Diff/Full outperform Classic the most.

This script reuses your plot script's loader (load_data + normalization),
so it supports the same JSON formats and CLI filtering behavior.

Ranking signals (robust):
- Final fitness improvement: median(Classic) - median(Diff/Full)
- AUC improvement (optional): AUC(Classic) - AUC(Diff/Full) over evaluations

Usage examples:
    python rank_functions.py --data results/*.json
    python rank_functions.py --data results/*.json --functions sphere rastrigin
    python rank_functions.py --data results/*.json --metric final --compare full
    python rank_functions.py --data results/*.json --metric auc --compare diff --max-evals 50000
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_finite(x: List[float]) -> np.ndarray:
    arr = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return arr

def _median(x: List[float]) -> float:
    arr = _safe_finite(x)
    return float(np.median(arr)) if arr.size else float("nan")

def _iqr(x: List[float]) -> float:
    arr = _safe_finite(x)
    if arr.size < 2:
        return float("nan")
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return float(q3 - q1)

def _pad_histories(histories: List[List[float]]) -> Optional[np.ndarray]:
    valid = [h for h in histories if h is not None and len(h) > 0 and np.all(np.isfinite(h))]
    if not valid:
        return None
    m = max(len(h) for h in valid)
    padded = []
    for h in valid:
        if len(h) < m:
            padded.append(list(h) + [h[-1]] * (m - len(h)))
        else:
            padded.append(list(h)[:m])
    return np.asarray(padded, dtype=float)

def _auc_from_histories(
    histories: List[List[float]],
    total_evals: Optional[int],
    max_evals: Optional[int] = None,
) -> float:
    """
    AUC of mean best-fitness curve vs evals (lower is better).
    Returns nan if no usable histories.
    """
    H = _pad_histories(histories)
    if H is None:
        return float("nan")
    mean_curve = np.nanmean(H, axis=0)
    if np.all(np.isnan(mean_curve)):
        return float("nan")

    n = len(mean_curve)
    if total_evals is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.linspace(0.0, float(total_evals), n)

    if max_evals is not None:
        mask = x <= float(max_evals)
        if not np.any(mask):
            return float("nan")
        x = x[mask]
        mean_curve = mean_curve[mask]

    # trapz AUC
    return float(np.trapz(mean_curve, x))

def _extract_alg_and_type(variant: str) -> Tuple[str, str]:
    parts = variant.split()
    alg = parts[0] if parts else "Unknown"
    vtype = parts[1] if len(parts) > 1 else "unknown"
    return alg, vtype

# -----------------------------------------------------------------------------
# Main ranking
# -----------------------------------------------------------------------------

def rank_functions(
    data: Dict,
    metadata: Dict,
    compare: str = "full",            # "diff" or "full"
    metric: str = "final",            # "final" or "auc"
    max_evals: Optional[int] = None,
    algorithms: Optional[List[str]] = None,
    min_runs: int = 5,
) -> List[Dict]:
    """
    Returns list of rows sorted by descending improvement:
        improvement = Classic - (Diff/Full)
    Positive means Diff/Full is better (smaller fitness or smaller AUC).
    """
    compare = compare.lower()
    metric = metric.lower()
    assert compare in ("diff", "full")
    assert metric in ("final", "auc")

    total_evals = metadata.get("max_evals", None)

    rows = []
    for func, func_data in data.items():
        # group by algorithm family
        per_alg = []
        for variant_name, entry in func_data.items():
            alg, vtype = _extract_alg_and_type(variant_name)
            if algorithms is not None and alg not in algorithms:
                continue

        # Build maps alg -> {classic, compare}
        alg_to = {}
        for variant_name, entry in func_data.items():
            alg, vtype = _extract_alg_and_type(variant_name)
            if algorithms is not None and alg not in algorithms:
                continue
            alg_to.setdefault(alg, {})
            alg_to[alg][vtype] = entry

        for alg, m in alg_to.items():
            if "classic" not in m or compare not in m:
                continue

            classic = m["classic"]
            other = m[compare]

            # Require enough runs
            n_c = len(classic.get("final_fitness", []))
            n_o = len(other.get("final_fitness", []))
            if n_c < min_runs or n_o < min_runs:
                continue

            if metric == "final":
                c_med = _median(classic.get("final_fitness", []))
                o_med = _median(other.get("final_fitness", []))
                if not np.isfinite(c_med) or not np.isfinite(o_med):
                    continue
                improv = c_med - o_med
                per_alg.append((alg, improv, c_med, o_med, _iqr(classic.get("final_fitness", [])), _iqr(other.get("final_fitness", []))))
            else:
                c_auc = _auc_from_histories(classic.get("history", []), total_evals, max_evals=max_evals)
                o_auc = _auc_from_histories(other.get("history", []), total_evals, max_evals=max_evals)
                if not np.isfinite(c_auc) or not np.isfinite(o_auc):
                    continue
                improv = c_auc - o_auc
                per_alg.append((alg, improv, c_auc, o_auc, float("nan"), float("nan")))

        if not per_alg:
            continue

        # Aggregate across algorithms: mean improvement + count + “wins”
        improvs = np.array([x[1] for x in per_alg], dtype=float)
        n = len(improvs)
        mean_improv = float(np.mean(improvs))
        median_improv = float(np.median(improvs))
        wins = int(np.sum(improvs > 0))

        rows.append({
            "function": func,
            "mean_improvement": mean_improv,
            "median_improvement": median_improv,
            "wins_over_algs": wins,
            "n_algs_compared": n,
            "per_algorithm": per_alg,  # keep details for printing
        })

    # Sort: most consistently beneficial first
    rows.sort(key=lambda r: (r["wins_over_algs"], r["mean_improvement"]), reverse=True)
    return rows

# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

def _load_plot_module(plot_script_path: str):
    p = Path(plot_script_path)
    if not p.exists():
        raise FileNotFoundError(f"Plot script not found: {plot_script_path}")

    spec = importlib.util.spec_from_file_location("plot_module", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-script", type=str, default="plot_benchmarks.py",
                    help="Path to your plot script (the one that defines load_data, normalize_function_name, etc.)")
    ap.add_argument("--data", type=str, nargs="*", default=None,
                    help="JSON files or glob patterns to load")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Directory containing JSON result files")
    ap.add_argument("--functions", type=str, nargs="+", default=None,
                    help="Filter to specific functions (same behavior as plot script)")
    ap.add_argument("--compare", type=str, default="full", choices=["diff", "full"],
                    help="Compare against Classic using either Diff or Full")
    ap.add_argument("--metric", type=str, default="final", choices=["final", "auc"],
                    help="Ranking metric: final fitness or AUC of convergence curve")
    ap.add_argument("--max-evals", type=int, default=None,
                    help="If metric=auc, integrate only up to this evaluation budget")
    ap.add_argument("--algorithms", type=str, nargs="*", default=None,
                    help="Restrict to a subset of algorithms (e.g., DE SHADE GA)")
    ap.add_argument("--min-runs", type=int, default=5,
                    help="Minimum runs required per variant to include in comparisons")
    ap.add_argument("--top", type=int, default=20,
                    help="Print top-N functions")
    args = ap.parse_args()

    plot_mod = _load_plot_module(args.plot_script)

    data, metadata = plot_mod.load_data(
        filepaths=args.data,
        data_dir=args.data_dir,
        filter_functions=args.functions,
    )
    if not data:
        print("No data loaded.")
        return

    rows = rank_functions(
        data=data,
        metadata=metadata,
        compare=args.compare,
        metric=args.metric,
        max_evals=args.max_evals,
        algorithms=args.algorithms,
        min_runs=args.min_runs,
    )

    if not rows:
        print("No comparable functions found (missing variants or insufficient runs).")
        return

    # Pretty print
    metric_name = "FinalFitness" if args.metric == "final" else f"AUC@{args.max_evals or metadata.get('max_evals', 'max')}"
    print("\n" + "=" * 90)
    print(f"RANKING FUNCTIONS BY {args.compare.upper()} vs CLASSIC  |  metric={metric_name}")
    print("=" * 90)
    print(f"{'Rank':>4}  {'Function':<18}  {'Wins':>4}  {'#Algs':>5}  {'MeanImprovement':>16}  {'MedianImprovement':>18}")
    print("-" * 90)

    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:>4}  {r['function']:<18}  {r['wins_over_algs']:>4}/{r['n_algs_compared']:<1}  {r['n_algs_compared']:>5}  "
              f"{r['mean_improvement']:>16.4e}  {r['median_improvement']:>18.4e}")

    print("\nDetails for top functions (per algorithm):")
    for i, r in enumerate(rows[:min(args.top, 10)], 1):
        print("\n" + "-" * 90)
        print(f"[{i}] {r['function']}  (wins {r['wins_over_algs']}/{r['n_algs_compared']}, mean={r['mean_improvement']:.4e})")
        for (alg, improv, c_val, o_val, c_iqr, o_iqr) in r["per_algorithm"]:
            if args.metric == "final":
                print(f"  {alg:<6}  improvement={improv:+.4e}  classic_med={c_val:.4e}  {args.compare}_med={o_val:.4e}  "
                      f"classic_IQR={c_iqr:.2e}  {args.compare}_IQR={o_iqr:.2e}")
            else:
                print(f"  {alg:<6}  improvement={improv:+.4e}  classic_auc={c_val:.4e}  {args.compare}_auc={o_val:.4e}")

    print("\nTip: try --metric auc --max-evals <budget> to find where Diff/Full is more sample-efficient.\n")

if __name__ == "__main__":
    main()