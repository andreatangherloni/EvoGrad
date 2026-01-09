#!/usr/bin/env python3
"""
EvoGrad Benchmark Visualization

Generates publication-quality figures from benchmark results:
  1. Boxplots comparing algorithm variants across functions
  2. Convergence curves showing optimization progress

This script keeps the *new* CLI/data features (function filtering, listing,
metadata handling, etc.) while restoring the *old* plotting style:
  - Boxplots: grouped by algorithm family with consistent variant colors,
    and a single shared legend at the bottom.
  - Convergence: mean ± std curves, colored by variant type, line style by
    algorithm family, with a two-part shared legend at the bottom.
  - No per-subplot legends, and no curve markers (cleaner, less clutter).

Usage:
    python plot_benchmarks.py --data results/*.json -o figures/
    python plot_benchmarks.py --data results/DE_*.json --functions sphere rastrigin
    python plot_benchmarks.py --data results/ --format png --dpi 300
"""

import argparse
import glob
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Default function order for plots
FUNCTIONS = [
    'Sphere', 'Ellipsoid', 'Rosenbrock', 'Rastrigin', 'Ackley',
    'Griewank', 'Schwefel', 'Levy', 'Michalewicz', 'Alpine',
    'Zakharov', 'Weierstrass', 'Salomon', 'StyblinskiTang',
]

# CEC 2017 functions
CEC2017_FUNCTIONS = [f'Cec2017_F{i}' for i in range(1, 31)]

# Variant display order and colors
VARIANT_ORDER = ['pymoo', 'classic', 'adaptive', 'diff', 'full']
VARIANT_COLORS = {
    'pymoo': '#1f77b4',      # Blue
    'classic': '#ff7f0e',    # Orange
    'adaptive': '#2ca02c',   # Green
    'diff': '#d62728',       # Red
    'full': '#9467bd',       # Purple
}

VARIANT_LABELS = {
    'pymoo': 'Pymoo',
    'classic': 'Classic',
    'adaptive': 'Adaptive',
    'diff': 'Diff',
    'full': 'Full',
}

# Algorithm family styling (style only; color stays variant-based)
ALGORITHM_ORDER = ['DE', 'SHADE', 'GA', 'PSO', 'CMAES']

FAMILY_LINESTYLES = {
    'DE': '-',
    'SHADE': '--',
    'GA': '-.',
    'PSO': ':',
    'CMAES': (0, (3, 1, 1, 1)),
}

# Publication-ish defaults (close to the "old" style)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# =============================================================================
# Helpers
# =============================================================================

def get_variant_type(variant_name: str) -> str:
    """Extract variant type from full name."""
    lower = variant_name.lower()
    for v in VARIANT_ORDER:
        if v in lower:
            return v
    return 'unknown'


def normalize_function_name(name: str) -> str:
    """Normalize function name to consistent format."""
    lower = name.lower().replace('_', '').replace('-', '')

    # CEC 2017 functions
    if 'cec2017' in lower:
        match = re.search(r'f?(\d+)', lower.replace('cec2017', ''))
        if match:
            num = int(match.group(1))
            return f'Cec2017_F{num}'

    # Classical functions
    name_map = {
        'sphere': 'Sphere',
        'ellipsoid': 'Ellipsoid',
        'rosenbrock': 'Rosenbrock',
        'rastrigin': 'Rastrigin',
        'ackley': 'Ackley',
        'griewank': 'Griewank',
        'schwefel': 'Schwefel',
        'levy': 'Levy',
        'michalewicz': 'Michalewicz',
        'alpine': 'Alpine',
        'zakharov': 'Zakharov',
        'weierstrass': 'Weierstrass',
        'salomon': 'Salomon',
        'styblinskitang': 'StyblinskiTang',
    }

    return name_map.get(lower, name)


def normalize_config_name(config: str) -> str:
    """Normalize config name to standard form."""
    config_map = {
        'classical': 'classic',
        'differentiable': 'diff',
    }
    return config_map.get(config.lower(), config.lower())


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    filepaths: List[str] = None,
    data_dir: str = None,
    filter_functions: List[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Load benchmark data from JSON files with verbose output.

    Returns:
        data: Dict[function][variant] = {
            'final_fitness': [...],
            'history': [[...], [...]],
            'seeds': [...],
        }
        metadata: Dict with n_var, xl, xu, max_evals
    """
    all_files = []

    if filepaths:
        for pattern in filepaths:
            all_files.extend(glob.glob(pattern))

    if data_dir:
        data_dir = Path(data_dir)
        all_files.extend(glob.glob(str(data_dir / "*.json")))

    all_files = sorted(list(set(all_files)))

    if not all_files:
        print("No JSON files found!")
        return {}, {}

    print(f"Found {len(all_files)} data file(s):")
    for f in all_files:
        print(f"  - {Path(f).name}")

    # Normalize filter functions if provided
    normalized_filter = None
    if filter_functions:
        normalized_filter = [normalize_function_name(f) for f in filter_functions]
        print(f"\nFiltering to functions: {normalized_filter}")

    data = defaultdict(lambda: defaultdict(lambda: {
        'final_fitness': [],
        'history': [],
        'seeds': [],
        # Optional: allow per-variant max_evals override if present later
        'max_evals': None,
    }))

    metadata = {}

    for filepath in all_files:
        fname = Path(filepath).name
        print(f"\nLoading {fname}...")

        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load {filepath}: {e}")
            continue

        # Metadata (first file wins)
        if not metadata:
            metadata = {
                'n_var': content.get('n_var', 10),
                'xl': content.get('xl', -100),
                'xu': content.get('xu', 100),
                'max_evals': content.get('max_evals', 50000),
                'n_runs': content.get('n_runs', 30),
            }

        algorithm = content.get('algorithm', 'Unknown')
        print(f"  Detected EvoGrad format for {algorithm}")

        for result in content.get('results', []):
            func_name = normalize_function_name(result.get('function', 'Unknown'))
            config = normalize_config_name(result.get('config', 'unknown'))

            if normalized_filter and func_name not in normalized_filter:
                continue

            variant = f"{algorithm} {config}"

            entry = data[func_name][variant]
            entry['final_fitness'].append(result.get('best_fitness', float('inf')))
            history = result.get('best_fitness_history', [])
            entry['history'].append(history)
            entry['seeds'].append(result.get('seed', -1))

            # Some formats store n_evals/max_evals per run; keep the largest seen
            run_max = result.get('n_evals') or result.get('max_evals')
            if isinstance(run_max, (int, float)) and run_max > 0:
                entry['max_evals'] = max(entry['max_evals'] or 0, int(run_max))

    # Print summary per function
    print("\nData summary by function:")
    for func_name in sorted(data.keys()):
        print(f"  {func_name}:")
        func_data = data[func_name]

        def sort_key(v):
            parts = v.split()
            alg = parts[0] if parts else ''
            var_type = parts[1] if len(parts) > 1 else ''
            alg_order = {'DE': 0, 'SHADE': 1, 'GA': 2, 'PSO': 3, 'CMAES': 4}.get(alg, 99)
            var_order = VARIANT_ORDER.index(var_type) if var_type in VARIANT_ORDER else 99
            return (alg_order, var_order)

        for variant in sorted(func_data.keys(), key=sort_key):
            var_data = func_data[variant]
            n_runs = len(var_data['final_fitness'])
            n_with_history = sum(1 for h in var_data['history'] if len(h) > 0)
            print(f"    - {variant}: {n_runs} runs ({n_with_history} with history)")

    print(f"\nLoaded data for {len(data)} functions")

    return dict(data), metadata


# =============================================================================
# Boxplot Figure (old style: grouped by family, shared legend)
# =============================================================================

def plot_boxplots(
    data: Dict,
    functions: List[str] = None,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
    log_scale: bool = True,
    title: str = None,
    show_individual_points: bool = True,
) -> plt.Figure:
    """
    Create grouped boxplots showing distribution of final fitness values.

    - Grouped by algorithm family (DE, SHADE, GA, PSO, CMAES)
    - Within each group: boxplots for variant types (pymoo, classic, adaptive, diff, full)
    - Consistent colors by variant type
    - Single shared legend at the bottom
    """
    if functions is None:
        functions = [f for f in FUNCTIONS + CEC2017_FUNCTIONS if f in data]
        for f in data.keys():
            if f not in functions:
                functions.append(f)
    else:
        functions = [f for f in functions if f in data]

    if not functions:
        print("No functions to plot!")
        return None

    n_funcs = len(functions)
    # Old script used up to 3 columns; keep plots less squashed.
    ncols = 3 if n_funcs > 4 else min(2, n_funcs)
    nrows = (n_funcs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_funcs == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for idx, func in enumerate(functions):
        ax = axes[idx]
        func_data = data.get(func, {})

        # Find present algorithms (preserve configured order)
        present_algorithms = {v.split()[0] for v in func_data.keys() if v}
        algorithms = [a for a in ALGORITHM_ORDER if a in present_algorithms]

        # Prepare grouped boxplots
        box_data = []
        colors = []
        positions = []

        pos = 0.0
        group_spacing = 2.0
        within_spacing = 0.7

        group_centers = []
        group_labels = []

        for alg in algorithms:
            family_start = pos
            n_in_group = 0

            for vtype in VARIANT_ORDER:
                variant = f"{alg} {vtype}"
                if variant in func_data:
                    values = func_data[variant].get('final_fitness', [])
                    values = [v for v in values if v is not None and np.isfinite(v)]
                    if values:
                        box_data.append(values)
                        colors.append(VARIANT_COLORS.get(vtype, '#333333'))
                        positions.append(pos)
                        n_in_group += 1
                pos += within_spacing

            if n_in_group > 0:
                group_center = family_start + (n_in_group - 1) * within_spacing / 2
                group_centers.append(group_center)
                group_labels.append(alg)

            pos += group_spacing - within_spacing

        if not box_data:
            ax.set_title(f"{func}\n(No data)")
            ax.set_axis_off()
            continue

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
            medianprops=dict(color='black', linewidth=1.5),
        )

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        if show_individual_points:
            # Light jittered scatter, same color as box
            for d, p, c in zip(box_data, positions, colors):
                jitter = np.random.normal(0, 0.06, len(d))
                ax.scatter(
                    p + jitter,
                    d,
                    alpha=0.3,
                    s=8,
                    c=[c],
                    edgecolors='none',
                    zorder=3,
                )

        ax.set_title(func, fontweight='bold', fontsize=13)
        ax.set_ylabel('Final Fitness')

        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontweight='bold')

        if log_scale:
            ax.set_yscale('log')

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # Light separators between families
        for i in range(1, len(group_centers)):
            sep_pos = (group_centers[i - 1] + group_centers[i]) / 2
            ax.axvline(x=sep_pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.25)

    # Hide unused axes
    for j in range(len(functions), len(axes)):
        axes[j].set_visible(False)

    legend_elements = [
        mpatches.Patch(
            facecolor=VARIANT_COLORS[v], alpha=0.7,
            edgecolor='black', linewidth=0.8,
            label=VARIANT_LABELS.get(v, v.capitalize())
        )
        for v in VARIANT_ORDER
        if v in VARIANT_COLORS
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        title='Variant Type',
        title_fontsize=10,
    )

    if title:
        fig.suptitle(title, fontweight='bold', y=0.98, fontsize=14)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"Saved boxplot figure to: {save_path}")

    return fig


# =============================================================================
# Convergence Figure (old style: shared legends, no markers)
# =============================================================================

def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample a curve for cleaner rendering on very long histories."""
    if max_points is None or max_points <= 0:
        return x, y
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def plot_convergence(
    data: Dict,
    metadata: Dict,
    functions: List[str] = None,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
    log_scale: bool = True,
    show_std: bool = True,
    max_evals: Optional[int] = None,
    title: str = None,
    max_points: int = 250,
) -> plt.Figure:
    """
    Create convergence curves showing average best fitness over evaluations.

    Old style:
    - Colors: per variant type
    - Line styles: per algorithm family
    - No markers
    - Two-part shared legend at the bottom
    """
    if functions is None:
        functions = [f for f in FUNCTIONS + CEC2017_FUNCTIONS if f in data]
        for f in data.keys():
            if f not in functions:
                functions.append(f)
    else:
        functions = [f for f in functions if f in data]

    if not functions:
        print("No functions to plot!")
        return None

    global_total_evals = max_evals or metadata.get('max_evals')

    n_funcs = len(functions)
    ncols = 3 if n_funcs > 4 else min(2, n_funcs)
    nrows = (n_funcs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_funcs == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    # Track empty histories (keep your diagnostics)
    empty_histories = []

    for idx, func in enumerate(functions):
        ax = axes[idx]
        func_data = data.get(func, {})

        present_algorithms = {v.split()[0] for v in func_data.keys() if v}
        algorithms = [a for a in ALGORITHM_ORDER if a in present_algorithms]

        for alg in algorithms:
            linestyle = FAMILY_LINESTYLES.get(alg, '-')

            for vtype in VARIANT_ORDER:
                variant = f"{alg} {vtype}"
                if variant not in func_data:
                    continue

                var_data = func_data[variant]
                histories_raw = var_data.get('history', [])
                if not histories_raw:
                    continue

                seeds = var_data.get('seeds', list(range(len(histories_raw))))

                valid_histories = []
                for run_idx, h in enumerate(histories_raw):
                    if not h:
                        seed = seeds[run_idx] if run_idx < len(seeds) else run_idx
                        empty_histories.append({
                            'function': func,
                            'variant': variant,
                            'run_index': run_idx,
                            'seed': seed,
                        })
                    else:
                        valid_histories.append(h)

                if not valid_histories:
                    continue

                max_len = max(len(h) for h in valid_histories)
                if max_len <= 0:
                    continue

                # Pad shorter histories with last value (converged)
                histories = []
                for h in valid_histories:
                    if len(h) < max_len:
                        padded = list(h) + [h[-1]] * (max_len - len(h))
                        histories.append(padded)
                    else:
                        histories.append(list(h)[:max_len])

                histories = np.array(histories, dtype=float)

                # Choose x-axis scale:
                # Prefer per-variant max_evals if present, else global_total_evals
                total_evals = var_data.get('max_evals') or global_total_evals
                if total_evals:
                    x = np.linspace(0, total_evals, max_len)
                else:
                    x = np.arange(max_len)

                # Compute mean/std robustly
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_f = np.nanmean(histories, axis=0)
                    std_f = np.nanstd(histories, axis=0)

                if np.all(np.isnan(mean_f)):
                    continue

                # Apply optional max_evals cut (after x computed)
                if max_evals is not None:
                    mask = x <= max_evals
                    x = x[mask]
                    mean_f = mean_f[mask]
                    std_f = std_f[mask]

                if len(x) == 0:
                    continue

                # Downsample for cleaner plotting
                x_ds, mean_ds = _downsample_xy(x, mean_f, max_points=max_points)
                _, std_ds = _downsample_xy(x, std_f, max_points=max_points)

                color = VARIANT_COLORS.get(vtype, '#333333')

                ax.plot(
                    x_ds, mean_ds,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    alpha=0.9,
                )

                if show_std and len(valid_histories) > 1:
                    ax.fill_between(
                        x_ds,
                        mean_ds - std_ds,
                        mean_ds + std_ds,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )

        ax.set_title(func, fontweight='bold', fontsize=13)
        ax.set_xlabel('Fitness Evaluations')
        ax.set_ylabel('Best Fitness')

        # Log scale only if strictly positive on this axis
        if log_scale:
            # Use current axis data range heuristic
            y_min, _ = ax.get_ylim()
            if y_min > 0:
                ax.set_yscale('log')
            else:
                ax.set_ylabel('Best Fitness (linear scale)')

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

    for j in range(len(functions), len(axes)):
        axes[j].set_visible(False)

    # Shared legends (old style)
    color_handles = [
        Line2D([0], [0], color=VARIANT_COLORS[v], linewidth=3,
               label=VARIANT_LABELS.get(v, v.capitalize()))
        for v in VARIANT_ORDER
    ]

    style_handles = [
        Line2D([0], [0], color='gray', linestyle=FAMILY_LINESTYLES.get(f, '-'),
               linewidth=2, label=f)
        for f in ALGORITHM_ORDER
    ]

    leg1 = fig.legend(
        handles=color_handles,
        loc='upper center',
        ncol=len(color_handles),
        bbox_to_anchor=(0.35, 0.02),
        title='Variant Type (Color)',
        title_fontsize=9,
        frameon=True,
    )

    leg2 = fig.legend(
        handles=style_handles,
        loc='upper center',
        ncol=len(style_handles),
        bbox_to_anchor=(0.78, 0.02),
        title='Algorithm Family (Style)',
        title_fontsize=9,
        frameon=True,
    )

    fig.add_artist(leg1)

    if title:
        fig.suptitle(title, fontweight='bold', y=0.98, fontsize=14)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Diagnostics for empty histories (keep exactly as in the new script)
    if empty_histories:
        print("\n" + "=" * 70)
        print("WARNING: Empty history arrays found (likely failed runs)")
        print("=" * 70)

        grouped = defaultdict(list)
        for item in empty_histories:
            key = (item['function'], item['variant'])
            grouped[key].append((item['run_index'], item['seed']))

        for (func, variant), runs in sorted(grouped.items()):
            alg = variant.split()[0] if ' ' in variant else variant
            var_type = variant.split()[1] if ' ' in variant else ''
            print(f"\n  Function:  {func}")
            print(f"  Algorithm: {alg}")
            print(f"  Variant:   {var_type}")
            print(f"  Failed runs: {len(runs)}")
            if len(runs) <= 5:
                for run_idx, seed in runs:
                    print(f"    - Run {run_idx}, Seed {seed}")

        print(f"\n  Total empty histories: {len(empty_histories)}")
        print("  These runs were skipped in convergence plots.")
        print("=" * 70 + "\n")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved convergence figure to: {save_path}")

    return fig


# =============================================================================
# Summary Table
# =============================================================================

def print_summary_table(data: Dict, metadata: Dict, functions: List[str] = None):
    """Print summary statistics table."""
    if functions is None:
        functions = [f for f in FUNCTIONS + CEC2017_FUNCTIONS if f in data]
        for f in data.keys():
            if f not in functions:
                functions.append(f)
    else:
        functions = [f for f in functions if f in data]

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"D={metadata.get('n_var', '?')}, bounds=[{metadata.get('xl', '?')}, {metadata.get('xu', '?')}], "
          f"max_evals={metadata.get('max_evals', '?')}")
    print("=" * 90)

    for func in functions:
        if func not in data:
            continue

        print(f"\n{func}")
        print("-" * 80)
        print(f"{'Variant':<20} {'Best':>12} {'Mean':>12} {'Std':>12} {'Runs':>6}")
        print("-" * 80)

        func_data = data[func]

        def sort_key(v):
            parts = v.split()
            alg = parts[0] if parts else ''
            var_type = parts[1] if len(parts) > 1 else ''
            alg_order = {'DE': 0, 'SHADE': 1, 'GA': 2, 'PSO': 3, 'CMAES': 4}.get(alg, 99)
            var_order = VARIANT_ORDER.index(var_type) if var_type in VARIANT_ORDER else 99
            return (alg_order, var_order)

        for variant in sorted(func_data.keys(), key=sort_key):
            var_data = func_data[variant]
            values = var_data.get('final_fitness', [])
            values = [v for v in values if v is not None and np.isfinite(v)]

            if values:
                best = np.min(values)
                mean = np.mean(values)
                std = np.std(values)
                n_runs = len(values)
                print(f"{variant:<20} {best:>12.4e} {mean:>12.4e} {std:>12.4e} {n_runs:>6}")

    print("\n" + "=" * 90)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoGrad Benchmark Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot all results from a directory
    python plot_benchmarks.py --data-dir results/ -o figures/

    # Plot specific JSON files
    python plot_benchmarks.py --data results/DE_*.json results/SHADE_*.json

    # Plot only specific functions
    python plot_benchmarks.py --data results/*.json --functions sphere rastrigin ackley

    # Plot CEC 2017 functions
    python plot_benchmarks.py --data results/*.json --functions cec2017_f1 cec2017_f11 cec2017_f21

    # High-resolution PNG output
    python plot_benchmarks.py --data results/*.json -o figures/ --format png --dpi 600

    # Linear scale instead of log scale
    python plot_benchmarks.py --data results/*.json -o figures/ --linear-scale
        """
    )

    parser.add_argument('--data', type=str, nargs='*', default=None,
                        help='JSON files or glob patterns to load')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing JSON result files')
    parser.add_argument('-o', '--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('-f', '--functions', type=str, nargs='+', default=None,
                        help='Specific functions to plot')
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'png', 'svg'],
                        help='Output format')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures')
    parser.add_argument('--no-boxplot', action='store_true',
                        help='Skip boxplot figure')
    parser.add_argument('--no-convergence', action='store_true',
                        help='Skip convergence figure')
    parser.add_argument('--linear-scale', action='store_true',
                        help='Use linear scale instead of log scale')
    parser.add_argument('--max-evals', type=int, default=None,
                        help='Override max evaluations for x-axis')
    parser.add_argument('--list-functions', action='store_true',
                        help='List available functions and exit')
    parser.add_argument('--no-summary', action='store_true',
                        help='Skip printing summary table')

    args = parser.parse_args()

    if not args.data and not args.data_dir:
        parser.error("Specify --data or --data-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams['savefig.dpi'] = args.dpi

    print("=" * 50)
    print("Loading benchmark data...")
    print("=" * 50)
    data, metadata = load_data(
        filepaths=args.data,
        data_dir=args.data_dir,
        filter_functions=args.functions,
    )

    if not data:
        print("No data loaded!")
        return

    if args.list_functions:
        print("\nAvailable functions in data:")
        for func in sorted(data.keys()):
            variants = list(data[func].keys())
            print(f"  {func}: {len(variants)} variants")
        return

    functions = None
    if args.functions:
        functions = [normalize_function_name(f) for f in args.functions]

    log_scale = not args.linear_scale

    if not args.no_summary:
        print_summary_table(data, metadata, functions)

    if not args.no_boxplot:
        print("\n" + "=" * 50)
        print("Generating boxplot figure...")
        print("=" * 50)
        n_runs = metadata.get('n_runs', 30)
        plot_boxplots(
            data,
            functions=functions,
            save_path=output_dir / f'benchmark_boxplots.{args.format}',
            log_scale=log_scale,
            title=f"Distribution of Final Fitness Values ({n_runs} runs)",
        )

    if not args.no_convergence:
        print("\n" + "=" * 50)
        print("Generating convergence figure...")
        print("=" * 50)
        n_runs = metadata.get('n_runs', 30)
        plot_convergence(
            data,
            metadata,
            functions=functions,
            save_path=output_dir / f'benchmark_convergence.{args.format}',
            log_scale=log_scale,
            max_evals=args.max_evals,
            title=f"Convergence Curves (Mean ± Std over {n_runs} runs)",
        )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
