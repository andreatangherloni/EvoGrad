import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from algorithms.cmaes import CMAES
from algorithms.de import DE
from algorithms.ga import GA
from algorithms.pso import PSO

from algorithms.evaluation_functions import evaluate
from algorithms.testing_functions import (
    base_funcs,
    FUNC_IDS,
    apply_shift_rot,
)

# ---------------------------------------------------------------------

def make_vectorised(f, name_suffix="sr"):
    def wrapped(x: torch.Tensor) -> torch.Tensor:
        y = f(x)
        return y.view(-1)
    wrapped.__name__ = getattr(f, "__name__", "f") + "_" + name_suffix
    return wrapped

# ---------------------------------------------------------------------

def plot_boxplots(results, algo_labels, output_dir="results/plots",
                  fig_name="boxplots_all_functions.png"):

    os.makedirs(output_dir, exist_ok=True)

    func_names = list(results.keys())
    n_funcs = len(func_names)
    n_algos = len(algo_labels)

    cols = 4
    rows = math.ceil(n_funcs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, fname in enumerate(func_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        data = [np.array(results[fname][alg]) for alg in algo_labels]
        positions = np.arange(1, n_algos + 1)

        bp = ax.boxplot(data, positions=positions, widths=0.6)

        means = [d.mean() for d in data]
        ax.scatter(positions, means, marker='o', color='red', zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(algo_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Best fitness")
        ax.set_title(fname, fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    for idx in range(n_funcs, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved boxplot grid to {out_path}")

# ---------------------------------------------------------------------

if __name__ == "__main__":

    D          = 30         
    pop_size   = 100
    max_evals  = 10000
    n_runs     = 10

    lower_bound = [-100.0] * D  
    upper_bound = [100.0] * D

    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo_defs = [
        ("CMAES", CMAES),
        ("DE",    DE),
        ("GA",    GA),
        ("PSO",   PSO),
    ]
    algo_labels = [a[0] for a in algo_defs]

    results = {name: {} for (_, name) in base_funcs}

    for f_basic, func_name in base_funcs:
        func_id = FUNC_IDS[func_name]

        print(f"\n=== {func_name} (shift+rot), D={D} ===")

        f_sr_raw = apply_shift_rot(f_basic, func_id=func_id, D=D)
        f_sr = make_vectorised(f_sr_raw, name_suffix="sr")

        for label, Alg in algo_defs:
            print(f"\nRunning {label} on {func_name}...")
            best_fitnesses = evaluate(
                algorithm=Alg,
                function=f_sr,
                dim=D,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                pop_size=pop_size,
                max_evals=max_evals,
                n_runs=n_runs,
                device=device,
                verbose=False,
                save_path=None,
            )
            results[func_name][label] = best_fitnesses

    plot_boxplots(results, algo_labels)
