import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------------------

def plot_boxplots_from_file(
    results_path,
    output_dir="results/plots",
    fig_name= None,
    cols=5,
):
    
    if fig_name is None:
        base = os.path.basename(results_path)           
        stem = os.path.splitext(base)[0]               
        fig_name = f"boxplots_{stem}.png"             

    # LOAD 
    with open(results_path, "rb") as f:
        data = pickle.load(f)

    results     = data["results"]   
    algo_labels = data["algo_labels"] 

    # BASIC INFO 
    func_names = sorted(results.keys())   
    n_funcs    = len(func_names)
    n_algos    = len(algo_labels)

    rows = math.ceil(n_funcs / cols)

    os.makedirs(output_dir, exist_ok=True)

    # FONT SETTINGS 
    plt.rcParams.update({
        "axes.titlesize": 20,   
        "axes.labelsize": 18,  
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)

    # PLOT
    for idx, fname in enumerate(func_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        data_func = [np.array(results[fname][alg]) for alg in algo_labels]
        positions = np.arange(1, n_algos + 1)

        ax.boxplot(data_func, positions=positions, widths=0.6)

        ax.set_xticks(positions)
        if r == rows - 1:
            ax.set_xticklabels(algo_labels, rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Best fitness")
        else:
            ax.set_ylabel("")

        ax.set_title(fname, pad=20)

        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    for idx in range(n_funcs, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved boxplot grid to {out_path}")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py path/to/results.pkl")
        sys.exit(1)

    results_path = sys.argv[1]
    plot_boxplots_from_file(results_path)
