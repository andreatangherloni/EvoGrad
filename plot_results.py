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
    fig_name=None,
    cols=5,
    log_scale=False,   
):
    
    if fig_name is None:
        base = os.path.basename(results_path)           
        stem = os.path.splitext(base)[0]               
        log_suffix = "_log" if log_scale else ""
        fig_name = f"boxplots{log_suffix}_grouped_{stem}.png"             

    # LOAD 
    try:
        with open(results_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    results     = data["results"]   
    all_algo_labels = data["algo_labels"] 

    label_mapping = {
        "PSO_diff": "PSO_diff", 
        "PSO_fst": "PSO_standard",    
        "GA_diff": "GA_diff",
        "GA_std": "GA_standard",
        "DE_diff": "DE_diff",
        "DE_std": "DE_standard",
        "CMAES_diff": "CMAES_diff",
        "CMAES_std": "CMAES_standard",
    }
    
    main_algos = ["PSO", "CMAES", "GA", "DE"]
    version_types = ["standard", "diff"]
    
    group_size = len(version_types)
    group_width = 0.8 
    box_width = group_width / group_size

    color_map = {
        "standard": 'skyblue', 
        "diff": 'lightcoral'
    }
    
    target_plot_labels = []
    expected_positions = []
    
    major_positions = np.arange(1, len(main_algos) + 1) * 2
    
    for i, algo in enumerate(main_algos):
        major_pos = major_positions[i]
        start_offset = - (group_width / 2) + (box_width / 2)
        
        for j, version in enumerate(version_types):
            plot_label = f"{algo}_{version}"
            position = major_pos + start_offset + j * box_width
            
            target_plot_labels.append(plot_label)
            expected_positions.append(position)

    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    data_keys_to_plot = []
    plot_positions = []
    plot_labels_for_color = []
    
    for plot_label, pos in zip(target_plot_labels, expected_positions):
        data_key = reverse_mapping.get(plot_label) 
        
        if data_key and data_key in all_algo_labels:
            data_keys_to_plot.append(data_key)
            plot_positions.append(pos)
            plot_labels_for_color.append(plot_label) 
            
    if not data_keys_to_plot:
        print("No valid data available to plot after mapping and filtering.")
        return

    missing_data_keys = set(reverse_mapping.values()) - set(all_algo_labels)
    if missing_data_keys:
        missing_plot_labels = [label_mapping[k] for k in missing_data_keys if k in label_mapping]
        print(f"Warning: Data missing for algorithms: {missing_plot_labels}")
        print(f"Falling back to plotting only available data ({len(data_keys_to_plot)} algorithms).")

    # BASIC INFO 
    func_names = sorted(results.keys())   
    n_funcs    = len(func_names)
    
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

    y_label = "Best fitness"
    if log_scale:
        y_label += " (Log Scale)"

    # PLOT
    for idx, fname in enumerate(func_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        data_func = [np.array(results[fname][alg_key]) for alg_key in data_keys_to_plot]
        
        if log_scale:
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        
        bplot = ax.boxplot(
            data_func,
            positions=plot_positions,
            widths=box_width * 0.9, 
            patch_artist=True,      
            showfliers=False       
        )

        for i, patch in enumerate(bplot['boxes']):
            consistent_label = plot_labels_for_color[i]
            
            version = consistent_label.split('_')[-1] 
            
            patch.set_facecolor(color_map.get(version, 'gray'))
            patch.set_alpha(0.7) 

        ax.set_xticks(major_positions)
        
        if r == rows - 1:
            ax.set_xticklabels(main_algos)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel("")

        ax.set_title(fname, pad=20)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
    if rows > 0 and cols > 0 and color_map:
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=color_map[version], alpha=0.7)
            for version in version_types if version in color_map
        ]
        
        if legend_handles:
            fig.legend(
                legend_handles, 
                version_types, 
                title="Algorithm Version", 
                loc='upper right', 
                bbox_to_anchor=(0.99, 0.98),
                fontsize='medium', 
                title_fontsize='large'
            )


    for idx in range(n_funcs, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, fig_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved boxplot grid ({'log' if log_scale else 'linear'} scale, grouped) to {out_path}")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path/to/results.pkl> [log]")
        sys.exit(1)

    results_path = sys.argv[1]
    
    is_log_scale = False
    if len(sys.argv) > 2:
        log_arg = sys.argv[2].lower()
        if log_arg in ("log", "true", "t", "1"):
            is_log_scale = True
            
    plot_boxplots_from_file(results_path, log_scale=is_log_scale)