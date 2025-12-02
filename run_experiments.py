import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import math
import pickle
import torch
import torch.nn as nn

from algorithms.cmaes import CMAES
from algorithms.de import DE
from algorithms.ga import GA
from algorithms.pso import PSO

from algorithms.non_diff_algorithms import (
    FSTPSO,
    GAStandard,
    DEStandard,
    CMAESStandard,
)

from algorithms.evaluation_functions import evaluate
from algorithms.testing_functions import (
    base_funcs,
    FUNC_IDS,
    apply_shift_rot,
)


def make_vectorised(f, name_suffix="sr"):
    def wrapped(x: torch.Tensor) -> torch.Tensor:
        y = f(x)
        return y.view(-1)
    wrapped.__name__ = getattr(f, "__name__", "f") + "_" + name_suffix
    return wrapped


def evaluate_baseline(algorithm, function, dim, lower_bound, upper_bound, 
                      pop_size, max_evals, n_runs, device, verbose=False):

    import numpy as np
    
    best_fitnesses = []
    
    for run in range(n_runs):
        if verbose:
            print(f"  Run {run+1}/{n_runs}...")
        
        alg = algorithm(
            function=function,
            dim=dim,
            pop_size=pop_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            max_evals=max_evals,
            device=device,
        )
        
        best_fitness = alg.run()
        best_fitnesses.append(best_fitness)
        
        if verbose:
            print(f"    Best fitness: {best_fitness:.4f}")
    
    best_fitnesses_np = np.array(best_fitnesses)
    mean_fitness = np.mean(best_fitnesses_np)
    std_fitness = np.std(best_fitnesses_np)
    min_fitness = np.min(best_fitnesses_np)
    max_fitness = np.max(best_fitnesses_np)
    
    algo_name = algorithm.__name__
    func_name = getattr(function, "__name__", "function")
    
    print(f"ABF for {algo_name} on {func_name}: {mean_fitness:.4f} ± {std_fitness:.4f} "
          f"(min: {min_fitness:.4f}, max: {max_fitness:.4f})")
    
    return best_fitnesses


if __name__ == "__main__":

    D          = 30
    pop_size   = 100
    max_evals  = 10000
    n_runs     = 10

    lower_bound = [-100.0] * D
    upper_bound = [100.0] * D

    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo_defs = [
        ("PSO_diff",   PSO, True),         
        ("PSO_fst",    FSTPSO, False),    

        ("GA_diff",    GA, True),
        ("GA_std",     GAStandard, False),

        ("DE_diff",    DE, True),
        ("DE_std",     DEStandard, False),

        ("CMAES_diff", CMAES, True),
        ("CMAES_std",  CMAESStandard, False),
    ]
    algo_labels = [a[0] for a in algo_defs]

    results = {name: {} for (_, name) in base_funcs}

    for f_basic, func_name in base_funcs:
        func_id = FUNC_IDS[func_name]

        print(f"\n=== {func_name} (shift+rot), D={D} ===")

        f_sr_raw = apply_shift_rot(f_basic, func_id=func_id, D=D)
        f_sr = make_vectorised(f_sr_raw, name_suffix="sr")

        for label, Alg, is_diff in algo_defs:
            print(f"\nRunning {label} on {func_name}...")
            
            if is_diff:
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
            else:
                best_fitnesses = evaluate_baseline(
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
                )
            
            results[func_name][label] = best_fitnesses

    out_dir = "results/raw"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"results_D{D}_evals{max_evals}.pkl")

    payload = {
        "D": D,
        "pop_size": pop_size,
        "max_evals": max_evals,
        "n_runs": n_runs,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "device": device,
        "algo_labels": algo_labels,
        "results": results,  
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n>>> Saved results to: {out_path}")