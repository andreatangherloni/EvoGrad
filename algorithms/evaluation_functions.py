import torch
import numpy as np
import pickle
import os
from algorithms.minimize import minimize as min_diff

def evaluate(algorithm,
             function,
             dim = 100,
             lower_bound = -100, 
             upper_bound = 100,
             pop_size = 100,
             max_evals=1000,
             n_runs = 30, 
             device = None,
             verbose = True,
             save_path=None):
    
    best_fitnesses = []
    histories = []

    for i in range(n_runs):
        alg = algorithm(function, dim=dim, pop_size=pop_size, lower_bound=lower_bound, 
                        upper_bound=upper_bound, seed=i, device=device)
        
        min_diff(alg, max_evals=max_evals, verbose=verbose)

        best_f = alg.history["best_f"][-1]
        best_fitnesses.append(best_f)
        histories.append(alg.history)

    best_fitnesses_np = np.array(best_fitnesses)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/{algorithm.__name__}_{function.__name__}_history_{n_runs}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(histories, f)

    # if verbose:
    mean = np.mean(best_fitnesses_np)
    std = np.std(best_fitnesses_np)
    min_val = np.min(best_fitnesses_np)
    max_val = np.max(best_fitnesses_np)
    print(f'ABF for {algorithm.__name__} on {function.__name__}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')

    return best_fitnesses_np