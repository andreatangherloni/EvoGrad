import torch
import numpy as np
import pickle
import os
from algorithms.minimize import minimize as min_diff

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA as GA_pymoo
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES as CMAES_pymoo
from pymoo.algorithms.soo.nonconvex.de import DE as DE_pymoo
from pymoo.algorithms.soo.nonconvex.pso import PSO as PSO_pymoo

def evaluate(algorithm,
             function,
             dim = 100,
             lower_bound = -100, 
             upper_bound = 100,
             pop_size = 100,
             max_evals=1000,
             n_runs = 30, 
             seed=None,
             device = None,
             verbose = True,
             save_path=None):
    
    best_fitnesses = []
    histories = []    

    for i in range(n_runs):
        
        if n_runs == 1 and seed is not None:
            run_seed = seed
        else:
            run_seed = i
        
        alg = algorithm(function, dim=dim, pop_size=pop_size, lower_bound=lower_bound, 
                        upper_bound=upper_bound, seed=run_seed, device=device)
        
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


class TorchProblemWrapper(Problem):
    def __init__(self, torch_func, dim=10, lower_bound=-10.0, upper_bound=5.0):
        if not isinstance(lower_bound, list):
            lower_bound = [lower_bound] * dim
        if not isinstance(upper_bound, list):
            upper_bound = [upper_bound] * dim
            
        super().__init__(n_var=dim, n_obj=1, n_constr=0, xl=np.array(lower_bound), xu=np.array(upper_bound))
        self.torch_func = torch_func
        self.func_name = torch_func.__name__
        
    
    def _evaluate(self, x, out, *args, **kwargs):
        x_torch = torch.tensor(x, dtype=torch.float32)
        
        result = self.torch_func(x_torch)
        
        if result.is_cuda:
            result = result.cpu()
            
        result_np = np.array(result.tolist())
        
        out["F"] = result_np

def evaluate_pso(function,
                 dim = 100,
                 lower_bound = -100, 
                 upper_bound = 100,
                 pop_size = 100,
                 max_evals=1000,
                 n_runs = 30,
                 seed=None,
                 verbose = True,
                 save_path=None):
    
    best_fitnesses = []
    histories = []

    for i in range(n_runs):
        
        if n_runs == 1 and seed is not None:
            run_seed = seed
        else:
            run_seed = i
        
        problem = TorchProblemWrapper(function,
                                      dim=dim, 
                                      lower_bound=lower_bound, 
                                      upper_bound=upper_bound)
        
        run_history = []
        def callback(alg):
            run_history.append(alg.opt.get("F").min())
        
        algorithm = PSO_pymoo(pop_size=pop_size,
                              sampling=FloatRandomSampling(),
                              seed=run_seed)

        res = minimize(problem,
                       algorithm,
                       ('n_evals', max_evals),
                       seed=run_seed,
                       verbose=False,
                       callback=callback)
        
        best_fitnesses.append(res.F[0])
        histories.append(run_history)

    best_fitnesses_np = np.array(best_fitnesses)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/PSO_{function.__name__}_history_{n_runs}.pkl"
        with open(filename, "wb") as f:
            formatted_histories = [{"best_f": hist} for hist in histories]
            pickle.dump(formatted_histories, f)

    mean = np.mean(best_fitnesses_np)
    std = np.std(best_fitnesses_np)
    min_val = np.min(best_fitnesses_np)
    max_val = np.max(best_fitnesses_np)

    if verbose:
        print(f'ABF for PSO on {function.__name__}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')
        print(f'  Number of runs: {n_runs}')

    return best_fitnesses_np

def evaluate_ga(function,
                dim = 100,
                lower_bound = -100, 
                upper_bound = 100,
                pop_size = 100,
                max_evals=1000,
                n_runs = 30, 
                seed=None,
                verbose = True,
                save_path=None):
    
    best_fitnesses = []
    histories = []

    for i in range(n_runs):
        
        if n_runs == 1 and seed is not None:
            run_seed = seed
        else:
            run_seed = i
        
        problem = TorchProblemWrapper(function,
                                      dim=dim, 
                                      lower_bound=lower_bound, 
                                      upper_bound=upper_bound)
        
        run_history = []
        def callback(alg):
            run_history.append(alg.opt.get("F").min())
            
        algorithm = GA_pymoo(pop_size=pop_size,
                             sampling=FloatRandomSampling(),
                             crossover=SBX(prob=0.9, eta=15),
                             mutation=PM(eta=20),
                             eliminate_duplicates=True,
                             seed=run_seed)

        res = minimize(problem,
                       algorithm,
                       ('n_evals', max_evals),
                       seed=run_seed,
                       verbose=False,
                       callback=callback)
        
        best_fitnesses.append(res.F[0])
        histories.append(run_history)

    best_fitnesses_np = np.array(best_fitnesses)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/GA_{function.__name__}_history_{n_runs}.pkl"
        with open(filename, "wb") as f:
            formatted_histories = [{"best_f": hist} for hist in histories]
            pickle.dump(formatted_histories, f)

    mean = np.mean(best_fitnesses_np)
    std = np.std(best_fitnesses_np)
    min_val = np.min(best_fitnesses_np)
    max_val = np.max(best_fitnesses_np)

    if verbose:
        print(f'ABF for GA on {function.__name__}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')
        print(f'  Number of runs: {n_runs}')

    return best_fitnesses_np

def evaluate_de(function,
                dim = 100,
                lower_bound = -100, 
                upper_bound = 100,
                pop_size = 100,
                max_evals=1000,
                n_runs = 30, 
                seed=None,
                verbose = True,
                save_path=None):
    
    best_fitnesses = []
    histories = []

    for i in range(n_runs):
        
        if n_runs == 1 and seed is not None:
            run_seed = seed
        else:
            run_seed = i
        
        problem = TorchProblemWrapper(function,
                                      dim=dim, 
                                      lower_bound=lower_bound, 
                                      upper_bound=upper_bound)
        
        run_history = []
        def callback(alg):
            run_history.append(alg.opt.get("F").min())
        
        
        algorithm = DE_pymoo(pop_size=pop_size,
                             sampling=FloatRandomSampling(),
                             variant="DE/rand/1/bin", 
                             CR=0.9,
                             dither="vector",
                             jitter=False,
                             seed=run_seed
                             )

        res = minimize(problem,
                       algorithm,
                       ('n_evals', max_evals),
                       seed=run_seed,
                       verbose=False,
                       callback=callback)
        
        best_fitnesses.append(res.F[0])
        histories.append(run_history)

    best_fitnesses_np = np.array(best_fitnesses)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/DE_{function.__name__}_history_{n_runs}.pkl"
        with open(filename, "wb") as f:
            formatted_histories = [{"best_f": hist} for hist in histories]
            pickle.dump(formatted_histories, f)

    mean = np.mean(best_fitnesses_np)
    std = np.std(best_fitnesses_np)
    min_val = np.min(best_fitnesses_np)
    max_val = np.max(best_fitnesses_np)

    if verbose:
        print(f'ABF for DE on {function.__name__}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')
        print(f'  Number of runs: {n_runs}')

    return best_fitnesses_np

def evaluate_cmaes(function,
                   dim = 100,
                   lower_bound = -100, 
                   upper_bound = 100,
                   pop_size = 100,
                   sigma=None,
                   x0=None,
                   max_evals=1000,
                   n_runs = 30, 
                   seed=None,
                   verbose = True,
                   save_path=None):
    
    best_fitnesses = []
    histories = []
    
    lb = np.array([lower_bound]*dim)
    ub = np.array([upper_bound]*dim)
    
    if sigma is None:
        sigma = np.mean(ub - lb) / 3

    if x0 is None:
        x0 = lb + (ub - lb) / 2

    for i in range(n_runs):
        
        if n_runs == 1 and seed is not None:
            run_seed = seed
        else:
            run_seed = i
        
        problem = TorchProblemWrapper(function,
                                      dim=dim, 
                                      lower_bound=lower_bound, 
                                      upper_bound=upper_bound)
        
        run_history = []
        def callback(alg):
            run_history.append(alg.opt.get("F").min())
        
        algorithm = algorithm = CMAES_pymoo(pop_size=pop_size,
                                            seed=run_seed)
        res = minimize(problem,
                       algorithm,
                       ('n_evals', max_evals),
                       seed=run_seed,
                       verbose=verbose,
                       callback=callback)
        
        best_fitnesses.append(res.F[0])
        histories.append(run_history)

    best_fitnesses_np = np.array(best_fitnesses)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/CMAES_{function.__name__}_history_{n_runs}.pkl"
        with open(filename, "wb") as f:
            formatted_histories = [{"best_f": hist} for hist in histories]
            pickle.dump(formatted_histories, f)

    mean = np.mean(best_fitnesses_np)
    std = np.std(best_fitnesses_np)
    min_val = np.min(best_fitnesses_np)
    max_val = np.max(best_fitnesses_np)

    if verbose:
        print(f'ABF for CMAES on {function.__name__}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})')
        print(f'  Number of runs: {n_runs}')

    return best_fitnesses_np