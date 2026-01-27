#!/usr/bin/env python3
"""
EvoGrad Parallel Benchmark Suite

Runs EvoGrad algorithms in four configurations:
    1. Classical (adaptive=False, differentiable=False)
    2. Differentiable (adaptive=False, differentiable=True)
    3. Adaptive (adaptive=True, differentiable=False)
    4. Full (adaptive=True, differentiable=True)

Also supports baselines (included by default when running evolutionary algorithms):
    - pymoo: Reference implementation baseline (--no_pymoo to disable)
    - Adam: Gradient-based optimizer baseline (--no_adam to disable)

Function categories:
    - Classical: Standard test functions (Sphere, Rosenbrock, Rastrigin, etc.)
    - CEC 2017: Competition benchmark suite (F1-F30)
    - Smoothed Funnel: Multi-basin problems designed for differentiable EAs

Usage:
    python run_benchmark.py --algorithm DE --n_runs 30
    python run_benchmark.py --algorithm SHADE --suite cec2017 --n_var 10
    python run_benchmark.py --algorithm GA -D 10 --xl -100 --xu 100 --max_evals 50000
    
    # Run only Adam optimizer
    python run_benchmark.py -a ADAM -s quick -D 10 -r 30
    python run_benchmark.py -a ADAM -s standard -D 10 --adam_lr 0.01
    
    # Run DE with all baselines (pymoo + Adam)
    python run_benchmark.py -a DE -s quick -D 10 -r 5
    
    # Run DE without Adam baseline
    python run_benchmark.py -a DE -s quick -D 10 -r 5 --no_adam
    
    # Run DE without pymoo baseline
    python run_benchmark.py -a DE -s quick -D 10 -r 5 --no_pymoo
    
    # CEC 2017 examples
    python run_benchmark.py -a DE -s cec2017_simple -D 10 -r 5    # F1-F10
    python run_benchmark.py -a DE -s cec2017_hybrid -D 10 -r 5    # F11-F20
    python run_benchmark.py -a DE -s cec2017 -D 10 -r 5           # All F1-F30
    
    # Smoothed funnel functions (designed for differentiable EAs)
    python run_benchmark.py -a DE -s funnel -D 10 -r 30           # All funnel functions
    python run_benchmark.py -a DE -f multibasinrastrigin -D 10    # Single funnel function
"""

from __future__ import annotations

# =============================================================================
# CRITICAL: Set thread limits BEFORE importing torch/numpy
# This prevents thread oversubscription on Linux with multiprocessing
# =============================================================================
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import json
import sys
import time
import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

warnings.filterwarnings('ignore')

# =============================================================================
# Path Setup - Make sure we can import both 'functions' and 'evograd'
# =============================================================================

# Directory containing this script (benchmarks/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent of benchmarks/ is evograd/, parent of evograd/ contains evograd package
EVOGRAD_PARENT = SCRIPT_DIR.parent.parent  # Go up two levels to find evograd package

# Add paths for imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))  # For 'functions' subpackage
if str(EVOGRAD_PARENT) not in sys.path:
    sys.path.insert(0, str(EVOGRAD_PARENT))  # For 'evograd' package

torch.set_num_threads(1)  # Limit PyTorch threads per process

# Import benchmark functions from the functions subpackage
from functions import (
    CLASSICAL_FUNCTIONS,
    ALL_FUNCTIONS,
)

# Import smoothed funnel functions
SMOOTHED_FUNNEL_AVAILABLE = False
SMOOTHED_FUNNEL_FUNCTIONS = {}
try:
    from functions.smoothed_funnel import (
        MultiBasinRastrigin,
        MultiBasinRosenbrock,
        DeceptiveLandscape,
        SMOOTHED_FUNNEL_FUNCTIONS as _FUNNEL_FUNCS,
    )
    SMOOTHED_FUNNEL_FUNCTIONS = _FUNNEL_FUNCS
    SMOOTHED_FUNNEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Smoothed funnel functions not available: {e}")

# Import CEC 2017 functions
CEC2017_AVAILABLE = False
CEC2017_FUNCTIONS = {}
try:
    from functions.cec2017 import (
        CEC2017_FUNCTIONS as _CEC2017_FUNCS,
        ALL_CEC2017_CLASSES,
        get_function as get_cec2017_function,
    )
    CEC2017_FUNCTIONS = _CEC2017_FUNCS
    CEC2017_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CEC 2017 functions not available: {e}")

# Import pymoo (optional)
try:
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

# Import EvoGrad components
EVOGRAD_AVAILABLE = False
EVOGRAD_IMPORT_ERROR = ""

try:
    from evograd.core.problem import Problem
    from evograd.core.termination import MaxEvaluations
    from evograd.core.minimize import minimize
    
    from evograd.algorithms.de import DE
    from evograd.algorithms.shade import SHADE
    from evograd.algorithms.pso import PSO
    from evograd.algorithms.ga import GA
    from evograd.algorithms.cmaes import CMAES
    
    from evograd.operators.selection import TournamentSelection
    from evograd.operators.crossover import SBXCrossover
    from evograd.operators.mutation import PolynomialMutation
    
    EVOGRAD_AVAILABLE = True
except ImportError as e:
    EVOGRAD_IMPORT_ERROR = str(e)


# =============================================================================
# Function Suites
# =============================================================================

# Build CEC 2017 function lists
CEC2017_SIMPLE = [f"cec2017_f{i}" for i in range(1, 11)]      # F1-F10
CEC2017_HYBRID = [f"cec2017_f{i}" for i in range(11, 21)]     # F11-F20
CEC2017_COMPOSITION = [f"cec2017_f{i}" for i in range(21, 31)] # F21-F30
CEC2017_ALL = [f"cec2017_f{i}" for i in range(1, 31) if i != 2]          # F1-F30

SUITES = {
    # Classical functions
    "classical": list(CLASSICAL_FUNCTIONS.keys()),
    "standard": ["sphere", "ellipsoid", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel", "levy"],
    "quick": ["sphere", "rastrigin", "ackley"],
    
    # CEC 2017 suites
    "cec2017": CEC2017_ALL,
    "cec2017_simple": CEC2017_SIMPLE,
    "cec2017_hybrid": CEC2017_HYBRID,
    "cec2017_composition": CEC2017_COMPOSITION,
    "cec2017_quick": ["cec2017_f1", "cec2017_f11", "cec2017_f21"],  # One from each category
    
    # Smoothed funnel functions (designed for differentiable EAs)
    "funnel": ["multibasinrastrigin", "multibasinrosenbrock", "deceptivelandscape"],
    "funnel_quick": ["multibasinrastrigin"],
    
    # Mixed suites
    "all": list(CLASSICAL_FUNCTIONS.keys()) + CEC2017_ALL,
    "all_with_funnel": list(CLASSICAL_FUNCTIONS.keys()) + CEC2017_ALL + ["multibasinrastrigin", "multibasinrosenbrock", "deceptivelandscape"],
}


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class RunResult:
    """Result of a single optimization run."""
    algorithm: str
    config: str
    function: str
    n_var: int
    seed: int
    best_fitness_history: List[float]
    best_fitness: float
    n_evals: int
    n_gen: int
    wall_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Collection of benchmark results."""
    algorithm: str
    timestamp: str
    device: str
    n_var: int
    xl: float
    xu: float
    max_evals: int
    n_runs: int
    results: List[RunResult] = field(default_factory=list)
    
    def add_result(self, result: RunResult):
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get summary statistics grouped by function and config."""
        summary = {}
        
        for result in self.results:
            func_name = result.function
            config = result.config
            
            if func_name not in summary:
                summary[func_name] = {}
            if config not in summary[func_name]:
                summary[func_name][config] = {"fitness_values": [], "times": [], "success_count": 0}
            
            summary[func_name][config]["fitness_values"].append(result.best_fitness)
            summary[func_name][config]["times"].append(result.wall_time)
            if result.success:
                summary[func_name][config]["success_count"] += 1
        
        for func_name in summary:
            for config in summary[func_name]:
                data = summary[func_name][config]
                fitness = np.array(data["fitness_values"])
                times = np.array(data["times"])
                n = len(fitness)
                valid = fitness[np.isfinite(fitness)]
                
                summary[func_name][config] = {
                    "best": float(np.min(valid)) if len(valid) > 0 else float('inf'),
                    "mean": float(np.mean(valid)) if len(valid) > 0 else float('inf'),
                    "std": float(np.std(valid)) if len(valid) > 0 else 0,
                    "median": float(np.median(valid)) if len(valid) > 0 else float('inf'),
                    "worst": float(np.max(valid)) if len(valid) > 0 else float('inf'),
                    "mean_time": float(np.mean(times)) if n > 0 else 0,
                    "success_rate": data["success_count"] / n if n > 0 else 0,
                    "n_runs": n,
                }
        
        return summary
    
    def save(self, output_dir: Path):
        """Save results to JSON and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"{self.algorithm}_{self.timestamp}"
        
        # JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "algorithm": self.algorithm,
                "timestamp": self.timestamp,
                "device": self.device,
                "n_var": self.n_var,
                "xl": self.xl,
                "xu": self.xu,
                "max_evals": self.max_evals,
                "n_runs": self.n_runs,
                "results": [r.to_dict() for r in self.results],
            }, f, indent=2)
        
        # CSV
        csv_path = output_dir / f"{base_name}_summary.csv"
        summary = self.get_summary()
        
        with open(csv_path, 'w') as f:
            f.write("function,config,best,mean,std,median,worst,mean_time,success_rate,n_runs\n")
            for func_name in sorted(summary.keys()):
                for config in sorted(summary[func_name].keys()):
                    s = summary[func_name][config]
                    f.write(f"{func_name},{config},{s['best']:.6e},{s['mean']:.6e},{s['std']:.6e},"
                            f"{s['median']:.6e},{s['worst']:.6e},{s['mean_time']:.2f},"
                            f"{s['success_rate']:.2f},{s['n_runs']}\n")
        
        print(f"\nResults saved to:\n  - {json_path}\n  - {csv_path}")


# =============================================================================
# Algorithm Factory
# =============================================================================

def create_evograd_algorithm(algorithm_name: str, config: str, pop_size: int, device: str, seed: int):
    """Create EvoGrad algorithm with specified configuration."""
    differentiable = config in ("differentiable", "full")
    adaptive = config in ("adaptive", "full")
    
    name = algorithm_name.upper()
    
    if name == "DE":
        return DE(pop_size=pop_size, F=0.5, CR=0.9, adaptive=adaptive, differentiable=differentiable, device=device, seed=seed)
    
    elif name == "SHADE":
        return SHADE(pop_size=pop_size, adaptive=adaptive, differentiable=differentiable, device=device, seed=seed)
    
    elif name == "PSO":
        return PSO(pop_size=pop_size, w=0.7, c1=1.5, c2=1.5, adaptive=adaptive, differentiable=differentiable, device=device, seed=seed)
    
    elif name == "GA":
        return GA(
            pop_size=pop_size,
            selection=TournamentSelection(tournament_size=3, adaptive=adaptive),
            crossover=SBXCrossover(eta=15, prob=0.9, adaptive=adaptive),
            mutation=PolynomialMutation(eta=20, adaptive=adaptive),
            differentiable=differentiable,
            device=device,
            seed=seed,
        )
    
    elif name == "CMAES":
        return CMAES(pop_size=pop_size, sigma=0.5, adaptive=adaptive, differentiable=differentiable, device=device, seed=seed, bipop=True, restarts=9)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def create_pymoo_algorithm(algorithm_name: str, pop_size: int):
    """Create pymoo algorithm for baseline comparison."""
    name = algorithm_name.upper()
    
    if name == "DE":
        from pymoo.algorithms.soo.nonconvex.de import DE as PymooDE
        return PymooDE(pop_size=pop_size)
    elif name == "SHADE":
        # Pymoo doesn't have SHADE, use DE as baseline
        from pymoo.algorithms.soo.nonconvex.de import DE as PymooDE
        return PymooDE(pop_size=pop_size)
    elif name == "PSO":
        from pymoo.algorithms.soo.nonconvex.pso import PSO as PymooPSO
        return PymooPSO(pop_size=pop_size)
    elif name == "GA":
        from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA
        return PymooGA(pop_size=pop_size)
    elif name == "CMAES":
        from pymoo.algorithms.soo.nonconvex.cmaes import CMAES as PymooCMAES
        return PymooCMAES()
    else:
        raise ValueError(f"Unknown pymoo algorithm: {algorithm_name}")


# =============================================================================
# Function Instance Creation
# =============================================================================

def get_function_instance(func_name: str, n_var: int, xl: float, xu: float):
    """
    Create a function instance from name.
    
    Handles classical functions, CEC 2017 functions, and smoothed funnel functions.
    CEC 2017 functions use fixed bounds [-100, 100].
    Smoothed funnel functions use default bounds [-5, 5].
    """
    # Check if it's a CEC 2017 function
    if func_name.startswith("cec2017_f"):
        if not CEC2017_AVAILABLE:
            raise ImportError("CEC 2017 functions not available")
        
        # Extract function number
        func_num = int(func_name.replace("cec2017_f", ""))
        
        # CEC 2017 uses fixed bounds [-100, 100]
        func_instance = get_cec2017_function(func_num, n_var=n_var)
        return func_instance
    
    # Check if it's a smoothed funnel function
    elif func_name in SMOOTHED_FUNNEL_FUNCTIONS:
        if not SMOOTHED_FUNNEL_AVAILABLE:
            raise ImportError("Smoothed funnel functions not available")
        
        func_class = SMOOTHED_FUNNEL_FUNCTIONS[func_name]
        # Smoothed funnels use [-5, 5] by default, but respect user bounds
        return func_class(n_var=n_var, xl=xl, xu=xu)
    
    # Classical function
    elif func_name in CLASSICAL_FUNCTIONS:
        func_class = CLASSICAL_FUNCTIONS[func_name]
        return func_class(n_var=n_var, xl=xl, xu=xu)
    
    # Check ALL_FUNCTIONS as fallback
    elif func_name in ALL_FUNCTIONS:
        func_class = ALL_FUNCTIONS[func_name]
        return func_class(n_var=n_var, xl=xl, xu=xu)
    
    else:
        raise ValueError(f"Unknown function: {func_name}")


# =============================================================================
# Adam Optimizer Baseline
# =============================================================================

def run_adam_population(
    func_instance,
    n_var: int,
    xl: float,
    xu: float,
    pop_size: int,
    max_evals: int,
    seed: int,
    device: str,
    lr: float = 0.05,
    b1: float = 0.9,
    b2: float = 0.999,
    wd: float = 0.0,
) -> Tuple[float, List[float], int, int, Optional[Tensor]]:
    """
    Run Adam optimizer on benchmark function.
    
    Optimizes directly in [xl, xu] space using projected gradient
    descent. After each Adam update, values are clamped to maintain
    feasibility. Uses multiple parallel "individuals" for fair 
    comparison with population-based methods.
    
    Args:
        func_instance: Benchmark function instance (callable).
        n_var: Number of variables.
        xl: Lower bound.
        xu: Upper bound.
        pop_size: Number of parallel solutions (for fair eval count comparison).
        max_evals: Maximum fitness evaluations.
        seed: Random seed.
        device: Torch device string.
        lr: Learning rate.
        b1: Adam beta1 parameter.
        b2: Adam beta2 parameter.
        wd: Weight decay (L2 regularisation).
        
    Returns:
        Tuple of (best_fitness, fitness_history, n_evaluations, n_generations, best_solution).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    
    # Initialize population uniformly in [xl, xu] (same as evolutionary algorithms)
    x_init = torch.rand(pop_size, n_var, device=dev, dtype=torch.float32) * (xu - xl) + xl
    x = x_init.clone().requires_grad_(True)
    
    opt = torch.optim.Adam([x], lr=lr, betas=(b1, b2), weight_decay=wd)

    best = float("inf")
    best_solution = None
    hist = []
    evals = 0
    n_gen = 0

    while evals < max_evals:
        opt.zero_grad()
        
        # Evaluate fitness
        f = func_instance(x)
        
        # Handle different output shapes
        if f.dim() > 1:
            f = f.squeeze()

        # Backpropagate mean loss across population
        loss = f.mean()
        loss.backward()
        opt.step()
        
        # Project back to feasible region [xl, xu]
        with torch.no_grad():
            x.clamp_(xl, xu)

            # Track best solution
            min_idx = f.argmin()
            min_val = float(f[min_idx])
            if min_val < best:
                best = min_val
                best_solution = x[min_idx].detach().clone()

        hist.append(best)
        evals += pop_size
        n_gen += 1

    return best, hist, evals, n_gen, best_solution


# =============================================================================
# Single Run Functions
# =============================================================================

def _worker_init():
    """Initialize worker process with thread limits."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    import torch
    torch.set_num_threads(1)


def run_evograd_single(
    algorithm_name: str,
    config: str,
    func_name: str,
    n_var: int,
    xl: float,
    xu: float,
    seed: int,
    max_evals: int,
    pop_size: int,
    lr_pop: float,
    lr_hyper: float,
    grad_clip_pop: float,
    grad_clip_hyper: float,
    device: str,
) -> RunResult:
    """Run a single EvoGrad optimization."""
    _worker_init()  # Ensure thread limits in worker
    
    if not EVOGRAD_AVAILABLE:
        return RunResult(
            algorithm=algorithm_name, config=config, function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=[], best_fitness=float('inf'),
            n_evals=0, n_gen=0, wall_time=0.0, success=False,
            error_message=f"EvoGrad not available: {EVOGRAD_IMPORT_ERROR}",
        )
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    try:
        # Get function instance
        func_instance = get_function_instance(func_name, n_var, xl, xu)
        
        # Get actual bounds from the instance
        actual_xl = func_instance.xl[0].item()
        actual_xu = func_instance.xu[0].item()
        
        # Create Problem with the function's __call__ as objective
        problem = Problem(
            objective=func_instance,
            n_var=n_var,
            xl=actual_xl,
            xu=actual_xu,
            device=device,
        )
        
        # Create Algorithm
        algorithm = create_evograd_algorithm(algorithm_name, config, pop_size, device, seed)
        
        # Run
        result = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=MaxEvaluations(max_evals),
            seed=seed,
            verbose=False,
            save_history=True,
            lr_pop=lr_pop,
            lr_hyper=lr_hyper,
            grad_clip_pop=grad_clip_pop,
            grad_clip_hyper=grad_clip_hyper,
        )
        
        return RunResult(
            algorithm=algorithm_name, config=config, function=func_name,
            n_var=n_var, seed=seed,
            best_fitness_history=result.history.get("best_fitness", []),
            best_fitness=result.best_fitness,
            n_evals=result.n_evals,
            n_gen=result.n_gen,
            wall_time=time.time() - start_time,
            success=True,
        )
        
    except Exception as e:
        import traceback
        return RunResult(
            algorithm=algorithm_name, config=config, function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=[], best_fitness=float('inf'),
            n_evals=0, n_gen=0, wall_time=time.time() - start_time,
            success=False, error_message=f"{str(e)}\n{traceback.format_exc()}",
        )


def run_pymoo_single(
    algorithm_name: str,
    func_name: str,
    n_var: int,
    xl: float,
    xu: float,
    seed: int,
    max_evals: int,
    pop_size: int,
) -> RunResult:
    """Run a single pymoo optimization."""
    _worker_init()  # Ensure thread limits in worker
    
    if not PYMOO_AVAILABLE:
        return RunResult(
            algorithm=algorithm_name, config="pymoo", function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=[], best_fitness=float('inf'),
            n_evals=0, n_gen=0, wall_time=0.0, success=False,
            error_message="pymoo not available",
        )
    
    np.random.seed(seed)
    start_time = time.time()
    
    try:
        # Get function instance
        func_instance = get_function_instance(func_name, n_var, xl, xu)
        
        # Get actual bounds from the instance
        actual_xl = func_instance.xl[0].item()
        actual_xu = func_instance.xu[0].item()
        
        # Pymoo problem wrapper
        class Wrapper(PymooProblem):
            def __init__(self):
                super().__init__(n_var=n_var, n_obj=1, n_constr=0,
                                 xl=np.full(n_var, actual_xl), xu=np.full(n_var, actual_xu))
            
            def _evaluate(self, x, out, *args, **kwargs):
                x_t = torch.tensor(x, dtype=torch.float32)
                out["F"] = func_instance(x_t).numpy().reshape(-1, 1)
        
        problem = Wrapper()
        algorithm = create_pymoo_algorithm(algorithm_name, pop_size)
        
        history = []
        def callback(algo):
            if algo.opt is not None:
                history.append(float(algo.opt.get("F").min()))
        
        result = pymoo_minimize(
            problem, algorithm, get_termination("n_eval", max_evals),
            seed=seed, verbose=False, callback=callback,
        )
        
        return RunResult(
            algorithm=algorithm_name, config="pymoo", function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=history,
            best_fitness=float(result.F[0]) if result.F is not None else float('inf'),
            n_evals=result.algorithm.evaluator.n_eval,
            n_gen=len(history),
            wall_time=time.time() - start_time, success=True,
        )
        
    except Exception as e:
        import traceback
        return RunResult(
            algorithm=algorithm_name, config="pymoo", function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=[], best_fitness=float('inf'),
            n_evals=0, n_gen=0, wall_time=time.time() - start_time,
            success=False, error_message=f"{str(e)}\n{traceback.format_exc()}",
        )


def run_adam_single(
    func_name: str,
    n_var: int,
    xl: float,
    xu: float,
    seed: int,
    max_evals: int,
    pop_size: int,
    device: str,
    lr: float = 0.05,
) -> RunResult:
    """Run a single Adam optimization."""
    _worker_init()  # Ensure thread limits in worker
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    try:
        # Get function instance
        func_instance = get_function_instance(func_name, n_var, xl, xu)
        
        # Get actual bounds from the instance
        actual_xl = func_instance.xl[0].item()
        actual_xu = func_instance.xu[0].item()
        
        # Run Adam
        best_fitness, history, n_evals, n_gen, best_solution = run_adam_population(
            func_instance=func_instance,
            n_var=n_var,
            xl=actual_xl,
            xu=actual_xu,
            pop_size=pop_size,
            max_evals=max_evals,
            seed=seed,
            device=device,
            lr=lr,
        )
        
        return RunResult(
            algorithm="ADAM", config="Adam", function=func_name,
            n_var=n_var, seed=seed,
            best_fitness_history=history,
            best_fitness=best_fitness,
            n_evals=n_evals,
            n_gen=n_gen,
            wall_time=time.time() - start_time,
            success=True,
        )
        
    except Exception as e:
        import traceback
        return RunResult(
            algorithm="ADAM", config="Adam", function=func_name,
            n_var=n_var, seed=seed, best_fitness_history=[], best_fitness=float('inf'),
            n_evals=0, n_gen=0, wall_time=time.time() - start_time,
            success=False, error_message=f"{str(e)}\n{traceback.format_exc()}",
        )


# =============================================================================
# Parallel Execution
# =============================================================================

def run_single_job(job: Dict[str, Any]) -> RunResult:
    """Worker function for parallel execution."""
    if job["config"] == "pymoo":
        return run_pymoo_single(
            job["algorithm"], job["function"], job["n_var"],
            job["xl"], job["xu"], job["seed"], job["max_evals"], job["pop_size"],
        )
    elif job["config"] == "Adam":
        return run_adam_single(
            job["function"], job["n_var"],
            job["xl"], job["xu"], job["seed"], job["max_evals"], job["pop_size"],
            job["device"], job.get("adam_lr", 0.05),
        )
    else:
        return run_evograd_single(
            job["algorithm"], job["config"], job["function"], job["n_var"],
            job["xl"], job["xu"], job["seed"], job["max_evals"], job["pop_size"],
            job["lr_pop"], job["lr_hyper"], job["grad_clip_pop"],
            job["grad_clip_hyper"], job["device"],
        )


def run_benchmark_parallel(
    algorithm_name: str,
    functions: List[str],
    n_var: int,
    xl: float,
    xu: float,
    max_evals: int,
    lr_pop: float,
    lr_hyper: float,
    grad_clip_pop: float,
    grad_clip_hyper: float,
    n_runs: int = 30,
    pop_size: int = 100,
    device: str = "cpu",
    include_pymoo: bool = True,
    include_adam: bool = True,
    adam_lr: float = 0.05,
    n_workers: int = -1,
) -> BenchmarkResults:
    """Run full benchmark suite in parallel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = BenchmarkResults(
        algorithm=algorithm_name, timestamp=timestamp, device=device,
        n_var=n_var, xl=xl, xu=xu, max_evals=max_evals, n_runs=n_runs,
    )
    
    if n_workers == -1:
        n_workers = cpu_count()
    n_workers = max(1, min(n_workers, cpu_count()))
    
    # Determine configs based on algorithm
    is_adam = algorithm_name.upper() == "ADAM"
    
    if is_adam:
        configs = ["Adam"]  # Adam only has one config
    else:
        configs = ["classical", "differentiable", "adaptive", "full"]
    
    # Build jobs
    jobs = []
    for func_name in functions:
        for config in configs:
            for seed in range(n_runs):
                job = {
                    "algorithm": algorithm_name, "config": config, "function": func_name,
                    "n_var": n_var, "xl": xl, "xu": xu, "seed": seed,
                    "max_evals": max_evals, "pop_size": pop_size, 
                    "lr_pop": lr_pop, "lr_hyper": lr_hyper, "grad_clip_pop": grad_clip_pop,
                    "grad_clip_hyper": grad_clip_hyper, "device": device,
                }
                if is_adam:
                    job["adam_lr"] = adam_lr
                jobs.append(job)
        
        # Add pymoo baseline (not for Adam algorithm)
        if include_pymoo and PYMOO_AVAILABLE and not is_adam:
            for seed in range(n_runs):
                jobs.append({
                    "algorithm": algorithm_name, "config": "pymoo", "function": func_name,
                    "n_var": n_var, "xl": xl, "xu": xu, "seed": seed,
                    "max_evals": max_evals, "pop_size": pop_size, "device": "cpu",
                })
        
        # Add Adam baseline (not for Adam algorithm)
        if include_adam and not is_adam:
            for seed in range(n_runs):
                jobs.append({
                    "algorithm": algorithm_name, "config": "Adam", "function": func_name,
                    "n_var": n_var, "xl": xl, "xu": xu, "seed": seed,
                    "max_evals": max_evals, "pop_size": pop_size, "device": device,
                    "adam_lr": adam_lr,
                })
    
    total = len(jobs)
    
    # Count CEC 2017 functions
    n_cec2017 = sum(1 for f in functions if f.startswith("cec2017_"))
    n_classical = len(functions) - n_cec2017
    
    # Count configs
    n_configs = len(configs)
    if include_pymoo and PYMOO_AVAILABLE and not is_adam:
        n_configs += 1
    if include_adam and not is_adam:
        n_configs += 1
    
    print(f"\n{'='*70}")
    print(f"EvoGrad Parallel Benchmark Suite")
    print(f"{'='*70}")
    print(f"Algorithm:    {algorithm_name}")
    print(f"Functions:    {len(functions)} ({n_classical} classical, {n_cec2017} CEC2017)")
    if is_adam:
        print(f"Configs:      {n_configs} (Adam gradient-based)")
        print(f"Adam LR:      {adam_lr}")
    else:
        baselines = []
        if include_pymoo and PYMOO_AVAILABLE:
            baselines.append("pymoo")
        if include_adam:
            baselines.append("Adam")
        baseline_str = f", baselines: {', '.join(baselines)}" if baselines else ""
        print(f"Configs:      {n_configs} (EvoGrad: 4{baseline_str})")
        if include_adam:
            print(f"Adam LR:      {adam_lr}")
    print(f"D (n_var):    {n_var}")
    print(f"Bounds:       [{xl}, {xu}] (CEC2017 uses [-100, 100])")
    print(f"Max evals:    {max_evals}")
    print(f"Runs/config:  {n_runs}")
    print(f"Total jobs:   {total}")
    print(f"Workers:      {n_workers}")
    print(f"Device:       {device}")
    print(f"{'='*70}\n")
    
    if not EVOGRAD_AVAILABLE and not is_adam:
        print(f"WARNING: EvoGrad not available - {EVOGRAD_IMPORT_ERROR}\n")
    
    if n_cec2017 > 0 and not CEC2017_AVAILABLE:
        print(f"WARNING: CEC 2017 functions requested but not available\n")
    
    completed = 0
    start = time.time()
    
    executor_cls = ProcessPoolExecutor if device == "cpu" else ThreadPoolExecutor
    
    with executor_cls(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        
        for future in as_completed(futures):
            try:
                results.add_result(future.result())
            except Exception as e:
                job = futures[future]
                results.add_result(RunResult(
                    algorithm=job["algorithm"], config=job["config"], function=job["function"],
                    n_var=job["n_var"], seed=job["seed"], best_fitness_history=[],
                    best_fitness=float('inf'), n_evals=0, n_gen=0,
                    wall_time=0.0, success=False, error_message=str(e),
                ))
            
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                elapsed = time.time() - start
                eta = (total - completed) / (completed / elapsed) if completed > 0 else 0
                print(f"Progress: {completed:3d}/{total:3d} ({100*completed/total:3.0f}%) | "
                      f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
    
    print(f"\nCompleted {len(results.results)}/{total} jobs in {time.time()-start:.1f}s")
    return results


def print_summary(results: BenchmarkResults):
    """Print formatted summary."""
    summary = results.get_summary()
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY (D={results.n_var}, bounds=[{results.xl}, {results.xu}])")
    print(f"{'='*80}")
    
    # Group functions by type
    classical_funcs = sorted([f for f in summary.keys() if not f.startswith("cec2017_")])
    cec2017_funcs = sorted([f for f in summary.keys() if f.startswith("cec2017_")],
                           key=lambda x: int(x.replace("cec2017_f", "")))
    
    def print_func_results(func_list, title):
        if not func_list:
            return
        
        print(f"\n{title}")
        print("=" * 80)
        
        for func in func_list:
            print(f"\n{func.upper()}")
            print("-" * 70)
            print(f"{'Config':<15} {'Best':>12} {'Mean':>12} {'Std':>12} {'Time':>8}")
            print("-" * 70)
            
            # Order: classical, differentiable, adaptive, full, Adam, pymoo
            config_order = ["classical", "differentiable", "adaptive", "full", "Adam", "pymoo"]
            for config in config_order:
                if config in summary[func]:
                    s = summary[func][config]
                    print(f"{config:<15} {s['best']:>12.4e} {s['mean']:>12.4e} "
                          f"{s['std']:>12.4e} {s['mean_time']:>7.2f}s")
    
    print_func_results(classical_funcs, "CLASSICAL FUNCTIONS")
    print_func_results(cec2017_funcs, "CEC 2017 FUNCTIONS")
    
    print(f"\n{'='*80}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoGrad Parallel Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 classical functions (includes pymoo + Adam baselines)
  python run_benchmark.py -a DE -s quick -D 10 -r 5
  
  # Run without Adam baseline
  python run_benchmark.py -a DE -s quick -D 10 -r 5 --no_adam
  
  # Run without pymoo baseline
  python run_benchmark.py -a DE -s quick -D 10 -r 5 --no_pymoo
  
  # Run only Adam optimizer
  python run_benchmark.py -a ADAM -s quick -D 10 -r 30
  
  # Adam with custom learning rate
  python run_benchmark.py -a ADAM -s standard -D 10 --adam_lr 0.01
  
  # CEC 2017 simple functions (F1-F10)
  python run_benchmark.py -a DE -s cec2017_simple -D 10 -r 30
  
  # CEC 2017 all functions (F1-F30)
  python run_benchmark.py -a SHADE -s cec2017 -D 10 -r 30
  
  # Smoothed funnel functions (designed for differentiable EAs)
  python run_benchmark.py -a DE -s funnel -D 10 -r 30
  
  # Single funnel function (best for demonstrating EvoGrad advantages)
  python run_benchmark.py -a DE -f multibasinrastrigin -D 10 -r 30
  
  # Specific functions
  python run_benchmark.py -a DE -f sphere rastrigin cec2017_f1 multibasinrastrigin -D 10 -r 10
        """,
    )
    
    parser.add_argument("-a", "--algorithm", type=str, default="DE",
                        choices=["DE", "SHADE", "PSO", "GA", "CMAES", "ADAM"],
                        help="Algorithm to benchmark (ADAM for gradient-based only)")
    parser.add_argument("-s", "--suite", type=str, default="standard",
                        choices=list(SUITES.keys()),
                        help="Function suite to benchmark")
    parser.add_argument("-f", "--functions", type=str, nargs="+", default=None,
                        help="Specific functions (overrides --suite)")
    parser.add_argument("-D", "--n_var", type=int, default=10,
                        help="Number of variables (default: 10)")
    parser.add_argument("--xl", type=float, default=-100.0,
                        help="Lower bound (default: -100)")
    parser.add_argument("--xu", type=float, default=100.0,
                        help="Upper bound (default: 100)")
    parser.add_argument("-e", "--max_evals", type=int, default=None,
                        help="Max evaluations (default: 10000*D for CEC2017, 5000*D otherwise)")
    parser.add_argument("-r", "--n_runs", type=int, default=30,
                        help="Runs per configuration (default: 30)")
    parser.add_argument("-p", "--pop_size", type=int, default=100,
                        help="Population size (default: 100)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("-w", "--workers", type=int, default=-1,
                        help="Parallel workers (-1 for all CPUs)")
    parser.add_argument("--no_pymoo", action="store_true",
                        help="Disable pymoo baseline comparison")
    parser.add_argument("--no_adam", action="store_true",
                        help="Disable Adam baseline comparison")
    parser.add_argument("--adam_lr", type=float, default=0.05,
                        help="Adam learning rate (default: 0.05)")
    parser.add_argument("-o", "--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--list_functions", action="store_true",
                        help="List all available functions and exit")
    
    parser.add_argument("--lr_pop", type=float, default=-1)
    parser.add_argument("--lr_hyper", type=float, default=-1)
    parser.add_argument("--grad_clip_pop", type=float, default=-1)
    parser.add_argument("--grad_clip_hyper", type=float, default=-1)
    
    args = parser.parse_args()
    
    # List functions mode
    if args.list_functions:
        print("\nAvailable functions:")
        print("\nClassical:")
        for name in sorted(CLASSICAL_FUNCTIONS.keys()):
            print(f"  {name}")
        
        if CEC2017_AVAILABLE:
            print("\nCEC 2017 (F1-F30):")
            for i in range(1, 31):
                category = "simple" if i <= 10 else ("hybrid" if i <= 20 else "composition")
                print(f"  cec2017_f{i} ({category})")
        
        if SMOOTHED_FUNNEL_AVAILABLE:
            print("\nSmoothed Funnel (designed for differentiable EAs):")
            print("  multibasinrastrigin   - Multiple smoothed Rastrigin basins")
            print("  multibasinrosenbrock  - Multiple smoothed Rosenbrock basins")
            print("  deceptivelandscape    - Configurable deceptive multi-basin")
        
        print("\nSuites:")
        for suite_name, funcs in SUITES.items():
            print(f"  {suite_name}: {len(funcs)} functions")
        
        print("\nAlgorithms:")
        print("  DE, SHADE, PSO, GA, CMAES - Evolutionary algorithms (4 configs + baselines)")
        print("  ADAM - Gradient-based optimizer (standalone)")
        
        print("\nBaselines (for evolutionary algorithms):")
        print("  pymoo - Reference implementation (--no_pymoo to disable)")
        print("  Adam  - Gradient-based baseline (--no_adam to disable)")
        
        sys.exit(0)
    
    # Get functions
    functions = args.functions if args.functions else SUITES.get(args.suite, [])
    
    # Filter to available functions
    available = set(CLASSICAL_FUNCTIONS.keys())
    if CEC2017_AVAILABLE:
        available.update(f"cec2017_f{i}" for i in range(1, 31))
    if SMOOTHED_FUNNEL_AVAILABLE:
        available.update(SMOOTHED_FUNNEL_FUNCTIONS.keys())
    
    functions = [f for f in functions if f in available]
    
    if not functions:
        print(f"No valid functions. Use --list_functions to see available options.")
        sys.exit(1)
    
    # Check if any CEC 2017 functions
    has_cec2017 = any(f.startswith("cec2017_") for f in functions)
    
    # Default max_evals: 10000*D for CEC2017, 5000*D otherwise
    if args.max_evals is None:
        args.max_evals = 10000 * args.n_var if has_cec2017 else 5000 * args.n_var
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    
    # Run
    results = run_benchmark_parallel(
        algorithm_name=args.algorithm,
        functions=functions,
        n_var=args.n_var,
        xl=args.xl,
        xu=args.xu,
        max_evals=args.max_evals,
        n_runs=args.n_runs,
        lr_pop=args.lr_pop,
        lr_hyper=args.lr_hyper,
        grad_clip_pop=args.grad_clip_pop,
        grad_clip_hyper=args.grad_clip_hyper,
        pop_size=args.pop_size,
        device=args.device,
        include_pymoo=not args.no_pymoo,
        include_adam=not args.no_adam,
        adam_lr=args.adam_lr,
        n_workers=args.workers,
    )
    
    print_summary(results)
    results.save(Path(args.output_dir))


if __name__ == "__main__":
    main()
