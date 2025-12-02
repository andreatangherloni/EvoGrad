from __future__ import annotations

from typing import Callable, Any, Dict, List, Union, Optional
import numpy as np
import torch

from fstpso import FuzzyPSO
import cma
from pyade import de as pyade_de
from deap import base, creator, tools, algorithms


def _to_numpy_bounds(lower_bound, upper_bound):
    lb = np.array(lower_bound, dtype=np.float32)
    ub = np.array(upper_bound, dtype=np.float32)
    return lb, ub


def _make_torch_objective(
    func: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cpu",
) -> Callable[[np.ndarray], float]:
    """
    Wraps a vectorised PyTorch objective f: [N,D] -> [N] into a scalar function
    f_np: [D] -> float suitable for external libraries.
    """
    def objective(x: np.ndarray) -> float:
        x_np = np.asarray(x, dtype=np.float32).reshape(1, -1)
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y = func(x_t).view(-1)
        return float(y[0].item())
    return objective



class FSTPSO:

    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        pop_size: int,
        lower_bound: Union[List[float], np.ndarray],
        upper_bound: Union[List[float], np.ndarray],
        max_evals: int,
        device: str = "cpu",
        **kwargs: Any,
    ):

        self.func = function
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_evals = max_evals
        self.device = device

    def run(self) -> float:
        lb, ub = _to_numpy_bounds(self.lower_bound, self.upper_bound)
        search_space = [[float(l), float(u)] for l, u in zip(lb, ub)]

        f_scalar = _make_torch_objective(self.func, device=self.device)

        fp = FuzzyPSO()
        fp.set_search_space(search_space)
        fp.set_fitness(f_scalar)

        fp.set_swarm_size(self.pop_size)

        max_iter = max(1, self.max_evals // self.pop_size)

        best_x, best_val = fp.solve_with_fstpso(max_iter=max_iter)
        return float(best_val)

    def optimize(self) -> float:
        return self.run()

    __call__ = run

class GAStandard:
    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        pop_size: int,
        lower_bound: Union[List[float], np.ndarray],
        upper_bound: Union[List[float], np.ndarray],
        max_evals: int,
        device: str = "cpu",
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.1,
        tournament_size: int = 3,
        **kwargs: Any,
    ):

        self.func = function
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_evals = max_evals
        self.device = device

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size

        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)

    def run(self) -> float:
        lb, ub = _to_numpy_bounds(self.lower_bound, self.upper_bound)

        tb = base.Toolbox()

        def init_individual():
            return [np.random.uniform(l, u) for l, u in zip(lb, ub)]

        tb.register("individual", tools.initIterate, creator.Individual, init_individual)
        tb.register("population", tools.initRepeat, list, tb.individual)

        f_scalar = _make_torch_objective(self.func, device=self.device)

        def eval_individual(individual):
            x = np.array(individual, dtype=np.float32)
            return (f_scalar(x),)

        tb.register("evaluate", eval_individual)
        tb.register("mate", tools.cxBlend, alpha=0.5)
        tb.register(
            "mutate",
            tools.mutGaussian,
            mu=0.0,
            sigma=self.mutation_sigma,
            indpb=self.mutation_rate,
        )
        tb.register("select", tools.selTournament, tournsize=self.tournament_size)

        pop = tb.population(n=self.pop_size)

        ngen = max(1, self.max_evals // self.pop_size)

        algorithms.eaSimple(
            population=pop,
            toolbox=tb,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=ngen,
            verbose=False,
        )

        best = tools.selBest(pop, 1)[0]
        return float(best.fitness.values[0])

    def optimize(self) -> float:
        return self.run()

    __call__ = run

class DEStandard:


    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        pop_size: int,
        lower_bound: Union[List[float], np.ndarray],
        upper_bound: Union[List[float], np.ndarray],
        max_evals: int,
        device: str = "cpu",
        **kwargs: Any,
    ):


        self.func = function
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_evals = max_evals
        self.device = device

    def run(self) -> float:
        lb, ub = _to_numpy_bounds(self.lower_bound, self.upper_bound)
        bounds = np.stack([lb, ub], axis=1) 

        f_scalar = _make_torch_objective(self.func, device=self.device)

        params = pyade_de.get_default_params(self.dim)
        params["population_size"] = self.pop_size
        params["max_evals"] = self.max_evals
        params["individual_size"] = self.dim
        params["bounds"] = bounds
        params["func"] = f_scalar

        best_x, best_val = pyade_de.apply(**params)
        return float(best_val)

    def optimize(self) -> float:
        return self.run()

    __call__ = run

class CMAESStandard:
    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        pop_size: int,
        lower_bound: Union[List[float], np.ndarray],
        upper_bound: Union[List[float], np.ndarray],
        max_evals: int,
        device: str = "cpu",
        **kwargs: Any,
    ):

        self.func = function
        self.dim = dim
        self.pop_size = pop_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_evals = max_evals
        self.device = device

    def run(self) -> float:
        lb, ub = _to_numpy_bounds(self.lower_bound, self.upper_bound)
        f_scalar = _make_torch_objective(self.func, device=self.device)

        x0 = ((lb + ub) / 2.0).astype(float)

        sigma0 = float(np.mean(ub - lb) / 6.0)

        options = {
            "bounds": [lb.tolist(), ub.tolist()],
            "popsize": int(self.pop_size),
            "maxfevals": int(self.max_evals),
            "verb_disp": 0,  
        }

        xbest, fbest, *_ = cma.fmin(
            f_scalar,
            x0.tolist(),
            sigma0,
            options=options,
        )

        return float(fbest)

    def optimize(self) -> float:
        return self.run()

    __call__ = run
