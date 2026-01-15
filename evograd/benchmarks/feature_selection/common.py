import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------
# Device & seeding
# ---------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Synthetic regression data
# ---------------------------------------------------------------------

def make_synthetic_regression(
    n_train: int,
    n_val: int,
    n_features: int,
    n_informative: int,
    noise: float,
    seed: int,
    device: torch.device,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train + n_val, n_features)).astype(np.float32)

    w = np.zeros(n_features, dtype=np.float32)
    idx = rng.choice(n_features, size=n_informative, replace=False)
    w[idx] = rng.standard_normal(n_informative).astype(np.float32)

    y = X @ w + noise * rng.standard_normal(n_train + n_val).astype(np.float32)

    return (
        torch.tensor(X[:n_train], device=device),
        torch.tensor(y[:n_train], device=device),
        torch.tensor(X[n_train:], device=device),
        torch.tensor(y[n_train:], device=device),
        idx,
    )


# ---------------------------------------------------------------------
# EvoGrad helpers
# ---------------------------------------------------------------------

def import_minimize():
    for m in ["evograd.core.minimize", "evograd.core"]:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "minimize"):
                return mod.minimize
        except Exception:
            pass
    raise ImportError("Could not import evograd minimize")


def import_ga_default():
    for m in ["evograd.algorithms.ga", "evograd.algorithms"]:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "ga_default"):
                return mod.ga_default
        except Exception:
            pass
    raise ImportError("Could not import ga_default")


def make_termination(max_evals: int):
    try:
        from evograd.core.termination import MaxEvaluations
        return MaxEvaluations(max_evals)
    except Exception:
        return {"max_evals": max_evals}


# ---------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------

def write_results_json(
    out_path: Path,
    algorithm: str,
    n_var: int,
    max_evals: int,
    n_runs: int,
    results: List[Dict[str, Any]],
):
    payload = {
        "algorithm": algorithm,
        "n_var": n_var,
        "xl": 0.0,
        "xu": 1.0,
        "max_evals": max_evals,
        "n_runs": n_runs,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")