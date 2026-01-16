"""
Common utilities for EvoGrad feature selection benchmarks.

Includes:
- Device resolution and seeding
- Synthetic data generation
- Feature recovery metrics (Jaccard, F1, Precision, Recall)
- EvoGrad import helpers
- JSON output utilities
"""

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------
# Device & Seeding
# ---------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    """
    Resolve device string to torch.device with availability checking.
    
    Args:
        device: One of 'cpu', 'cuda', 'mps'.
        
    Returns:
        Resolved torch.device object.
    """
    device = device.lower()
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Synthetic Regression Data
# ---------------------------------------------------------------------

def make_synthetic_regression(
    n_train: int,
    n_val: int,
    n_features: int,
    n_informative: int,
    noise: float,
    seed: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray, Tensor]:
    """
    Generate synthetic regression data with known informative features.
    
    Creates a sparse linear regression problem where only n_informative
    features have non-zero coefficients.
    
    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_features: Total number of features (D).
        n_informative: Number of features with non-zero weights.
        noise: Standard deviation of Gaussian noise added to targets.
        seed: Random seed for reproducibility.
        device: Torch device for output tensors.
        
    Returns:
        Tuple of:
            - X_train: Training features, shape (n_train, n_features)
            - y_train: Training targets, shape (n_train,)
            - X_val: Validation features, shape (n_val, n_features)
            - y_val: Validation targets, shape (n_val,)
            - informative_idx: Indices of informative features
            - weights: Ground-truth weight vector, shape (n_features,)
    """
    rng = np.random.default_rng(seed)
    
    # Generate features from standard normal
    X = rng.standard_normal((n_train + n_val, n_features)).astype(np.float32)

    # Sparse weight vector
    w = np.zeros(n_features, dtype=np.float32)
    idx = rng.choice(n_features, size=n_informative, replace=False)
    w[idx] = rng.standard_normal(n_informative).astype(np.float32)

    # Generate targets with noise
    y = X @ w + noise * rng.standard_normal(n_train + n_val).astype(np.float32)

    return (
        torch.tensor(X[:n_train], device=device),
        torch.tensor(y[:n_train], device=device),
        torch.tensor(X[n_train:], device=device),
        torch.tensor(y[n_train:], device=device),
        idx,
        torch.tensor(w, device=device),
    )


# ---------------------------------------------------------------------
# Feature Recovery Metrics
# ---------------------------------------------------------------------

def compute_feature_recovery_metrics(
    predicted_mask: Union[Tensor, np.ndarray],
    true_indices: np.ndarray,
    n_features: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute feature recovery metrics comparing predicted mask to ground truth.
    
    Args:
        predicted_mask: Soft or binary mask, shape (n_features,).
        true_indices: Array of ground-truth informative feature indices.
        n_features: Total number of features.
        threshold: Threshold for converting soft mask to binary.
        
    Returns:
        Dictionary with metrics:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1: Harmonic mean of precision and recall
            - jaccard: |intersection| / |union|
            - n_selected: Number of features selected
            - n_true: Number of true informative features
    """
    # Convert to numpy if needed
    if isinstance(predicted_mask, Tensor):
        predicted_mask = predicted_mask.detach().cpu().numpy()
    
    # Binarise predicted mask
    pred_selected = set(np.where(predicted_mask > threshold)[0])
    true_selected = set(true_indices)
    
    # Compute confusion matrix components
    tp = len(pred_selected & true_selected)
    fp = len(pred_selected - true_selected)
    fn = len(true_selected - pred_selected)
    
    # Metrics with safe division
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Jaccard similarity
    union = len(pred_selected | true_selected)
    jaccard = tp / union if union > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "n_selected": len(pred_selected),
        "n_true": len(true_selected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def compute_mask_statistics(
    mask: Union[Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute statistics of a soft feature mask.
    
    Args:
        mask: Soft mask values in [0, 1], shape (n_features,).
        
    Returns:
        Dictionary with statistics:
            - mean: Average mask value
            - std: Standard deviation
            - sparsity: Fraction of values below 0.1
            - effective_features: Sum of mask values (soft count)
    """
    if isinstance(mask, Tensor):
        mask = mask.detach().cpu().numpy()
    
    return {
        "mean": float(np.mean(mask)),
        "std": float(np.std(mask)),
        "sparsity": float(np.mean(mask < 0.1)),
        "effective_features": float(np.sum(mask)),
        "max": float(np.max(mask)),
        "min": float(np.min(mask)),
    }


def compute_recovery_over_thresholds(
    predicted_mask: Union[Tensor, np.ndarray],
    true_indices: np.ndarray,
    n_features: int,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """
    Compute feature recovery metrics over multiple thresholds.
    
    Useful for generating precision-recall curves.
    
    Args:
        predicted_mask: Soft mask values.
        true_indices: Ground-truth informative feature indices.
        n_features: Total number of features.
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.9).
        
    Returns:
        Dictionary with lists of metrics at each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {
        "thresholds": thresholds,
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
        "n_selected": [],
    }
    
    for t in thresholds:
        metrics = compute_feature_recovery_metrics(
            predicted_mask, true_indices, n_features, threshold=t
        )
        results["precision"].append(metrics["precision"])
        results["recall"].append(metrics["recall"])
        results["f1"].append(metrics["f1"])
        results["jaccard"].append(metrics["jaccard"])
        results["n_selected"].append(metrics["n_selected"])
    
    return results


# ---------------------------------------------------------------------
# EvoGrad Import Helpers
# ---------------------------------------------------------------------

def import_minimize():
    """Import the minimize function from EvoGrad."""
    for m in ["evograd.core.minimize", "evograd.core"]:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "minimize"):
                return mod.minimize
        except Exception:
            pass
    raise ImportError("Could not import evograd minimize")


def import_ga_default():
    """Import the ga_default factory from EvoGrad."""
    for m in ["evograd.algorithms.ga", "evograd.algorithms"]:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "ga_default"):
                return mod.ga_default
        except Exception:
            pass
    raise ImportError("Could not import ga_default")


def make_termination(max_evals: int):
    """Create a termination criterion for EvoGrad."""
    try:
        from evograd.core.termination import MaxEvaluations
        return MaxEvaluations(max_evals)
    except Exception:
        return {"max_evals": max_evals}


# ---------------------------------------------------------------------
# JSON Output
# ---------------------------------------------------------------------

def write_results_json(
    out_path: Path,
    algorithm: str,
    n_var: int,
    max_evals: int,
    n_runs: int,
    results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Write benchmark results to JSON file.
    
    Args:
        out_path: Output file path.
        algorithm: Name of the primary algorithm.
        n_var: Problem dimensionality.
        max_evals: Maximum fitness evaluations.
        n_runs: Number of independent runs.
        results: List of result dictionaries per run.
        metadata: Optional additional metadata to include.
    """
    payload = {
        "algorithm": algorithm,
        "n_var": n_var,
        "xl": 0.0,
        "xu": 1.0,
        "max_evals": max_evals,
        "n_runs": n_runs,
        "results": results,
    }
    
    if metadata:
        payload["metadata"] = metadata
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")


def load_results_json(path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)
