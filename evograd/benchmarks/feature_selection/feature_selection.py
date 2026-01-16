"""
Static Feature Selection Problem with ELM-based evaluation.

This module implements a feature selection objective that couples sparse
mask learning with model training using a fixed random-feature neural
network (Extreme Learning Machine architecture).

The objective is:
    L(m) = MSE_val(X ⊙ m) + λ · mean(m)

where m ∈ [0,1]^D is a soft feature mask.

Supports both classical and differentiable optimisation modes:
- Classical: Hard clamp to [0,1] bounds
- Differentiable: Sigmoid squashing to preserve gradient flow
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple


class FeatureSelectELMProblem:
    """
    Feature selection problem using Extreme Learning Machine evaluation.
    
    Given input data X ∈ R^{n×D}, learns a sparse feature mask m ∈ [0,1]^D
    that minimises validation error while encouraging compact representations.
    
    The model uses a fixed random-feature neural network (single hidden layer
    with ReLU activations), while the output layer is trained via closed-form
    ridge regression. This ensures the objective remains fully differentiable
    with respect to the feature mask.
    
    Args:
        X_train: Training features, shape (n_train, n_features).
        y_train: Training targets, shape (n_train,).
        X_val: Validation features, shape (n_val, n_features).
        y_val: Validation targets, shape (n_val,).
        hidden: Number of hidden units in the ELM.
        ridge_alpha: Ridge regression regularisation parameter.
        lambda_sparsity: Sparsity penalty coefficient.
        seed: Random seed for ELM weight initialisation.
        device: Torch device for computation.
        differentiable: If True, use sigmoid squashing instead of hard clamp.
    
    Example:
        >>> problem = FeatureSelectELMProblem(
        ...     X_train, y_train, X_val, y_val,
        ...     hidden=128, lambda_sparsity=0.01, differentiable=True
        ... )
        >>> fitness = problem.evaluate(population)  # shape: (pop_size,)
    """
    
    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Tensor,
        y_val: Tensor,
        hidden: int = 128,
        ridge_alpha: float = 1e-2,
        lambda_sparsity: float = 1e-2,
        seed: int = 0,
        device: Optional[torch.device] = None,
        differentiable: bool = False,
    ):
        self.device = device or X_train.device
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.X_val = X_val.to(self.device)
        self.y_val = y_val.to(self.device)

        self.n_var = X_train.shape[1]
        self.xl = torch.zeros(self.n_var, device=self.device)
        self.xu = torch.ones(self.n_var, device=self.device)

        self.hidden = hidden
        self.ridge_alpha = ridge_alpha
        self.lambda_sparsity = lambda_sparsity
        self.differentiable = differentiable

        # Pre-allocate ridge regularisation matrix (avoid repeated creation)
        self._ridge_eye = ridge_alpha * torch.eye(hidden, device=self.device)

        # Fixed random ELM weights (Xavier initialisation)
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.W = torch.randn(self.n_var, hidden, generator=g, device=self.device)
        self.W = self.W / np.sqrt(self.n_var)  # Xavier scaling
        self.b = torch.randn(hidden, generator=g, device=self.device)
        
        # Store seed for reproducibility tracking
        self._seed = seed

    def _constrain_mask(self, population: Tensor) -> Tensor:
        """
        Constrain population to [0, 1] bounds.
        
        In differentiable mode, uses sigmoid squashing to preserve gradients.
        In classical mode, uses hard clamping.
        
        Args:
            population: Raw population tensor, shape (pop_size, n_var).
            
        Returns:
            Constrained mask tensor in [0, 1]^D.
        """
        if self.differentiable:
            # Sigmoid squashing: smooth, differentiable, gradients never zero
            # Scale factor of 6 makes sigmoid(0)=0.5, sigmoid(±3)≈0/1
            # This maps roughly [-0.5, 1.5] -> [0.05, 0.95]
            centered = (population - 0.5) * 6.0
            return torch.sigmoid(centered)
        else:
            # Hard clamp: efficient but zero gradients at boundaries
            return population.clamp(self.xl, self.xu)

    def evaluate(self, population: Tensor) -> Tensor:
        """
        Evaluate fitness of a population of feature masks.
        
        Computes MSE on validation set + sparsity penalty for each individual.
        
        Args:
            population: Feature masks, shape (pop_size, n_var).
                       Values should be in [0, 1] or will be constrained.
                       
        Returns:
            Fitness values, shape (pop_size,). Lower is better.
        """
        m = self._constrain_mask(population)
        pop_size = m.shape[0]

        # Apply feature mask: X_masked[i] = X * m[i] for each individual
        # Shape: (pop_size, n_samples, n_features)
        Xtr = self.X_train.unsqueeze(0) * m.unsqueeze(1)
        Xva = self.X_val.unsqueeze(0) * m.unsqueeze(1)

        # Hidden layer activations: H = ReLU(X @ W + b)
        # Shape: (pop_size, n_samples, hidden)
        Htr = torch.relu(torch.einsum("nid,dh->nih", Xtr, self.W) + self.b)
        Hva = torch.relu(torch.einsum("nid,dh->nih", Xva, self.W) + self.b)

        # Ridge regression: beta = (H'H + αI)^{-1} H'y
        # Batched over population
        ytr = self.y_train.view(1, -1, 1).expand(pop_size, -1, -1)
        
        # A = H'H + αI, shape: (pop_size, hidden, hidden)
        A = torch.einsum("nth,ntk->nhk", Htr, Htr)
        A = A + self._ridge_eye.unsqueeze(0)
        
        # B = H'y, shape: (pop_size, hidden, 1)
        B = torch.einsum("nih,niq->nhq", Htr, ytr)

        # Solve for output weights
        beta = torch.linalg.solve(A, B)
        
        # Predictions on validation set
        yhat = torch.einsum("nih,nhq->niq", Hva, beta).squeeze(-1)

        # MSE loss per individual
        mse = ((yhat - self.y_val) ** 2).mean(dim=1)
        
        # Sparsity penalty: encourage small mask values
        sparsity = self.lambda_sparsity * m.mean(dim=1)
        
        return mse + sparsity

    def get_effective_mask(self, population: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Get binary feature selection from soft masks.
        
        Args:
            population: Soft masks, shape (pop_size, n_var).
            threshold: Threshold for considering a feature selected.
            
        Returns:
            Binary masks, shape (pop_size, n_var).
        """
        m = self._constrain_mask(population)
        return (m > threshold).float()

    def reset(self):
        """Reset any stateful counters (for compatibility with dynamic version)."""
        pass
