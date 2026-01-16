"""
Dynamic Feature Selection Problem with regime shifts.

This module extends the static feature selection problem to include
time-varying ground-truth weights that shift periodically. This tests
the adaptive capabilities of differentiable evolutionary algorithms.

The objective remains:
    L(m) = MSE_val(X ⊙ m) + λ · mean(m)

but the underlying regression target y = Xw changes over time as the
ground-truth weight vector w shifts between pre-defined regimes.

Key features:
- Configurable number of regimes with distinct informative feature sets
- Controllable overlap between consecutive regimes
- Cycling or terminal regime behaviour
- Evaluation counter for tracking regime transitions
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional, List, Tuple


class DynamicFeatureSelectELMProblem:
    """
    Dynamic feature selection problem with regime shifts.
    
    Extends the static ELM-based feature selection to include time-varying
    ground-truth weights. The informative features change periodically,
    testing the algorithm's ability to adapt to distribution shifts.
    
    Args:
        X_train: Training features, shape (n_train, n_features).
        X_val: Validation features, shape (n_val, n_features).
        n_informative: Number of informative features per regime.
        noise: Noise level added to targets.
        hidden: Number of hidden units in the ELM.
        ridge_alpha: Ridge regression regularisation parameter.
        lambda_sparsity: Sparsity penalty coefficient.
        n_regimes: Number of distinct regimes (weight vectors).
        shift_every: Number of evaluations between regime shifts.
        overlap: Fraction of informative features shared between consecutive
                regimes (0.0 = no overlap, 1.0 = identical regimes).
        cycle_regimes: If True, cycle through regimes; if False, stay at last.
        seed: Random seed for weight generation.
        device: Torch device for computation.
    
    Example:
        >>> problem = DynamicFeatureSelectELMProblem(
        ...     X_train, X_val, n_informative=20, noise=0.1,
        ...     n_regimes=5, shift_every=5000, overlap=0.25,
        ... )
        >>> fitness = problem.evaluate(population)
        >>> print(f"Current regime: {problem.current_regime}")
    """
    
    def __init__(
        self,
        X_train: Tensor,
        X_val: Tensor,
        n_informative: int,
        noise: float,
        hidden: int,
        ridge_alpha: float,
        lambda_sparsity: float,
        n_regimes: int,
        shift_every: int,
        overlap: float,
        cycle_regimes: bool,
        seed: int,
        device: torch.device
    ):
        self.device = device
        self.X_train = X_train.to(device)
        self.X_val = X_val.to(device)

        self.n_var = X_train.shape[1]
        self.xl = torch.zeros(self.n_var, device=device)
        self.xu = torch.ones(self.n_var, device=device)

        self.hidden = hidden
        self.ridge_alpha = ridge_alpha
        self.lambda_sparsity = lambda_sparsity
        self.shift_every = shift_every
        self.cycle_regimes = cycle_regimes
        self.n_informative = n_informative
        self.overlap = overlap
        self.noise = noise
        self._seed = seed

        # Pre-allocate ridge regularisation matrix
        self._ridge_eye = ridge_alpha * torch.eye(hidden, device=device)

        # Generate regime weights with controlled overlap
        rng = np.random.default_rng(seed)
        self.weights, self.informative_indices = self._generate_regime_weights(
            rng, n_regimes, n_informative, overlap
        )
        
        # Fixed ELM weights (shared across all regimes)
        g = torch.Generator(device=device).manual_seed(seed)
        self.W = torch.randn(self.n_var, hidden, generator=g, device=device)
        self.W = self.W / np.sqrt(self.n_var)
        self.b = torch.randn(hidden, generator=g, device=device)

        # Evaluation tracking
        self.eval_counter = 0
        self._regime_history: List[Tuple[int, int]] = []  # (eval_count, regime)

    def _generate_regime_weights(
        self, 
        rng: np.random.Generator,
        n_regimes: int,
        n_informative: int,
        overlap: float,
    ) -> Tuple[List[Tensor], List[np.ndarray]]:
        """
        Generate weight vectors for each regime with controlled overlap.
        
        The overlap parameter controls how many informative features are
        shared between consecutive regimes:
        - overlap=0.0: Completely disjoint feature sets (if possible)
        - overlap=0.5: Half the features shared with previous regime
        - overlap=1.0: Identical feature sets (only weights change)
        
        Args:
            rng: NumPy random generator.
            n_regimes: Number of regimes to generate.
            n_informative: Number of informative features per regime.
            overlap: Fraction of features to share with previous regime.
            
        Returns:
            Tuple of (weight tensors, informative index arrays).
        """
        weights = []
        indices = []
        
        n_shared = int(n_informative * overlap)
        n_new = n_informative - n_shared
        
        prev_idx = None
        
        for r in range(n_regimes):
            w = np.zeros(self.n_var, dtype=np.float32)
            
            if r == 0 or overlap == 0.0:
                # First regime or no overlap: random selection
                idx = rng.choice(self.n_var, n_informative, replace=False)
            else:
                # Subsequent regimes: maintain overlap with previous
                # Select n_shared features from previous regime
                shared_idx = rng.choice(prev_idx, size=min(n_shared, len(prev_idx)), replace=False)
                
                # Select n_new features from remaining (not in previous regime)
                available = np.setdiff1d(np.arange(self.n_var), prev_idx)
                if len(available) >= n_new:
                    new_idx = rng.choice(available, size=n_new, replace=False)
                else:
                    # Not enough new features, sample from all non-shared
                    remaining_needed = n_new - len(available)
                    new_idx = np.concatenate([
                        available,
                        rng.choice(
                            np.setdiff1d(prev_idx, shared_idx),
                            size=remaining_needed,
                            replace=False
                        )
                    ])
                
                idx = np.concatenate([shared_idx, new_idx])
            
            # Generate non-zero weights for informative features
            w[idx] = rng.standard_normal(len(idx)).astype(np.float32)
            
            weights.append(torch.tensor(w, device=self.device))
            indices.append(idx.copy())
            prev_idx = idx
        
        return weights, indices

    @property
    def current_regime(self) -> int:
        """Get the current regime index based on evaluation count."""
        return self._regime()

    def _regime(self) -> int:
        """Compute current regime from evaluation counter."""
        r = self.eval_counter // self.shift_every
        if self.cycle_regimes:
            return int(r % len(self.weights))
        else:
            return int(min(r, len(self.weights) - 1))

    def _constrain_mask(self, population: Tensor) -> Tensor:
        """Constrain population to [0, 1]"""
        return population.clamp(self.xl, self.xu)

    def evaluate(self, population: Tensor) -> Tensor:
        """
        Evaluate fitness of a population under current regime.
        
        The ground-truth target y = Xw uses the weight vector of the
        current regime, which changes based on evaluation count.
        
        Args:
            population: Feature masks, shape (pop_size, n_var).
            
        Returns:
            Fitness values, shape (pop_size,). Lower is better.
        """
        m = self._constrain_mask(population)
        pop_size = m.shape[0]

        # Get current regime's weight vector
        r = self._regime()
        
        # Track regime transitions
        if not self._regime_history or self._regime_history[-1][1] != r:
            self._regime_history.append((self.eval_counter, r))

        # Generate targets for current regime
        ytr = self.X_train @ self.weights[r]
        yva = self.X_val @ self.weights[r]
        
        # Add noise to training targets (optional, for robustness testing)
        if self.noise > 0:
            ytr = ytr + self.noise * torch.randn_like(ytr)

        # Apply feature mask
        Xtr = self.X_train.unsqueeze(0) * m.unsqueeze(1)
        Xva = self.X_val.unsqueeze(0) * m.unsqueeze(1)

        # ELM hidden activations
        Htr = torch.relu(torch.einsum("nid,dh->nih", Xtr, self.W) + self.b)
        Hva = torch.relu(torch.einsum("nid,dh->nih", Xva, self.W) + self.b)

        # Ridge regression
        ytr_batch = ytr.view(1, -1, 1).expand(pop_size, -1, -1)
        A = torch.einsum("nth,ntk->nhk", Htr, Htr)
        A = A + self._ridge_eye.unsqueeze(0)
        B = torch.einsum("nih,niq->nhq", Htr, ytr_batch)

        beta = torch.linalg.solve(A, B)
        yhat = torch.einsum("nih,nhq->niq", Hva, beta).squeeze(-1)

        # Fitness computation
        mse = ((yhat - yva) ** 2).mean(dim=1)
        sparsity = self.lambda_sparsity * m.mean(dim=1)

        # Update evaluation counter
        self.eval_counter += pop_size
        
        return mse + sparsity

    def get_current_informative_features(self) -> np.ndarray:
        """Get indices of informative features in current regime."""
        return self.informative_indices[self._regime()]

    def get_regime_overlap(self, regime_a: int, regime_b: int) -> float:
        """
        Compute actual overlap between two regimes.
        
        Returns:
            Jaccard similarity between informative feature sets.
        """
        idx_a = set(self.informative_indices[regime_a])
        idx_b = set(self.informative_indices[regime_b])
        intersection = len(idx_a & idx_b)
        union = len(idx_a | idx_b)
        return intersection / union if union > 0 else 0.0

    def reset(self):
        """Reset evaluation counter and regime history."""
        self.eval_counter = 0
        self._regime_history = []

    def get_effective_mask(self, population: Tensor, threshold: float = 0.5) -> Tensor:
        """Get binary feature selection from soft masks."""
        m = self._constrain_mask(population)
        return (m > threshold).float()
