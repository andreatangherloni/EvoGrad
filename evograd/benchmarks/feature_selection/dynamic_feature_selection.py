import numpy as np
import torch


class DynamicFeatureSelectELMProblem:
    def __init__(
        self,
        X_train,
        X_val,
        n_informative,
        noise,
        hidden,
        ridge_alpha,
        lambda_sparsity,
        n_regimes,
        shift_every,
        overlap,
        cycle_regimes,
        seed,
        device,
    ):
        self.device = device
        self.X_train = X_train
        self.X_val = X_val

        self.n_var = X_train.shape[1]
        self.xl = torch.zeros(self.n_var, device=device)
        self.xu = torch.ones(self.n_var, device=device)

        self.hidden = hidden
        self.ridge_alpha = ridge_alpha
        self.lambda_sparsity = lambda_sparsity
        self.shift_every = shift_every
        self.cycle_regimes = cycle_regimes

        rng = np.random.default_rng(seed)
        self.weights = []
        for _ in range(n_regimes):
            w = np.zeros(self.n_var, dtype=np.float32)
            idx = rng.choice(self.n_var, n_informative, replace=False)
            w[idx] = rng.standard_normal(n_informative)
            self.weights.append(torch.tensor(w, device=device))

        g = torch.Generator(device=device).manual_seed(seed)
        self.W = torch.randn(self.n_var, hidden, generator=g, device=device) / np.sqrt(self.n_var)
        self.b = torch.randn(hidden, generator=g, device=device)

        self.eval_counter = 0
        self.noise = noise

    def _regime(self):
        r = self.eval_counter // self.shift_every
        return int(r % len(self.weights)) if self.cycle_regimes else int(min(r, len(self.weights) - 1))

    def evaluate(self, population):
        m = population.clamp(self.xl, self.xu)
        N = m.shape[0]

        r = self._regime()
        ytr = self.X_train @ self.weights[r]
        yva = self.X_val @ self.weights[r]

        Xtr = self.X_train.unsqueeze(0) * m.unsqueeze(1)
        Xva = self.X_val.unsqueeze(0) * m.unsqueeze(1)

        Htr = torch.relu(torch.einsum("nid,dh->nih", Xtr, self.W) + self.b)
        Hva = torch.relu(torch.einsum("nid,dh->nih", Xva, self.W) + self.b)

        ytr = ytr.view(1, -1, 1).expand(N, -1, -1)
        A = torch.einsum("nth,ntk->nhk", Htr, Htr)
        A += self.ridge_alpha * torch.eye(self.hidden, device=self.device).unsqueeze(0)
        B = torch.einsum("nih,niq->nhq", Htr, ytr)

        beta = torch.linalg.solve(A, B)
        yhat = torch.einsum("nih,nhq->niq", Hva, beta).squeeze(-1)

        mse = ((yhat - yva) ** 2).mean(dim=1)
        sparsity = self.lambda_sparsity * m.mean(dim=1)

        self.eval_counter += N
        return mse + sparsity