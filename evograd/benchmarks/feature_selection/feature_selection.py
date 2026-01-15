import numpy as np
import torch


class FeatureSelectELMProblem:
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        hidden=128,
        ridge_alpha=1e-2,
        lambda_sparsity=1e-2,
        seed=0,
        device=None,
    ):
        self.device = device or X_train.device
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_var = X_train.shape[1]
        self.xl = torch.zeros(self.n_var, device=self.device)
        self.xu = torch.ones(self.n_var, device=self.device)

        self.hidden = hidden
        self.ridge_alpha = ridge_alpha
        self.lambda_sparsity = lambda_sparsity

        g = torch.Generator(device=self.device).manual_seed(seed)
        self.W = torch.randn(self.n_var, hidden, generator=g, device=self.device) / np.sqrt(self.n_var)
        self.b = torch.randn(hidden, generator=g, device=self.device)

    def evaluate(self, population):
        m = population.clamp(self.xl, self.xu)

        Xtr = self.X_train.unsqueeze(0) * m.unsqueeze(1)
        Xva = self.X_val.unsqueeze(0) * m.unsqueeze(1)

        Htr = torch.relu(torch.einsum("nid,dh->nih", Xtr, self.W) + self.b)
        Hva = torch.relu(torch.einsum("nid,dh->nih", Xva, self.W) + self.b)

        ytr = self.y_train.view(1, -1, 1).expand(m.shape[0], -1, -1)
        A = torch.einsum("nth,ntk->nhk", Htr, Htr)
        A += self.ridge_alpha * torch.eye(self.hidden, device=self.device).unsqueeze(0)
        B = torch.einsum("nih,niq->nhq", Htr, ytr)

        beta = torch.linalg.solve(A, B)
        yhat = torch.einsum("nih,nhq->niq", Hva, beta).squeeze(-1)

        mse = ((yhat - self.y_val) ** 2).mean(dim=1)
        sparsity = self.lambda_sparsity * m.mean(dim=1)
        return mse + sparsity