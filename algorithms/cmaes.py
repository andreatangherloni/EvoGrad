import torch
import torch.nn as nn

class CMAES(nn.Module):
    """
    """
    def __init__(self,
                 obj_func,
                 dim,
                 pop_size=30,
                 lower_bound=None,
                 upper_bound=None,
                 init_sigma=0.3,
                 lr=0.001,
                 optimizer=None,
                 seed=42):
        
        super().__init__()
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if lower_bound is None:
            lower_bound = [-5.0]*dim
        if upper_bound is None:
            upper_bound = [5.0]*dim
        self.register_buffer("lower_bound", torch.tensor(lower_bound, dtype=torch.float32))
        self.register_buffer("upper_bound", torch.tensor(upper_bound, dtype=torch.float32))

        # mean
        lb = torch.tensor(lower_bound, dtype=torch.float32)
        ub = torch.tensor(upper_bound, dtype=torch.float32)
        init_m = 0.5*(lb+ub)
        self.m = nn.Parameter(init_m) # shape [dim]
        
        # L_tri
        init_L = torch.eye(dim, dtype=torch.float32)*init_sigma
        self.L_tri = nn.Parameter(init_L)

        # Evaluate initial single-sample for old best
        with torch.no_grad():
            init_fitness = self.obj_func(self.m.unsqueeze(0))
        
        self.fitnesses = torch.full((pop_size,), torch.finfo(torch.float32).max )
        
        # We store a global best from prior iterations
        self.g_best_fitness = init_fitness[0].detach().clone()
        self.g_best_position= self.m.detach().clone()
        self.n_evals = self.m.shape[0]

        if optimizer is None:
            self.optimizer = torch.optim.Adam([self.m, self.L_tri], lr=lr)
        else:
            self.optimizer = optimizer

    def forward(self):
        """
        Single-eval iteration with elitism:

         1) Sample population x_i = m + L @ eps
         2) Boundary bounce
         3) Evaluate => single objective
         4) Elitism: if old best < new best => replace worst w/ old best
         5) Soft weighting => produce candidate (m_new, L_new)
         6) Return best new fitness as scalar loss
        """
        N, D = self.pop_size, self.dim

        # Lower-tri factor
        L = torch.tril(self.L_tri)

        # (1) Sample population
        eps = torch.randn(N, D, device=self.m.device)
        pop = self.m.unsqueeze(0) + eps @ L.T  # shape [N, dim]

        # (2) Boundary bounce
        lb = self.lower_bound
        ub = self.upper_bound
        pop = torch.max(pop, lb)
        pop = torch.min(pop, ub)

        # (3) Evaluate
        fitnesses = self.obj_func(pop)  # shape [N]
        self.n_evals += pop.shape[0]
        
        best_fit, _ = torch.min(fitnesses, dim=0)
        _, worst_idx = torch.max(fitnesses, dim=0)

        # (4) Elitism: if self.g_best_fitness < best_fit => replace worst
        replaced_pop = pop.clone()
        replaced_fit = fitnesses.clone()

        if self.g_best_fitness < best_fit:
            replaced_pop[worst_idx] = self.g_best_position
            replaced_fit[worst_idx] = self.g_best_fitness

        # Now final best after elitism
        final_best_val, final_best_idx = torch.min(replaced_fit, dim=0)

        # (5) Soft weighting => new (m, L) using replaced_pop
        scores = -replaced_fit
        w = torch.softmax(scores, dim=0)
        m_new = (w.unsqueeze(1)*replaced_pop).sum(dim=0)

        # Cov update
        diffs = replaced_pop - m_new.unsqueeze(0)
        C_new = torch.zeros(D, D, dtype=torch.float32, device=pop.device)
        for i in range(N):
            C_new += w[i]*(diffs[i].unsqueeze(1) @ diffs[i].unsqueeze(0))
        # factor
        diag_idx = torch.arange(D, device=pop.device)
        C_new[diag_idx, diag_idx] += 1e-5
        # C_new = 0.5*(C_new + C_new.T)
        
        L_new = torch.linalg.cholesky(C_new)

        # (6) Store candidate
        self._cand_pop = replaced_pop
        self._cand_fitnesses = replaced_fit
        self._cand_best_fit = final_best_val
        self._cand_best_idx = final_best_idx
        self._cand_m = m_new
        self._cand_L = L_new
        
        return final_best_val

    def update_state(self):
        """
        """
        with torch.no_grad():
            # commit new m, L
            self.m.copy_(self._cand_m)
            self.m.requires_grad_(True)

            self.L_tri.copy_(self._cand_L)
            self.L_tri.requires_grad_(True)
            
            self.fitnesses.copy_(self._cand_fitnesses.detach())

            # possibly update global best
            if self._cand_best_fit < self.g_best_fitness:
                self.g_best_fitness.copy_(self._cand_best_fit.detach())
                self.g_best_position.copy_(self._cand_pop[self._cand_best_idx].detach())