import math
import torch
import torch.nn as nn

def _expected_norm(dim: int) -> float:
    """
    E||N(0, I)||  (accurate to O(1/dim))
    Ref: Hansen 2013 tutorial
    """
    d = float(dim)
    return math.sqrt(d) * (1.0 - 1. / (4 * d) + 1. / (21 * d * d))

def _heuristic_eps(min_abs_L):
    # log–log rule: eps = 1e-15 * min_abs_L^{-1.75}
    log_eps = -15.0 - 1.75 * math.log10(min_abs_L)
    eps_adapt = math.pow(10.0, log_eps)

    # keep ε in [1e-7, 0.1] to avoid over/underflow
    return float(min(max(eps_adapt, 1e-7), 1e-1))


class CMAES(nn.Module):

    def __init__(self,
                 obj_func,
                 dim: int,
                 pop_size: int = 30,
                 init_sigma: float = 0.5,
                 elitism=False,
                 lower_bound=None,
                 upper_bound=None,
                 initialisation='uniform',
                 log_movement=False,
                 seed: int | None = 0,
                 lr: float = 0.01,
                 device=None
                 ):
        
        super().__init__()
        
        self.history = {}
        self.history["best_f"] = []
        self.history["best_x"] = []
        
        self.obj_func = obj_func
        self.dim      = dim
        self.pop_size = pop_size
        self.mu       = pop_size // 2  # number of selected parents
        self.log_movement = log_movement
        self.eps = 1e-14 * (dim ** 7)
        self.eps = float(min(self.eps, 1e-1))
        self.elitism = elitism
                
        self.fitnesses = torch.full((pop_size,), torch.finfo(torch.float32).max)
        
        if device is None:
            
            self.device = "cpu" 

            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda" 
        else:
            self.device = device

        # ---------------- Initial search volume ----------------
        if lower_bound is None:
            lower_bound = [-100.0] * dim
        if upper_bound is None:
            upper_bound = [100.0] * dim
            
        if self.log_movement:
            self.register_buffer("lb", torch.tensor([0] * dim, device=self.device).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor([1] * dim, device=self.device).float().unsqueeze(0))
            self.register_buffer("actual_lb", torch.tensor(lower_bound, device=self.device).float().unsqueeze(0))
            self.register_buffer("actual_ub", torch.tensor(upper_bound, device=self.device).float().unsqueeze(0))
        else:            
            self.register_buffer("lb", torch.tensor(lower_bound, device=self.device).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor(upper_bound, device=self.device).float().unsqueeze(0))

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        if initialisation is None:
            initialisation = "uniform"
        else:
            initialisation = initialisation.lower()
        
        # Mean parameter      
        if initialisation in ["log", "logarithm"] and log_movement==False:
            init_m = 0.5 * (torch.log(self.lb) + torch.log(self.ub))
        else:
            init_m = 0.5 * (self.lb + self.ub)
        
        std_vec = torch.full_like(init_m, init_sigma)
        m0 = torch.normal(mean=init_m, std = std_vec).to(self.device)
        
        m0 = self._reflect_bounds(m0)     
        self.m = nn.Parameter(m0)

        # log‑sigma so that \sigma = exp(\sigma) is positive and differentiable
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma), device=self.device))

        # Cholesky factor of C (normalised to det = 1 initially)
        self.L_tri = nn.Parameter(torch.eye(dim, dtype=torch.float32, device=self.device))

        # ---------------- Needed if when learnable hyperparameters -----------------------
        # They are in (0,1); we can store logits and map with sigmoid
        def _inv_sigmoid(x):
            x = min(max(x, self.eps), 1 - self.eps)
            return math.log(x / (1 - x))

        # ---------------- Not optimised with backpropagation) ------------
        self.register_buffer("raw_cc",   torch.tensor((4.0 / dim), device=self.device))
        self.register_buffer("raw_cs",   torch.tensor((4.0 / dim), device=self.device))
        self.register_buffer("raw_c1",   torch.tensor((2.0 / dim**2), device=self.device))
        self.register_buffer("raw_cmu",  torch.tensor((0.4), device=self.device))
        self.register_buffer("raw_damp", torch.tensor(math.log(1.0 + 2.0), device=self.device))
        self.register_buffer("p_c",      torch.zeros(dim, device=self.device))
        self.register_buffer("p_sigma",  torch.zeros(dim, device=self.device))
        self.register_buffer("iter_idx", torch.tensor(0, device=self.device))
        
        with torch.no_grad():
            if self.log_movement:
                x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*self.m
                x = 10**x                
                f0 = self.obj_func(x)
            else:
                f0 = self.obj_func(self.m)
        
        idx_sorted = torch.argsort(f0)         
        self.register_buffer("best_f", f0[idx_sorted[0]].clone())
        self.register_buffer("best_x", self.m.squeeze(0).clone().detach())
        self.n_evals = 1
        
        self.history["best_f"].append(self.best_f.clone().item())
        self.history["best_x"].append(self.best_x.clone())
        
        self.pop = self.m.squeeze(0).clone().detach().repeat(pop_size, 1)

        # Adam optimiser for exploitation & hyperparameters learning
        self.optimizer = torch.optim.Adam([self.m,
                                           self.log_sigma,
                                           self.L_tri
                                           ],
                                          lr=lr)

    def _decode_coeffs(self):
        """Decode sigmoids & clamp to obey c1 + c_mu ≤ 1"""
        cc  = torch.sigmoid(self.raw_cc)
        cs  = torch.sigmoid(self.raw_cs)
        c1  = torch.sigmoid(self.raw_c1)
        cmu = torch.sigmoid(self.raw_cmu)

        total = (c1 + cmu).item() 
        if total > 0.999: # softly rescale to keep valid
            c1 = c1 / total * 0.999
            cmu = cmu / total * 0.999
        damps = self.raw_damp.exp()
        return cc, cs, c1, cmu, damps
    
    def _reflect_bounds(self, x):
        """
        Reflect x into [lb, ub] even if |x-lb| > (ub-lb).
        Supports tensors of any shape; lb/ub can be scalars or broadcastable.
        """
        span = self.ub - self.lb
        # map to half-open interval (0, 2·span] then fold with |sin| pattern
        x = (x - self.lb) % (2 * span)            # modulo 2·span
        x = torch.where(x > span, 2*span - x, x)
        return self.lb + x

    # -------------------------------------------------------------------------
    # Main differentiable evolutionary generation
    # -------------------------------------------------------------------------
    def forward(self):
        """Run one CMAES generation and return the scalar best fitness of that generation"""
        N, D, mu = self.pop_size, self.dim, self.mu
        cc, cs, c1, c_mu, damps = self._decode_coeffs()

        # Lower‑triangular factor (guaranteed tri‑angular) 
        L = torch.tril(self.L_tri)
        
        min_abs_L = torch.abs(L).masked_select(L != 0).min().item()
        self.eps = _heuristic_eps(min_abs_L)

        # (1) sampling (re‑parameterisation semantics) 
        z = torch.randn(N, D, device=self.device)                   # ~ N(0,I)
        y = (L @ z.T).T                                          # shape [N,D]
        self.sigma = self.log_sigma.exp()
        x_offspring = self.m + self.sigma * y                              # phenotypes
        
        # (2) Boundary conditions (boundary bounce)
        x_offspring = self._reflect_bounds(x_offspring)
        
        # (3) Fitness evaluation        
        if self.log_movement:
            x_log = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*x_offspring
            x_log = 10**x_log
            fit_offspring = self.obj_func(x_log)
        else:
            fit_offspring = self.obj_func(x_offspring)
        
        self.n_evals += N
        
        x, fit_new  = x_offspring.clone(), fit_offspring.clone()
                
        if self.elitism:
            worst_idx = torch.argmax(fit_offspring)
            x[worst_idx]       = self.best_x 
            fit_new[worst_idx] = self.best_f

        # (4) Sort and parent weights
        idx_sorted = torch.argsort(fit_new)                     
        idx_sel = idx_sorted[:mu]
        y_sel = y[idx_sel] # parents in y‑space
        
        mu_fac = torch.tensor(mu + 0.5, dtype=torch.float32, device=self.device)
        w_raw = torch.log(mu_fac) - torch.log(torch.arange(1, mu + 1, device=self.device, dtype=torch.float32))
        
        w = w_raw / w_raw.sum() # normalise (\sum w = 1)
        mu_eff = 1.0 / torch.sum(w ** 2)

        y_w = torch.sum(w.view(-1, 1) * y_sel, dim=0) # weighted mean

        # (5) Evolution paths     
        z_w = torch.linalg.solve_triangular(L, y_w.unsqueeze(-1), upper=False ).squeeze(-1)
        
        p_sigma_new = (1 - cs) * self.p_sigma + torch.sqrt(cs * (2 - cs) * mu_eff) * z_w

        # heuristic h_σ  (smoothed hard threshold)
        chi_n = _expected_norm(D)
        norm_p_sigma = p_sigma_new.norm()
        h_sigma = torch.sigmoid(10 * (1.4 + 2.0 / (D + 1) - norm_p_sigma / chi_n))

        p_c_new = (1 - cc) * self.p_c + h_sigma * torch.sqrt(cc * (2 - cc) * mu_eff) * y_w

        # (6) Covariance update
        C = L @ L.T
        rank_mu = (w.view(-1, 1, 1) * y_sel.unsqueeze(-1) * y_sel.unsqueeze(-2)).sum(dim=0)
        C_new = (1 - c1 - c_mu) * C + c1 * torch.ger(p_c_new, p_c_new) + c_mu * rank_mu
        
        # Mumerical safety for Cholesky decomposition     
        diag = torch.arange(D, device=self.device)
        C_new[diag, diag] += self.eps

        try:
            L_new = torch.linalg.cholesky(C_new)
        except:  # not PD → eigen-fix
            eigval, eigvec = torch.linalg.eigh(C_new)
            eigval = torch.clamp(eigval, min=1e-12)
            C_fix = (eigvec * eigval) @ eigvec.T
            L_new = torch.linalg.cholesky(C_fix)

        # (7) step‑size update
        sigma_factor = torch.exp((cs / damps) * (norm_p_sigma / chi_n - 1.0))
        sigma_new = self.sigma * sigma_factor

        m_new = self.m + self.sigma * y_w

        best_val = fit_new[idx_sorted[0]]
        best_x   = x[idx_sorted[0]].detach()
                
        self._cand = {
            "population":  x,
            "m":           m_new,
            "log_sigma":   torch.log(sigma_new),
            "L_tri":       L_new,
            "p_c":         p_c_new,
            "p_sigma":     p_sigma_new,
            "fitness":     fit_new,
            "best_f":      best_val,
            "best_x":      best_x,
            
            "hyperparams": {
                "sigma": sigma_new.detach(),
                "cc":    cc.detach(),
                "cs":    cs.detach(),
                "c1":    c1.detach(),
                "c_mu":  c_mu.detach(),
                "damp":  damps.detach(),
            },
        }
        return best_val
    
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_state(self):
        
        self.pop.copy_(self._cand["population"])
        self.m.copy_(self._cand["m"])
        self.log_sigma.copy_(self._cand["log_sigma"])
        self.L_tri.copy_(self._cand["L_tri"])
        self.p_c.copy_(self._cand["p_c"])
        self.p_sigma.copy_(self._cand["p_sigma"])
        self.fitnesses.copy_(self._cand["fitness"].detach())

        # keep best‑so‑far
        if self._cand["best_f"] < self.best_f:
            self.best_f.copy_(self._cand["best_f"])
            self.best_x.copy_(self._cand["best_x"])
        
        self.history["best_f"].append(self.best_f.clone().item())
        self.history["best_x"].append(self.best_x.clone())