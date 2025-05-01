import torch
import torch.nn as nn

class DE(nn.Module):
        
    def __init__(self,
                obj_func,
                dim: int,
                pop_size: int = 30,
                init_tau=0.8,
                init_F=2.,
                init_cr=0.9,
                mutation="rand/1",
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
        
        self.obj_func  = obj_func
        self.dim       = dim
        self.pop_size  = pop_size
        self.mutation  = mutation
        self.log_movement = log_movement
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # ---------------- Initial search volume ----------------
        if lower_bound is None:
            lower_bound = [-100.0] * dim
        if upper_bound is None:
            upper_bound = [100.0] * dim
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.log_movement:
            self.register_buffer("lb", torch.tensor([0] * dim, device=self.device).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor([1] * dim, device=self.device).float().unsqueeze(0))
            self.register_buffer("actual_lb", torch.tensor(lower_bound, device=self.device).float().unsqueeze(0))
            self.register_buffer("actual_ub", torch.tensor(upper_bound, device=self.device).float().unsqueeze(0))
        else:            
            self.register_buffer("lb", torch.tensor(lower_bound, device=self.device).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor(upper_bound, device=self.device).float().unsqueeze(0))
            
        if initialisation is None:
            initialisation = "uniform"
        else:
            initialisation = initialisation.lower()
             
        if initialisation in ["log", "logarithm"] and log_movement==False:
            pop0 = torch.exp(torch.log(self.lb) + torch.log(self.ub/self.lb)*torch.rand(pop_size, dim, device=self.device))
        
        elif initialisation in ["normal", "norm", "gaussian", "gauss"]:
            mean = 0.5 * (self.lb + self.ub)          # centre of the box
            std  = 0.5 * (self.ub - self.lb) / 3.0    # 3-σ rule  ⇒  99.7 % inside bounds
            pop0 = mean + std * torch.randn(pop_size, dim, device=self.device)
            pop0 = torch.max(torch.min(pop0, self.ub), self.lb)
        else:
            pop0 = self.lb + (self.ub - self.lb) * torch.rand(pop_size, dim, device=self.device)
        
        self.pop = nn.Parameter(pop0)

        self.F         = nn.Parameter(torch.tensor([init_F], device=self.device))
        self.tau       = nn.Parameter(torch.tensor([init_tau], device=self.device))
        self.cr_logits = nn.Parameter(torch.full((dim,), torch.logit(torch.tensor(init_cr)), device=self.device))

        with torch.no_grad():
            if log_movement:
                x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*self.pop
                x = 10**x
                f0 = obj_func(x)
            else:
                f0 = obj_func(self.pop)
        
        self.fitnesses = f0.clone()
        self.fitnesses = self.fitnesses.to(self.device)
        self.n_evals   = pop_size
        
        g_idx = torch.argmin(self.fitnesses)
        self.best_f = self.fitnesses[g_idx].clone()
        self.best_x = self.pop[g_idx].clone()
        self.best_f = self.best_f.to(self.device)
        self.best_x = self.best_x.to(self.device)
        
        self.history["best_f"].append(self.best_f.clone().item())
        self.history["best_x"].append(self.best_x.clone())
        
        self.optimizer = torch.optim.Adam([self.pop,
                                           self.tau,
                                           self.F,
                                           self.cr_logits],
                                          lr=lr)  

    # ---------------- Gumbel reparam => alpha => continuous mixture #
    def _soft_parent(self, logp):
        g = -torch.log(-torch.log(torch.rand_like(logp, device=self.device) + 1e-8) + 1e-8)
        alpha = torch.softmax(logp + g, dim=0)
        return (alpha.unsqueeze(1) * self.pop).sum(dim=0)
    
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

    def forward(self):
        N, D = self.pop_size, self.dim
        temp = torch.clamp(self.tau, 1e-3, 5.0)
        logp = -self.fitnesses / temp
        F    = torch.clamp(self.F, 1e-8, 2.0)
        cr   = torch.sigmoid(self.cr_logits)
        
        # Offspring generation
        offspring = []
        for i in range(N):
            x_i = self.pop[i]

            if self.mutation.startswith("current-to-best"):
                # x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
                x_best = self.best_x
                r1 = self._soft_parent(logp)
                r2 = self._soft_parent(logp)
                v = x_i + F*(x_best - x_i) + F*(r1 - r2)
            else:  # rand/1
                r1, r2, r3 = (self._soft_parent(logp) for _ in range(3))
                v = r1 + F*(r2 - r3)
                
            mask = torch.rand(D, device=self.device) < cr
            j_rand = torch.randint(0, D, (1,), device=self.device)   # ensure ≥1 donor gene
            mask[j_rand] = True 
            child = torch.where(mask, v, x_i)

            # boundary bounce
            child = self._reflect_bounds(child)
            
            offspring.append(child)
        
        offspring = torch.cat(offspring, 0)
        
        if self.log_movement:
            x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*offspring
            x = 10**x
            fit_offspring = self.obj_func(x)
        else:
            fit_offspring = self.obj_func(offspring)
        
        self.n_evals += N
        
        # “one‑to‑one” replacement 
        better = fit_offspring < self.fitnesses          # vector of booleans, 1‑per‑parent
        pop_new = torch.where(better.unsqueeze(1), offspring, self.pop)
        fit_new = torch.where(better, fit_offspring, self.fitnesses)

        best_val, best_idx = torch.min(fit_new, 0)

        self._cand = {
            "population": pop_new,
            "fitness":    fit_new,
            "best_f":     best_val,
            "best_x":     pop_new[best_idx].detach(),
            "hyperparams": {
                "F":                F.detach(),
                "selection_temp":   temp.detach(),
                "crossover_rate":   cr.detach(),
            },
        }
        return best_val

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_state(self):
        self.pop.copy_(self._cand["population"])
        self.pop.requires_grad_(True)
        self.fitnesses.copy_(self._cand["fitness"].detach())

        if self._cand["best_f"] < self.best_f:
            self.best_f.copy_(self._cand["best_f"].detach())
            self.best_x.copy_(self._cand["best_x"])
        
        self.history["best_f"].append(self.best_f.clone().item())
        self.history["best_x"].append(self.best_x.clone())