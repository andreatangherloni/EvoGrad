import torch
import torch.nn as nn

class DE(nn.Module):
        
    def __init__(self,
                obj_func,
                dim: int,
                pop_size: int = 30,
                tau_s=1.,
                tau_c=1.,
                hard=True,
                init_F=0.5,
                init_cr=0.9,
                mutation="rand/1",
                crossover="binomial",
                eta_c=15.0,
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
        
        self.obj_func  = obj_func
        self.dim       = dim
        self.pop_size  = pop_size
        self.mutation  = mutation.lower()
        self.crossover = crossover.lower()
        self.log_movement = log_movement
        self.elitism = elitism
        self.hard = hard
        
        if device is None:
            
            self.device = "cpu" 

            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda" 
        else:
            self.device = device

        # Set the boundaries of the search space
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
        self.tau_s     = nn.Parameter(torch.tensor([tau_s], device=self.device))
        self.tau_c     = nn.Parameter(torch.tensor([tau_c], device=self.device))
        self.eta_c     = nn.Parameter(torch.tensor([eta_c], device=self.device))
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
                                           self.tau_s,
                                           self.tau_c,
                                           self.F,
                                           self.cr_logits],
                                          lr=lr)  

    # Gumbel Softmax selection
    def _gumbel_softmax(self, logits, dim = -1, eps = 1e-8):
        
        temp = self.tau_s
        
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y_soft = torch.softmax((logits + gumbels) / temp, dim=dim)

        if self.hard:
            # Straight-through: y_hard - y_soft is detached, so gradients flow through y_soft
            index  = y_soft.argmax(dim, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
            y_soft = (y_hard - y_soft).detach() + y_soft
        
        return y_soft
    
    # Binomial crossover
    def _binomial_crossover(self, p1, p2):
        N, D   = p1.shape
        temp   = self.tau_c

        # Binary-Concrete mask (straight-through)
        u      = torch.rand(N, D, device=self.device)
        logits = self.cr_logits
        s      = torch.sigmoid((torch.log(u) - torch.log1p(-u) + logits) / temp)
        mask   = (s > 0.5).float() - s.detach() + s   # STE  ∈{0,1}

        # Guarantee at least one donor gene
        j_rand = torch.randint(0, D, (N,), device=self.device)
        mask[torch.arange(N, device=self.device), j_rand] = 1.0

        return mask * p2 + (1.0 - mask) * p1
    
    #  Exponential crossover (a contiguous segment of donor genes)
    def _exponential_crossover(self, p1, p2):
       
        N, D   = p1.shape
        cr     = torch.sigmoid(self.cr_logits)
        cr_val = cr.mean()

        # Random start position for each offspring 
        j_rand = torch.randint(0, D, (N,), device=self.device)

        # Random numbers to decide segment length
        U      = torch.rand(N, D, device=self.device)               # (N, D)
        # Roll so that column 0 is the starting gene
        cols   = torch.arange(D, device=self.device).unsqueeze(0)   # (1, D)
        indices= (cols - j_rand.unsqueeze(1)) % D                   # (N, D)
        U_roll = U.gather(1, indices)                               # (N, D)

        cont   = (U_roll < cr_val).float()                          # <CR ? 1:0
        cont[:, 0] = 1.0                                            # Always copy first gene
        seg    = torch.cumprod(cont, dim=1)                         # 1 until first 0

        # Scatter segment back to original gene order
        mask   = torch.zeros_like(seg)
        mask.scatter_(1, indices, seg)

        # straight-through: treat mask as hard in fwd, soft grads in bwd
        mask = mask.detach()

        return mask * p2 + (1.0 - mask) * p1
    
    # SBX crossover
    def _sbx_crossover(self, p1, p2):
        N, D = p1.shape
        eta = self.eta_c

        # Binary-Concrete mask (straight-through)
        temp = self.tau_c
        
        u = torch.rand(N, D, device=self.device)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.cr_logits) / temp)
        mask = (s > 0.5).float() - s.detach() + s       # straight-through

        # SBX coefficients 
        u2    = torch.rand(N, D, device=self.device)
        beta  = torch.where(u2 <= 0.5,
                            (2*u2)**(1/(eta+1)),
                            (2*(1-u2))**(-1/(eta+1)))

        beta  = mask * beta + (1.0 - mask) # 1.0 means “no crossover”
        child = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        return child

    # Blend crossover
    def _blend_crossover(self, p1, p2):
        alpha  = torch.rand_like(p1)
        return alpha * p1 + (1.0 - alpha) * p2
    
    # Boundary bounce
    def _reflect_bounds(self, x):
        span = self.ub - self.lb
        x = (x - self.lb) % (2 * span)
        x = torch.where(x > span, 2*span - x, x)
        return self.lb + x

    # Forward pass
    def forward(self):
        N, D = self.pop_size, self.dim
   
        # 1. Soft selection  (Gumbel-Softmax)
        temp  = self.tau_s
        logp  = (-self.fitnesses / temp).expand(N, -1) 
        
        F   = self.F
        cr  = torch.sigmoid(self.cr_logits)
        
        # 2. Mutation
        if self.mutation.startswith("current-to-best"):
            x_best = self.best_x
            
            alpha1 = self._gumbel_softmax(logp, dim=1)
            alpha2 = self._gumbel_softmax(logp, dim=1)

            r1 = alpha1 @ self.pop
            r2 = alpha2 @ self.pop
            
            v = self.pop + F*(x_best - self.pop) + F*(r1 - r2)
        
        elif self.mutation.startswith("rand/1"):
            alpha1 = self._gumbel_softmax(logp, dim=1)
            alpha2 = self._gumbel_softmax(logp, dim=1)
            alpha3 = self._gumbel_softmax(logp, dim=1)

            r1 = alpha1 @ self.pop
            r2 = alpha2 @ self.pop
            r3 = alpha3 @ self.pop
            
            v = r1 + F*(r2 - r3)
        
        else:
            raise ValueError(f"Unknown crossover '{self.mutation}'")
        
        # 3. Crossover
        if self.crossover in ["bin", "binomial"]:
            offspring = self._binomial_crossover(self.pop, v)
        
        elif self.crossover in ["exp", "exponential"]:
            offspring = self._exponential_crossover(self.pop, v)
        
        elif self.crossover in ["sbx"]:
            offspring = self._sbx_crossover(self.pop, v)
        
        elif self.crossover == "blend":
            offspring = self._blend_crossover(self.pop, v)
        
        else:
            raise ValueError(f"Unknown crossover '{self.crossover}'")
            
        # 4. Keep inside bounds (reflect)
        offspring = self._reflect_bounds(offspring)
        
        if self.log_movement:
            x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*offspring
            x = 10**x
            fit_offspring = self.obj_func(x)
        else:
            fit_offspring = self.obj_func(offspring)

        self.n_evals += N
        
        # “one‑to‑one” replacement  + elitism
        better = fit_offspring < self.fitnesses # vector of booleans, 1‑per‑parent
        pop_new = torch.where(better.unsqueeze(1), offspring.clone(), self.pop.clone())
        fit_new = torch.where(better, fit_offspring.clone(), self.fitnesses.clone())
        
        if self.elitism:
            best_old, idx_old = torch.min(self.fitnesses, 0)
            best_kid, _       = torch.min(fit_offspring, 0)
        
            if best_old < best_kid:
                worst = torch.argmax(fit_offspring)
                pop_new[worst] = self.pop[idx_old]
                fit_new[worst] = best_old

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
        
        self.F.clamp_(min=1e-8, max=2.0)
        self.tau_s.clamp_(min=1e-5, max=5)
        self.tau_c.clamp_(min=1e-5, max=5)
        self.eta_c.clamp_(min=0.1,  max=100.0)

        if self._cand["best_f"] < self.best_f:
            self.best_f.copy_(self._cand["best_f"].detach())
            self.best_x.copy_(self._cand["best_x"])
        
        self.history["best_f"].append(self.best_f.clone().item())
        self.history["best_x"].append(self.best_x.clone())