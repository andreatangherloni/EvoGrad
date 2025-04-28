import torch
import torch.nn as nn
from math import sqrt

class GA(nn.Module):
       
    def __init__(self,
                 obj_func,
                 dim: int,
                 pop_size: int = 30,
                 tau_c=0.8,
                 tau_m=0.1,
                 init_mut_scale=0.1,
                 init_cr=1.0,
                 init_mr=None,
                 crossover="sbx",      # "sbx", "undx", or "blend"
                 mutation="pm",
                 eta_c=15.0,
                 eta_m=20.0,
                 lower_bound=None,
                 upper_bound=None,
                 initialisation='uniform',
                 log_movement=False,
                 seed: int | None = 0,
                 lr: float = 0.001,
                 device=None
                ):
        
        super().__init__()
        
        self.obj_func = obj_func
        self.dim      = dim
        self.pop_size = pop_size
        self.crossover = crossover
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
            
        if self.log_movement:
            self.register_buffer("lb", torch.tensor([0] * dim).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor([1] * dim).float().unsqueeze(0))
            self.register_buffer("actual_lb", torch.tensor(lower_bound).float().unsqueeze(0))
            self.register_buffer("actual_ub", torch.tensor(upper_bound).float().unsqueeze(0))
        else:            
            self.register_buffer("lb", torch.tensor(lower_bound).float().unsqueeze(0))
            self.register_buffer("ub", torch.tensor(upper_bound).float().unsqueeze(0))

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
                    
        if initialisation == "log" and self.log_movement==False:
            genes = torch.exp(torch.log(self.lb) + torch.log(self.ub/self.lb)*torch.rand(pop_size, dim))
        else:
            genes = self.lb + (self.ub - self.lb) * torch.rand(pop_size, dim)
        
        self.genes = nn.Parameter(genes)

        self.tau_c     = nn.Parameter(torch.tensor([tau_c]))
        self.tau_m     = nn.Parameter(torch.tensor([tau_m]))
        self.mut_scale = nn.Parameter(torch.tensor([init_mut_scale]))
        self.cr_logits = nn.Parameter(torch.full((dim,), torch.logit(torch.tensor(init_cr))))
        
        if init_mr is None or init_mr == 0:
            init_mr = 1.0 / dim  # sensible default
        
        logit_mr0 = torch.logit(torch.tensor(init_mr))
        # self.mr_logit = nn.Parameter(logit_mr0.clone())
        self.mr_logits = nn.Parameter(torch.full((dim,), logit_mr0))
        
        # Learnable so the optimiser can anneal them)
        self.eta_c = nn.Parameter(torch.tensor([eta_c]))
        self.eta_m = nn.Parameter(torch.tensor([eta_m]))

        with torch.no_grad():
            if log_movement:
                x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*self.genes
                x = 10**x
                f0 = obj_func(x)
            else:
                f0 = obj_func(self.genes)
        
        self.fitnesses = f0.clone()
        self.fitnesses = self.fitnesses.to(self.device)
        self.n_evals   = pop_size
        
        g_idx = torch.argmin(self.fitnesses)
        self.best_f = self.fitnesses[g_idx].clone()
        self.best_x = self.genes[g_idx].clone()

        self.optimizer = torch.optim.Adam([self.genes,
                                           self.tau_c,
                                           self.tau_m,
                                           self.mut_scale,
                                           self.cr_logits,
                                           self.mr_logits,
                                           self.eta_c,
                                           self.eta_m],
                                          lr=lr)    

    # ---------------- Gumbel reparam => alpha => continuous mixture #
    def _soft_parent(self, logp):
        g = -torch.log(-torch.log(torch.rand_like(logp, device=self.device) + 1e-8) + 1e-8)
        alpha = torch.softmax(logp + g, dim=0)
        return (alpha.unsqueeze(1) * self.genes).sum(dim=0)
    
    def _sbx(self, p1, p2, cr):
        D = p1.shape[0]
        eta = torch.clamp(self.eta_c, 1.0, 100.0)
        # randomly decide which coords undergo SBX
        mask = torch.rand(D, device=p1.device) < cr
        u    = torch.rand(D, device=p1.device)
        beta = torch.where(u <= 0.5,
                           (2*u)**(1/(eta+1)),
                           (2*(1-u))**(-1/(eta+1)))
        beta = torch.where(mask, beta, torch.ones_like(beta))
        c1   = 0.5*((1+beta)*p1 + (1-beta)*p2)
        return c1
    
    def _undx(self, p1, p2, p3):
        # Implements Minimal‐Normal UNDX (Takagi + Ono)
        base = 0.5*(p1 + p2)
        d    = p2 - p1
        norm = d.norm()
        e_d  = d / (norm + 1e-12)
        perp = p3 - base
        perp = perp - (perp @ e_d) * e_d
        sigma_xi, sigma_eta = 0.5, 0.35 / sqrt(self.dim)
        xi   = torch.randn(1, device=p1.device) * sigma_xi * norm
        eta  = torch.randn(self.dim, device=p1.device) * sigma_eta * norm
        child = base + xi*e_d + eta*perp/ (perp.norm()+1e-12)
        return child
    
    def _poly_mutate(self, x):
        # Continuous polynomial mutation with Binary-Concrete mask.
        D, dev = x.shape[0], x.device
        eta = torch.clamp(self.eta_m, 1.0, 100.0)

        # ----- Binary-Concrete mask -----------------------------------------
        temp = torch.clamp(self.tau_c, 1e-3, 5.0)
        u = torch.rand(D, device=dev)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.mr_logits) / temp)
        mask = (s > 0.5).float() - s.detach() + s       # straight-through

        # ----- polynomial perturbation -------------------------------------
        u2 = torch.rand(D, device=dev)
        mut_pow = 1.0 / (eta + 1.0)
        delta = torch.where(
            u2 < 0.5,
            (2.0 * u2) ** mut_pow - 1.0,
            1.0 - (2.0 * (1.0 - u2)) ** mut_pow
        )

        y = x + mask * delta * (self.ub[0] - self.lb[0])
        return y


    def forward(self):
        N, D = self.pop_size, self.dim
        temp = torch.clamp(self.tau_c, 1e-3, 5.0)

        logp = -self.fitnesses / temp
                
        cr   = torch.sigmoid(self.cr_logits)
        mut  = torch.clamp(self.mut_scale, 1e-5, 10.0)

        # Crossover and mutation
        offspring = []
        for _ in range(N):
            p1 = self._soft_parent(logp)
            p2 = self._soft_parent(logp)
            if self.crossover == "sbx":
                child = self._sbx(p1, p2, cr)            # SBX
            elif self.crossover == "undx":
                p3 = self._soft_parent(logp)
                child = self._undx(p1, p2, p3)           # UNDX
            else:                                        
                child = cr * p1 + (1 - cr) * p2
            
            if self.mutation == "pm":
                child = self._poly_mutate(child)
            else:
                child += mut * torch.randn(D, device=self.device)    # Gaussian mutation

            # boundary bounce
            lo, hi = child < self.lb, child > self.ub
            child = torch.where(lo, 2*self.lb - child, child)
            child = torch.where(hi, 2*self.ub - child, child)
            
            offspring.append(child)
        offspring = torch.cat(offspring, 0)

        if self.log_movement:
            x = torch.log10(self.actual_ub) + (torch.log10(self.actual_lb) - torch.log10(self.actual_ub))*offspring
            x = 10**x
            fit_offspring = self.obj_func(x)
        else:
            fit_offspring = self.obj_func(offspring)
                
        self.n_evals += N

        best_old, idx_old = torch.min(self.fitnesses, 0)
        best_kid, _       = torch.min(fit_offspring, 0)
        pop_new, fit_new  = offspring.clone(), fit_offspring.clone()
        
        if best_old < best_kid:
            worst = torch.argmax(fit_offspring)
            pop_new[worst] = self.genes[idx_old]
            fit_new[worst] = best_old

        best_val, best_idx = torch.min(fit_new, 0)
        self._cand = {
            "population": pop_new,
            "fitness":    fit_new,
            "best_f":     best_val,
            "best_x":     pop_new[best_idx].detach(),
            "hyperparams": {
                "selection_temp": temp.detach(),
                "mutation_scale": mut.detach(),
                "crossover_rate": cr.detach(),
            },
        }
        return best_val

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_state(self):
        self.genes.copy_(self._cand["population"])
        self.genes.requires_grad_(True)
        self.fitnesses.copy_(self._cand["fitness"].detach())

        if self._cand["best_f"] < self.best_f:
            self.best_f.copy_(self._cand["best_f"].detach())
            self.best_x.copy_(self._cand["best_x"])

        self.optimizer.zero_grad(set_to_none=True)