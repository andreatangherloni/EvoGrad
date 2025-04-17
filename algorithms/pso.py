import torch
import torch.nn as nn

class PSO(nn.Module):
        
    def __init__(self,
                obj_func,
                dim: int,
                pop_size: int = 30,
                init_inertia=0.7,
                init_cognitive=1.4,
                init_social=1.4,
                init_v_min=-1,
                init_v_max=1,
                lower_bound=None,
                upper_bound=None,
                seed: int | None = 0,
                lr: float = 0.001,
                device=None
                ):
    
        super().__init__()
        
        self.obj_func = obj_func
        self.dim      = dim
        self.pop_size = pop_size
        
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
        
        self.register_buffer("lb", torch.tensor(lower_bound).float().unsqueeze(0))
        self.register_buffer("ub", torch.tensor(upper_bound).float().unsqueeze(0))

        # state --------------------------------------------------------------
        init_pos = self.lb + (self.ub - self.lb) * torch.rand(pop_size, dim)
        self.pos = nn.Parameter(init_pos)
        self.vel = torch.zeros_like(self.pos, device=self.device)

        with torch.no_grad():
            f0 = obj_func(self.pos)
        
        self.fitnesses = f0.clone()
        self.fitnesses = self.fitnesses.to(self.device)
        self.n_evals   = pop_size

        self.p_best_pos = self.pos.clone().detach()
        self.p_best_fit = self.fitnesses.clone().detach()
        
        self.p_best_pos =  self.p_best_pos.to(self.device)
        self.p_best_fit =  self.p_best_fit.to(self.device)
        
        g_idx = torch.argmin(self.p_best_fit)
        self.best_f = self.p_best_fit[g_idx].clone()
        self.best_x = self.p_best_pos[g_idx].clone()

        # learnable hyper‑parameters ----------------------------------------
        eye = torch.ones(pop_size, 1)
        self.inertia   = nn.Parameter(init_inertia   * eye)
        self.cognitive = nn.Parameter(init_cognitive * eye)
        self.social    = nn.Parameter(init_social    * eye)
        self.v_min     = nn.Parameter(init_v_min * eye)
        self.v_max     = nn.Parameter(init_v_max * eye)
        
        self.optimizer = torch.optim.Adam([self.pos,
                                           self.inertia,
                                           self.cognitive,
                                           self.social,
                                           self.v_min,
                                           self.v_max
                                           ],
                                          lr=lr) 

    # --------------------------------------------------------------------- #
    def forward(self):
        "one PSO generation (differentiable)"
        r1, r2 = torch.rand_like(self.pos), torch.rand_like(self.pos)
        
        cog = self.cognitive * r1 * (self.p_best_pos - self.pos)
        soc = self.social    * r2 * (self.best_x - self.pos)

        vel_new = self.inertia * self.vel + cog + soc
        vel_new = torch.clamp(vel_new, min=self.v_min, max=self.v_max)
        
        pos_new = self.pos + vel_new
        mask_lo, mask_hi = pos_new < self.lb, pos_new > self.ub
        
        pos_new = torch.where(mask_lo, 2 * self.lb - pos_new, pos_new)
        pos_new = torch.where(mask_hi, 2 * self.ub - pos_new, pos_new)
        vel_new = torch.where(mask_lo | mask_hi, -vel_new, vel_new)

        fit_new = self.obj_func(pos_new)
        self.n_evals += self.pop_size

        # Personal best -----------------------------------------------------
        improved = fit_new < self.p_best_fit
        p_best_pos_new = torch.where(improved.unsqueeze(1), pos_new, self.p_best_pos)
        p_best_fit_new = torch.where(improved,             fit_new, self.p_best_fit)

        best_val, best_idx = torch.min(fit_new, 0)

        # Candidate dict ----------------------------------------------------
        self._cand = {
            "positions":  pos_new,
            "velocities": vel_new,
            "fitness":    fit_new,
            "best_f":     best_val,
            "best_x":     pos_new[best_idx].detach(),
            "p_best_pos": p_best_pos_new,
            "p_best_fit": p_best_fit_new,
            "hyperparams": {
                "inertia":   self.inertia.detach(),
                "cognitive": self.cognitive.detach(),
                "social":    self.social.detach(),
                "v_min":     self.v_min.detach(),
                "v_max":     self.v_max.detach(),
            },
        }
        return best_val

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def update_state(self):
        self.pos.copy_(self._cand["positions"])
        self.pos.requires_grad_(True)
        self.vel.copy_(self._cand["velocities"])
        self.fitnesses.copy_(self._cand["fitness"].detach())

        self.p_best_pos.copy_(self._cand["p_best_pos"].detach())
        self.p_best_fit.copy_(self._cand["p_best_fit"].detach())

        if self._cand["best_f"] < self.best_f:
            self.best_f.copy_(self._cand["best_f"].detach())
            self.best_x.copy_(self._cand["best_x"])