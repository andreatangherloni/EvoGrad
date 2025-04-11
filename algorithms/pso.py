import torch
import torch.nn as nn

class PSO(nn.Module):
    """
    """

    def __init__(self,
                 obj_func,
                 dim,
                 pop_size=30,
                 lower_bound=None,
                 upper_bound=None,
                 init_inertia=0.7,
                 init_cognitive=1.4,
                 init_social=1.4,
                 init_v_min=-1,
                 init_v_max=1,
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

        # Bounds
        if lower_bound is None:
            lower_bound = [-100.0] * dim
        if upper_bound is None:
            upper_bound = [100.0] * dim

        self.register_buffer(
            "lower_bound",
            torch.tensor(lower_bound, dtype=torch.float32).unsqueeze(0)  # shape [1, dim]
        )
        self.register_buffer(
            "upper_bound",
            torch.tensor(upper_bound, dtype=torch.float32).unsqueeze(0)  # shape [1, dim]
        )
        
        # Per-particle velocity bounds
        self.v_min = nn.Parameter(torch.full((pop_size, 1), init_v_min, dtype=torch.float32))
        self.v_max = nn.Parameter(torch.full((pop_size, 1), init_v_max, dtype=torch.float32))
        
        # Positions as a learnable parameter
        init_positions = (self.lower_bound + (self.upper_bound - self.lower_bound)* torch.rand(pop_size, dim))
        self.positions = nn.Parameter(init_positions)  # shape [swarm_size, dim]
        
        # Velocities
        self.velocities = torch.zeros_like(self.positions)

        # Evaluate initial fitness & personal best
        with torch.no_grad():
            init_fitnesses = self.obj_func(self.positions)
        
        self.fitnesses = init_fitnesses.clone()
        self.n_evals   = self.pop_size

        self.p_best_positions = self.positions.clone().detach()
        self.p_best_fitnesses = self.fitnesses.detach()

        # Global best
        best_idx = torch.argmin(self.p_best_fitnesses)
        self.g_best_fitness = self.p_best_fitnesses[best_idx].clone()
        self.g_best_position = self.p_best_positions[best_idx].clone()
        
        # Particle-wise hyperparams
        self.inertia   = nn.Parameter(torch.full((pop_size,1), init_inertia,   dtype=torch.float32))
        self.cognitive = nn.Parameter(torch.full((pop_size,1), init_cognitive, dtype=torch.float32))
        self.social    = nn.Parameter(torch.full((pop_size,1), init_social,    dtype=torch.float32))
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                [self.positions, self.inertia, self.cognitive, self.social,
                 self.v_min, self.v_max],
                lr=lr
            )
        else:
            self.optimizer = optimizer
    
    def forward(self):
        """
        """
        
        # -- Velocity update --
        r1 = torch.rand_like(self.positions)
        r2 = torch.rand_like(self.positions)
        
        cognitive_term = self.cognitive * r1 * (self.p_best_positions - self.positions)
        social_term    = self.social    * r2 * (self.g_best_position - self.positions)
        
        new_velocities = self.inertia * self.velocities + cognitive_term + social_term
        # clamp velocities
        new_velocities = torch.clamp(new_velocities, min=self.v_min, max=self.v_max)

        # candidate positions
        candidate_positions = self.positions + new_velocities
        
        # boundary bounce
        mask_lower    = candidate_positions < self.lower_bound
        bounced_lower = 2*self.lower_bound - candidate_positions
        mask_upper    = candidate_positions > self.upper_bound
        bounced_upper = 2*self.upper_bound - candidate_positions

        candidate_positions = torch.where(mask_lower, bounced_lower, candidate_positions)
        candidate_positions = torch.where(mask_upper, bounced_upper, candidate_positions)
        new_velocities      = torch.where(mask_lower|mask_upper, -new_velocities, new_velocities)

        # Evaluate candidate positions => single objective call
        cand_fitnesses = self.obj_func(candidate_positions)
        self.n_evals += self.pop_size

        # Update personal best: if new_fitness < p_best_fitness => improved
        improved = (cand_fitnesses < self.p_best_fitnesses)
        cand_p_best_positions = torch.where(improved.unsqueeze(1),
                                            candidate_positions,
                                            self.p_best_positions)
        cand_p_best_fitnesses = torch.where(improved, cand_fitnesses, self.p_best_fitnesses)

        # Then define iteration's best as min of new_fitnesses 
        # (or cand_p_best_fitnesses, either approach is possible)
        cand_best_fitness, cand_idx = torch.min(cand_fitnesses, dim=0)

        # store
        self._cand_positions  = candidate_positions
        self._cand_velocities = new_velocities
        self._cand_fitnesses  = cand_fitnesses
        self._cand_best_fit   = cand_best_fitness
        self._cand_best_idx   = cand_idx
        self._cand_p_best_positions = cand_p_best_positions
        self._cand_p_best_fitnesses = cand_p_best_fitnesses

        return cand_best_fitness

    def update_state(self):
        """
        """
        with torch.no_grad():
            # finalize
            self.positions.copy_(self._cand_positions)
            self.positions.requires_grad_(True)

            self.velocities.copy_(self._cand_velocities)
            self.fitnesses.copy_(self._cand_fitnesses.detach())

            self.p_best_positions.copy_(self._cand_p_best_positions.detach())
            self.p_best_fitnesses.copy_(self._cand_p_best_fitnesses.detach())

            # update global best
            if self._cand_best_fit < self.g_best_fitness:
                self.g_best_fitness.copy_(self._cand_best_fit.detach())
                self.g_best_position.copy_(self._cand_p_best_positions[self._cand_best_idx].detach())