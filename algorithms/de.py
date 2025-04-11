import torch
import torch.nn as nn

class DE(nn.Module):
    """
    """
    def __init__(self,
                 obj_func,
                 dim,
                 pop_size=30,
                 lower_bound=None,
                 upper_bound=None,
                 init_F=2,
                 init_selection_temp=1.0,
                 initial_crossover=0.9,
                 lr=0.001,
                 optimizer=None,
                 seed=42
                 ):
        
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
        self.register_buffer("lower_bound",
            torch.tensor(lower_bound, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("upper_bound",
            torch.tensor(upper_bound, dtype=torch.float32).unsqueeze(0))

        # population
        init_pop = (self.lower_bound
                    + (self.upper_bound - self.lower_bound)
                    * torch.rand(pop_size, dim))
        self.population = nn.Parameter(init_pop)  # shape [N, dim]

        # F as a learnable scale
        self.F = nn.Parameter(torch.tensor([init_F], dtype=torch.float32))
        # selection temperature
        self.selection_temp   = nn.Parameter(torch.tensor([init_selection_temp], dtype=torch.float32))
        self.crossover_logits = nn.Parameter(torch.ones(dim)*initial_crossover)

        # Evaluate initial
        with torch.no_grad():
            init_fitnesses = self.obj_func(self.population)
        
        self.fitnesses = init_fitnesses.clone()
        self.n_evals = self.population.shape[0]
            
        best_idx = torch.argmin(self.fitnesses)
        self.g_best_fitness = self.fitnesses[best_idx].detach().clone()
        self.g_best_position= self.population[best_idx].detach().clone()

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                [self.population, self.F, self.selection_temp, self.crossover_logits], lr=lr)
        else:
            self.optimizer = optimizer

    def forward(self):
        """
        """
        N, D = self.pop_size, self.dim
        pop_old = self.population  # shape [N, dim]
        old_fit = self.fitnesses   # shape [N]

        # find old best
        old_best_val, old_best_idx = torch.min(old_fit, dim=0)

        # for soft selection
        scores = -old_fit  # shape [N]
        temp_clamp = torch.clamp(self.selection_temp, min=1e-3, max=5.0)
        logp = scores / temp_clamp

        # difference scale
        F_clamp = torch.clamp(self.F, min=1e-8, max=2.0)

        def soft_sel():
            # Gumbel reparam => alpha => continuous mixture
            g = -torch.log(-torch.log(torch.rand_like(logp) + 1e-8) + 1e-8)
            alpha = torch.softmax(logp + g, dim=0)  # shape [N]
            # parent = sum_i alpha_i * pop_old[i]
            return (alpha.unsqueeze(1)*pop_old).sum(dim=0)  # shape [dim]
        
        def dimension_wise_crossover(p1, trial):
            """
            Returns a child, where each dimension j
            is alpha_j*trial[j] + (1-alpha_j)*p1[j],
            with alpha_j = sigmoid(self.crossover_logits[j]).
            """
            alpha = torch.sigmoid(self.crossover_logits)  # shape [dim]
            # child = alpha * trial + (1 - alpha) * p1
            return alpha*trial + (1 - alpha)*p1
        
        # build candidate pop
        new_pop_list = []
        for i in range(N):
            p1 = soft_sel()
            p2 = soft_sel()
            p3 = soft_sel()

            trial = p1 + F_clamp*(p2 - p3)
            # blend crossover
            # alpha = torch.rand(1)
            # child = alpha*trial + (1-alpha)*p1
            # dimension-wise crossover
            child = dimension_wise_crossover(p1, trial)

            # boundary bounce
            mask_lower = child < self.lower_bound[0]
            mask_upper = child > self.upper_bound[0]
            bounce_lower= 2*self.lower_bound[0] - child
            bounce_upper= 2*self.upper_bound[0] - child
            child = torch.where(mask_lower, bounce_lower, child)
            child = torch.where(mask_upper, bounce_upper, child)

            new_pop_list.append(child.unsqueeze(0))

        candidate_pop = torch.cat(new_pop_list, dim=0)  # [N, dim]

        # single objective call => new fitness
        cand_fit = self.obj_func(candidate_pop)
        self.n_evals += candidate_pop.shape[0]

        # find new best & worst
        cand_best_val, _ = torch.min(cand_fit, dim=0)
        _, cand_worst_idx= torch.max(cand_fit, dim=0)

        # Elitism: if old_best_val < cand_best_val => replace worst
        replaced_pop = candidate_pop.clone()
        replaced_fit = cand_fit.clone()

        if old_best_val < cand_best_val:
            # replace the worst
            replaced_pop[cand_worst_idx] = pop_old[old_best_idx]
            replaced_fit[cand_worst_idx] = old_best_val

        # final best after elitism
        final_best_val, final_best_idx = torch.min(replaced_fit, dim=0)

        # store
        self._cand_positions = replaced_pop
        self._cand_fitnesses = replaced_fit
        self._cand_best_fit  = final_best_val
        self._cand_best_idx  = final_best_idx

        # single scalar loss => synergy with backprop
        return final_best_val


    def update_state(self):
        """
        """
        with torch.no_grad():
            self.population.copy_(self._cand_positions)
            self.population.requires_grad_(True)
            
            # finalize fitnesses for the new population
            self.fitnesses.copy_(self._cand_fitnesses.detach())

            if self._cand_best_fit  < self.g_best_fitness:
                self.g_best_fitness.copy_(self._cand_best_fit.detach())
                self.g_best_position.copy_(self._cand_positions[self._cand_best_idx].detach())