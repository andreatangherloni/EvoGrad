import torch
import torch.nn as nn

class GA(nn.Module):
    """
    """
    def __init__(self,
                 obj_func,
                 dim,
                 pop_size=30,
                 lower_bound=None,
                 upper_bound=None,
                 init_selection_temp=1.0,  # Gumbel-Softmax temperature
                 init_mutation_scale=0.1,  # scale for Gaussian mutation
                 initial_crossover=0.9,
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

        # Bounds for bounce
        if lower_bound is None:
            lower_bound = [-1.0] * self.dim
        if upper_bound is None:
            upper_bound = [1.0] * self.dim
        self.register_buffer("lower_bound",
            torch.tensor(lower_bound, dtype=torch.float32).unsqueeze(0)) # [1, dim]
        self.register_buffer("upper_bound",
            torch.tensor(upper_bound, dtype=torch.float32).unsqueeze(0)) # [1, dim]

        # The population as a learnable parameter
        init_genes = (self.lower_bound
                      + (self.upper_bound - self.lower_bound)
                      * torch.rand(pop_size, self.dim))
        self.genes = nn.Parameter(init_genes)  # shape [N, dim]

        # Hyperparameters
        self.selection_temp = nn.Parameter(
            torch.tensor([init_selection_temp], dtype=torch.float32))
        self.mutation_scale = nn.Parameter(
            torch.tensor([init_mutation_scale], dtype=torch.float32))
        
        self.crossover_logits = nn.Parameter(torch.ones(dim)*initial_crossover)

        # Evaluate initial population if desired
        with torch.no_grad():
            init_fitnesses = self.obj_func(self.genes)
        
        self.fitnesses = init_fitnesses.clone()
        self.n_evals = self.genes.shape[0]
        
        best_idx = torch.argmin(self.fitnesses)
        self.g_best_position = self.genes[best_idx].detach().clone()
        self.g_best_fitness  = self.fitnesses[best_idx].detach().clone()

        # If no external optimizer is provided, create a default
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                [self.genes, self.selection_temp, self.mutation_scale, self.crossover_logits],
                lr=lr
            )
        else:
            self.optimizer = optimizer


    def forward(self):
        """
        """
        N, D = self.pop_size, self.dim
        pop_old = self.genes
        old_fit = self.fitnesses

        # find old best
        old_best_val, old_best_idx = torch.min(old_fit, dim=0)

        # for soft selection
        scores = -old_fit
        temp_clamp = torch.clamp(self.selection_temp, min=1e-3, max=5.0)
        logp = scores / temp_clamp
        
        def soft_sel():
            # Gumbel reparam => alpha => continuous mixture
            g = -torch.log(-torch.log(torch.rand_like(logp) + 1e-8) + 1e-8)
            alpha = torch.softmax(logp + g, dim=0)  # shape [N]
            # parent = sum_i alpha_i * pop_old[i]
            return (alpha.unsqueeze(1)*pop_old).sum(dim=0)  # shape [dim]
        
        def dimension_wise_crossover(parent1, parent2):
            """
            """
            alpha = torch.sigmoid(self.crossover_logits)  # shape [dim]
            return alpha*parent1 + (1-alpha)*parent2

        # build candidate population
        candidate_list = []
        for _ in range(N):
            # sample parent1
            g1 = -torch.log(-torch.log(torch.rand_like(logp)+1e-8)+1e-8)
            alpha1 = torch.softmax(logp + g1, dim=0) # shape [N]
            parent1 = (alpha1.unsqueeze(1)*pop_old).sum(dim=0) # shape [dim]

            # sample parent2
            parent2 = soft_sel()

            # blend crossover
            # beta = torch.rand(1)
            # child = beta*parent1 + (1-beta)*parent2
            child = dimension_wise_crossover(parent1, parent2)

            # continuous mutation
            noise = torch.randn(D)
            scale = torch.clamp(self.mutation_scale, 1e-5, 10.0)
            child = child + scale*noise

            candidate_list.append(child.unsqueeze(0))

        cand_genes = torch.cat(candidate_list, dim=0) # shape [N, dim]

        # boundary bounce
        mask_lower = cand_genes < self.lower_bound
        mask_upper = cand_genes > self.upper_bound
        bounced_lower= 2*self.lower_bound - cand_genes
        bounced_upper= 2*self.upper_bound - cand_genes
        cand_genes = torch.where(mask_lower, bounced_lower, cand_genes)
        cand_genes = torch.where(mask_upper, bounced_upper, cand_genes)

        # single objective call => new fitnesses
        cand_fit = self.obj_func(cand_genes)
        self.n_evals += cand_genes.shape[0]

        # find new best & worst
        cand_best_val, _ = torch.min(cand_fit, dim=0)
        _, cand_worst_idx= torch.max(cand_fit, dim=0)

        # elitism: if old_best_val < cand_best_val => replace worst with old best
        replaced_genes = cand_genes.clone()
        replaced_fit   = cand_fit.clone()

        if old_best_val < cand_best_val:
            replaced_genes[cand_worst_idx] = pop_old[old_best_idx]
            replaced_fit[cand_worst_idx]   = old_best_val

        # final best
        final_best_val, final_best_idx = torch.min(replaced_fit, dim=0)

        # store candidate buffers
        self._cand_genes        = replaced_genes
        self._cand_fitnesses    = replaced_fit
        self._cand_best_fit     = final_best_val
        self._cand_best_idx     = final_best_idx

        return final_best_val

    
    def update_state(self):
        """
        """
        with torch.no_grad():
            # commit
            self.genes.copy_(self._cand_genes)
            self.genes.requires_grad_(True)

            self.fitnesses.copy_(self._cand_fitnesses.detach())

            # global best
            if self._cand_best_fit < self.g_best_fitness:
                self.g_best_fitness.copy_(self._cand_best_fit.detach())
                self.g_best_position.copy_(self._cand_genes[self._cand_best_idx].detach())