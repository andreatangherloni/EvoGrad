import torch

def minimize(algorithm,
             max_evals=10000,
             optimizer=None,
             grad_clip: float = None,
             scheduler: str | None = "plateau",
             step_size: int = 100,
             plateau_factor: float = 0.5,
             plateau_min_lr: float = 1e-6,
             verbose=True):
    
    if optimizer is not None:
        algorithm.optimizer = optimizer
    
    # Scheduler choice
    if scheduler is None:
        lr_sched = None
    elif scheduler == "plateau":
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(algorithm.optimizer,
                                                              mode="min",
                                                              factor=plateau_factor,
                                                              patience=step_size,
                                                              min_lr=plateau_min_lr)
    elif scheduler == "step":
        # halve the LR every 100 generations (example)
        lr_sched = torch.optim.lr_scheduler.StepLR(algorithm.optimizer, step_size=step_size, gamma=0.5)
    else:
        # user supplied a ready-made scheduler instance
        lr_sched: torch.optim.lr_scheduler._LRScheduler = scheduler
    
    generation = 1
    while algorithm.n_evals < max_evals:
        
        algorithm.optimizer.zero_grad(set_to_none=True)
        
        # (1) Forward pass => builds graph, returns candidate best fitness (scalar)
        loss = algorithm()
                
        # 2) back-prop through the whole generation
        loss.backward()
        
        # gradient clipping (if desired)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(algorithm.parameters(), grad_clip)
        
        # 3) exploitation / hyper-learning step
        algorithm.optimizer.step()
        
        # 4) commit the evolutionary changes
        algorithm.update_state()
        
        # LR scheduler update
        if lr_sched is not None:
            if isinstance(lr_sched,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_sched.step(loss.item())
            else:
                lr_sched.step()
                        
        if verbose:
            print(f"Generation {generation:4d} | Loss = {loss.item():.4f}, best_f = {algorithm.best_f.item():.4f}")
        
        generation +=1   
    
    if verbose:
        print(f"best_f = {algorithm.best_f.item():.4f}")