def minimize(algorithm, max_evals=10000, verbose=True):
    
    generation = 1
    while algorithm.n_evals < max_evals:
        
        # (1) Forward pass => builds graph, returns candidate best fitness (scalar)
        loss = algorithm()
        
        # (2) Backprop on the scalar loss  
        algorithm.optimizer.zero_grad(set_to_none=True)
        algorithm.optimizer.step()
        algorithm.update_state()
                        
        if verbose:
            print(f"Generation {generation:4d} | Loss = {loss.item():.4f}, best_f = {algorithm.best_f.item():.4f}")
        
        generation +=1   
    
    print(f"best_f = {algorithm.best_f.item():.4f}")