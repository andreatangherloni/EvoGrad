def minimize(algorithm, max_evals=100000, verbose=True):
    
    generation = 1
    while algorithm.n_evals < max_evals:
        # (1) Forward pass => builds graph, returns candidate best fitness (scalar)
        loss = algorithm()
        
        # (2) Backprop on the scalar loss     
        algorithm.optimizer.zero_grad()
        loss.backward()
        algorithm.optimizer.step()
        algorithm.update_state()
                        
        if verbose:
            print(f"Generation {generation}, Loss = {loss.item():.4f}, g_best = {algorithm.g_best_fitness.item():.4f}")
        
        generation +=1   
    
    print(f"g_best = {algorithm.g_best_fitness.item():.4f}")