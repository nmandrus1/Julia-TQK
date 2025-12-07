function tune_kernel(config::RBFKernelHyperparameterSearchConfig, X, y)
    best_kta = -Inf
    best_gamma = 0.1
    
    # 1. Compute Distance Matrix ONCE (Optimization)
    D2 = pairwise_distances(X) 
    
    for g in config.gamma_range
        # Fast update: K = exp(-g * D2)
        K = exp.(-g .* D2)
        score = kernel_target_alignment(K, y)
        
        if score > best_kta
            best_kta = score
            best_gamma = g
        end
    end
    return RBFHyperparameters(gamma=best_gamma)
end
