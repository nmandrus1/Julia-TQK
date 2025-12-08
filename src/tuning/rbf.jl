using Distances

function tune_kernel(config::RBFMethod, X, y)
    @warn "RBF TUNING IS NOT BATCHED"

    best_kta = -Inf
    best_gamma = 0.1
    
    # [CORRECTION] Compute distances column-wise (dims=2)
    # This matrix is (N_samples x N_samples)
    D2 = pairwise(SqEuclidean(), X, dims=2) 
    
    for g in config.gamma_grid # Note: was gamma_range in your snippet, check struct def
        K = exp.(-g .* D2)
        score = kernel_target_alignment(K, y)
        
        if score > best_kta
            best_kta = score
            best_gamma = g
        end
    end
    return TrainedRBFKernel(best_gamma)
end
