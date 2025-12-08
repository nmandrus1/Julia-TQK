using StableRNGs

function tune_kernel(method::RBFMethod, X::AbstractMatrix, y::AbstractVector, config::TuningConfig)
    rng = StableRNG(seed=config.seed)
    
    best_kta = -Inf
    best_gamma = method.gamma_grid[1]
    history = Float64[]
    
    # Optimization: Precompute distance matrix if possible? 
    # Actually, for batched KTA, precomputing full D is wasteful if batch is small.
    # We will just iterate.
    
    for g in method.gamma_grid
        # Closure for the current kernel
        kernel_func(data) = rbf_kernel_matrix(data, g) 
        
        score = compute_batched_kta(kernel_func, X, y, config.batch_size, rng)
        push!(history, score)
        
        if score > best_kta
            best_kta = score
            best_gamma = g
        end
    end
    
    return TuningResult(
        TrainedRBFKernel(best_gamma),
        best_kta,
        history
    )
end

# Helper for matrix generation
function rbf_kernel_matrix(X, gamma)
    D2 = pairwise(SqEuclidean(), X, dims=2)
    return exp.(-gamma .* D2)
end
