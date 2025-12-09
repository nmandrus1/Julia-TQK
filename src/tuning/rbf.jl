using Distances
using StableRNGs
using LinearAlgebra

"""
    tune_kernel(method::RBFMethod, X, y, config::TuningConfig)

Performs a Grid Search over `gamma` values using Batched KTA to ensure
fair comparison with quantum methods.
"""
function tune_kernel(method::RBFMethod, X::AbstractMatrix, y::AbstractVector, config::TuningConfig)
    # We only need the sampling RNG for RBF (no optimizer RNG needed for grid search)
    sample_rng = config.sampling_rng
    
    best_kta = -Inf
    best_gamma = method.gamma_grid[1]
    history = Float64[]
    
    # 1. Iterate through the Gamma Grid
    for g in method.gamma_grid
        
        # 2. Define Kernel Closure
        #    This matches the signature required by compute_batched_kta: f(data_subset) -> K_matrix
        function kernel_func(data_subset)
            # Compute squared Euclidean distances column-wise (dims=2)
            D2 = pairwise(SqEuclidean(), data_subset, dims=2)
            return exp.(-g .* D2)
        end
        
        # 3. Compute Batched KTA
        #    This will sample a mini-batch using sample_rng (or use full data if batch_size=0)
        score = compute_batched_kta(kernel_func, X, y, config.batch_size, sample_rng)
        
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
