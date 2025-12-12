# src/tuning/interface.jl
using LinearAlgebra
using StatsBase
using StableRNGs

"""
    tune_kernel(method::AbstractKernelMethod, X, y, config::TuningConfig) -> TuningResult

Abstract interface. Must be implemented by each method.
"""
function tune_kernel(method::AbstractKernelMethod, X::AbstractMatrix, y::AbstractVector, config::TuningConfig)
    error("tune_kernel not implemented for method $(typeof(method))")
end

"""
    compute_batched_kta(kernel_fn, X, y, batch_size, rng)

Helper to compute KTA on a subset of data.
"""
function compute_batched_kta(kernel_fn::Function, X::AbstractMatrix, y::AbstractVector, batch_size::Int, rng::AbstractRNG)
    if batch_size <= 0 || batch_size >= length(y)
        # Full KTA
        K = kernel_fn(X)
        return kernel_target_alignment(K, y)
    else
        # Batched KTA
        indices = sample(rng, 1:length(y), batch_size, replace=false)
        X_batch = X[:, indices] # Assuming Column-Major
        y_batch = y[indices]
        
        K_batch = kernel_fn(X_batch)
        return kernel_target_alignment(K_batch, y_batch)
    end
end
