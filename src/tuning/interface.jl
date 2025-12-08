# src/tuning/interface.jl
using LinearAlgebra
using StatsBase
using StableRNGs

"""
    TuningConfig

Controls the execution of the kernel tuning process (not the hyperparameters themselves).
Allows for batched KTA calculation to speed up optimization on large datasets or slow hardware.
"""
@kwdef struct TuningConfig
    rng::AbstractRNG
    batch_size::Int = 0    # 0 = Full dataset. >0 = Random batch per step.
end

"""
    TuningResult{T}

Standardized output for all tuning methods.
"""
struct TuningResult{T <: AbstractTrainedKernel}
    best_params::T              # The optimal kernel parameters (e.g., Gamma, PauliString, Thetas)
    best_score::Float64         # The best KTA score achieved
    history::Vector{Float64}    # The history of scores (or loss) during optimization
end

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


function compute_final_matrix(k::TrainedReuploadingKernel, X)
    # We know exactly how to build the circuit because k.config is inside!
    return compute_kernel_matrix_hardware(k.config, k.params, X)
end

function compute_final_matrix(k::TrainedRBFKernel, X)
    return rbf_kernel_matrix(X, k.gamma)
end
