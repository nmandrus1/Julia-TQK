using Random
using Parameters
using StableRNGs

abstract type AbstractDataParams end
abstract type AbstractKernelMethod end


"""
Abstract parent for all trained kernels. 
Any subtype T must support a dispatch:
    compute_kernel_matrix(kernel::T, X)
"""
abstract type AbstractTrainedKernel end


"""
    TuningConfig

Controls the execution of the kernel tuning process (not the hyperparameters themselves).
Allows for batched KTA calculation to speed up optimization on large datasets or slow hardware.
"""

"""
    TuningConfig

- `sampling_rng`: Used ONLY for selecting data batches.
- `optimizer_rng`: Used for algorithm stochasticity (e.g., SPSA perturbations).
- `batch_size`:  0 = Full dataset. >0 = Random batch per step.
"""
@kwdef struct TuningConfig
    sampling_rng::AbstractRNG
    optimizer_rng::AbstractRNG
    batch_size::Int = 0 
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

