using Yao
using LinearAlgebra
using Zygote

"""
    compute_statevectors(config::AbstractFeatureMapConfig, params::AbstractVector, X::AbstractMatrix)

Maps a batch of data X (size N x Features) to a list of statevectors.
This is the "Forward Pass" of your quantum model.
"""
function compute_statevectors(config::AbstractFeatureMapConfig, params::AbstractVector, X::AbstractMatrix)
    n_samples = size(X, 1)
    
    # 1. Define a helper that processes a single row
    #    Zygote can differentiate through this closure easily.
    function make_state(i)
        x_row = view(X, i, :)
        circ = build_circuit(config, params, x_row)
        return apply(zero_state(n_qubits(config)), circ)
    end

    # 2. Map over all samples
    #    Zygote handles 'map' well.
    return map(make_state, 1:n_samples)
end

"""
    compute_kernel_matrix_pure(config, params, X)

Computes the full kernel matrix K(X, X) where K_ij = |<ψ(x_i)|ψ(x_j)>|²
Pure function: No buffers, no mutation.
"""
function compute_kernel_matrix_pure(config::AbstractFeatureMapConfig, params::AbstractVector, X::AbstractMatrix)
    # 1. Get all statevectors (Batch transformation)
    #    states is a Vector{ArrayReg}
    states = compute_statevectors(config, params, X)
    
    # 2. Extract raw state vectors (Vector{ComplexF64}) for linear algebra
    #    We assume the backend is standard Julia Arrays for simulation.
    raw_states = state.(states) 
    
    # 3. Stack into a matrix S (Dimension: 2^n_qubits x n_samples)
    #    This allows us to use Matrix-Matrix multiplication for the kernel.
    #    Zygote loves Matrix multiplications (BLAS).
    S = hcat(raw_states...)
    
    # 4. Compute Gram Matrix
    #    G = S' * S  (Inner products)
    #    K = |G|²    (Fidelity)
    G = S' * S
    K = abs2.(G)
    
    return K
end


"""
    kernel_target_alignment(K, y)

Pure mathematical definition of KTA. 
Agnostic to how K was generated (RBF, Quantum, etc.).
Returns the raw score (higher is better).
"""
function kernel_target_alignment(K::AbstractMatrix, y::AbstractVector)
    # y should be labels +1/-1
    T = y * y'
    
    # Frobenius Inner Products
    inner_KT = dot(K, T)
    norm_K = dot(K, K)
    norm_T = dot(T, T)
    
    # Alignment calculation
    return inner_KT / sqrt(norm_K * norm_T)
end

"""
    variational_kta_loss(config, params, X, y)

The bridge between parameters and the metric.
Computes the kernel from parameters, then calculates negative KTA.
"""
function variational_kta_loss(config, params, X, y)
    # 1. Forward Pass (Quantum Simulation)
    K = compute_kernel_matrix_pure(config, params, X)
    
    # 2. Metric Calculation (Pure Math)
    # Note: We negate it because optimizers MINIMIZE, but we want to MAXIMIZE KTA.
    return -kernel_target_alignment(K, y)
end
