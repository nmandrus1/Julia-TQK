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
        return apply(zero_state(config.n_qubits), circ)
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
    kernel_target_alignment_loss(config, params, X, y)

The final objective function for training.
Maximizes alignment = minimizes negative alignment.
"""
function kernel_target_alignment_loss(config, params, X, y)
    # 1. Compute Kernel
    K = compute_kernel_matrix_pure(config, params, X)
    
    # 2. Compute Target Matrix T = y * y'
    #    (y should be labels +1/-1)
    T = y * y'
    
    # 3. Compute Frobenius Inner Products
    #    <A, B>_F = sum(A .* B)
    inner_KT = dot(K, T)      # <K, T>
    norm_K = dot(K, K)        # <K, K>
    norm_T = dot(T, T)        # <T, T>
    
    # 4. Calculate Alignment
    alignment = inner_KT / sqrt(norm_K * norm_T)
    
    # Return negative alignment because optimizers minimize loss
    return -alignment 
end
