using Yao
using LinearAlgebra

"""
    compute_fidelity_hardware_compatible(config, params, x, y)

Calculates the kernel entry K(x,y) using the Compute-Uncompute method:
    |<0| U†(x) U(y) |0>|²

On a Simulator: Calculates the amplitude of |0...0> exactly.
On Hardware: This corresponds to the frequency of measuring "00...0".
"""
function compute_fidelity_hardware_compatible(config, params, x::AbstractVector, y::AbstractVector)
    # 1. Build the two circuits
    #    Note: We are using the Factory we built in Phase 1
    U_x = build_circuit(config, params, x)
    U_y = build_circuit(config, params, y)
    
    # 2. Construct the Compute-Uncompute Routine
    #    Sequence: Start |0>, Apply U(y), Apply U†(x), Measure
    #    In Yao, chain(A, B) applies A then B.
    #    We want U†(x) * U(y) |0>
    routine = chain(U_y, adjoint(U_x))
    
    # 3. Run Simulation
    final_state = apply(zero_state(config.n_qubits), routine)
    
    # 4. Extract Probability of |0...0>
    #    In Yao's state vector, index 1 is always |0...0> (if using Little Endian, which Yao does).
    #    We access the amplitude directly for efficiency/differentiability.
    amplitude_zero = state(final_state)[1]
    
    return abs2(amplitude_zero)
end

"""
    compute_kernel_matrix_hardware(config, params, X)

Computes the full kernel matrix using the hardware-compatible method.
WARNING: This is O(N^2) circuit evaluations. Slower than statevector method, but realistic.
"""
function compute_kernel_matrix_hardware(config, params, X::AbstractMatrix)
    n_samples = size(X, 2)
    K = zeros(Float64, n_samples, n_samples)
    
    # We must loop because on hardware we can't "clone" states to do matrix multiplication.
    for i in 1:n_samples
        for j in i:n_samples # Exploit symmetry
            val = compute_fidelity_hardware_compatible(config, params, view(X, :, i), view(X, :, j))
            K[i, j] = val
            K[j, i] = val
        end
    end
    
    return K
end
