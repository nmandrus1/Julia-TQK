
# src/utils/mock_hardware.jl
using Yao
using StatsBase

"""
    mock_quantum_execution(circuit::AbstractBlock, n_shots::Int)

Simulates a real quantum computer execution.
1. Calculates the exact state (Cheating, but necessary for simulation).
2. Samples 'n_shots' bitstrings from that distribution.
3. Returns the empirical probability of measuring the all-zero state |0...0>.

This introduces statistical noise (Shot Noise).
"""
function mock_quantum_execution(circuit::AbstractBlock, n_shots::Int)
    # 1. Exact Simulation
    reg = apply(zero_state(nqubits(circuit)), circuit)
    
    # 2. Measure / Sample
    #    Yao's measure returns bitstrings (integers). 0 is |0...0>.
    results = measure(reg, nshots=n_shots)
    
    # 3. Count occurrences of the target state (Zero)
    count_zeros = count(x -> x == 0, results)
    
    # 4. Return empirical probability
    return count_zeros / n_shots
end

"""
    hardware_compatible_loss(config, params, X, y, n_shots)

A loss function that uses the noisy Mock Hardware instead of exact linear algebra.
This is what the SPSA optimizer will "see".
"""
function hardware_compatible_loss(config, params, X, y, n_shots)
    # Compute the "Projected Kernel" element-wise using shots
    # Note: KTA is expensive (N^2), so for SPSA validation we usually use 
    # a simpler loss or small batches, but we'll stick to KTA for consistency.
    
    n_samples = size(X, 1)
    K_empirical = zeros(n_samples, n_samples)
    
    for i in 1:n_samples
        for j in i:n_samples
            # 1. Build Circuit U = U†(x) U(y)
            U_x = build_circuit(config, params, view(X, i, :))
            U_y = build_circuit(config, params, view(X, j, :))
            circuit = chain(U_y, adjoint(U_x))
            
            # 2. Run on "Hardware"
            prob = mock_quantum_execution(circuit, n_shots)
            
            K_empirical[i,j] = prob
            K_empirical[j,i] = prob
        end
    end
    
    # Calculate Alignment on the noisy kernel
    T = y * y'
    inner = dot(K_empirical, T)
    norm_K = dot(K_empirical, K_empirical)
    norm_T = dot(T, T)
    
    alignment = inner / sqrt(norm_K * norm_T)
    return -alignment
end
