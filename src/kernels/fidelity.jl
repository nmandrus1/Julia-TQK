using ArgCheck
using LRUCache
using LinearAlgebra
using LinearAlgebra.BLAS
using Yao
using YaoBlocks

"""
    FidelityKernel

A quantum kernel implementation using the uncomputation method.
Computes kernel values as K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Fields
- `feature_map`: A quantum circuit that maps input data to a quantum state
- `use_cache`: Whether to cache kernel evaluations
- `cache`: LRU cache for kernel values
- `parallel`: Whether to use parallel computation
"""
mutable struct FidelityKernel
    feature_map::AbstractQuantumFeatureMap
    use_cache::Bool
    cache::Union{LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}, Nothing}
    parallel::Bool
end

"""
    FidelityKernel(feature_map; use_cache=true, cache_size=10000, parallel=true)

Construct a quantum kernel.

# Arguments
- `feature_map`: Circuit that builds feature map
- `use_cache`: Enable caching of kernel evaluations
- `cache_size`: Maximum number of cached entries
- `parallel`: Enable parallel computation
"""
function FidelityKernel(feature_map::AbstractQuantumFeatureMap;
                       use_cache::Bool=true, 
                       cache_size::Int=10000, 
                       parallel::Bool=true)
    cache = use_cache ? LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}(maxsize=cache_size) : nothing
    return FidelityKernel(feature_map, use_cache, cache, parallel)
end

"""
    compute_kernel_value!(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64}, workspace::ArrayReg)

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².
"""
function compute_kernel_value!(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector, workspace::ArrayReg)
    # Reset workspace to |0⟩
    workspace_state = state(workspace)
    workspace_state .= 0
    workspace_state[1] = 1.0
    
    # Apply uncompute circuit: U(y)|0⟩
    map_inputs!(kernel.feature_map, y)
    apply!(workspace, kernel.feature_map.circuit)
    
    # Compute |⟨ψ_x|ψ_y⟩|² using BLAS for efficiency
    dot_product = BLAS.dotc(length(state(x_statevec)), state(x_statevec), 1, state(workspace), 1)
    return abs2(dot_product)
end

"""
    compute_kernel_value_cached(x_statevec::ArrayReg, y_statevec::ArrayReg)

Compute kernel value between two pre-computed statevectors.
"""
function compute_kernel_value_cached(x_statevec::ArrayReg, y_statevec::ArrayReg)
    dot_product = BLAS.dotc(length(state(x_statevec)), state(x_statevec), 1, state(y_statevec), 1)
    return abs2(dot_product)
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)

Compute the symmetric kernel matrix K(X,X) with memory-aware caching.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)
    # determine which caching strategy to use based on memory reqs

    n_samples = size(X, 1)
    @argcheck size(K) == (n_samples, n_samples) "K must be $n_samples × $n_samples"
    
    # Calculate memory requirements
    num_qubits = n_qubits(kernel.feature_map)
    bytes_per_statevec = (2^num_qubits) * 16  # Complex{Float64}
    required_memory_bytes = n_samples * bytes_per_statevec
    memory_budget_bytes = memory_budget_gb * (1024^3)
    
    if required_memory_bytes <= memory_budget_bytes
        # Use cached strategy
        evaluate_symmetric_cached!(K, kernel, X)
    else
        # Use original memory-efficient strategy
        evaluate_symmetric_nocache!(K, kernel, X)
    end
    
    return K
end

"""
    evaluate_symmetric_cached!(K::Matrix, kernel::FidelityKernel, X::Matrix)

Compute symmetric kernel matrix with pre-cached statevectors.
"""
function evaluate_symmetric_cached!(K::Matrix, kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 1)
    num_qubits = n_qubits(kernel.feature_map)
    
    # Pre-compute all statevectors
    statevectors = Vector{ArrayReg}(undef, n_samples)
    for i in 1:n_samples
        statevectors[i] = zero_state(num_qubits)
        map_inputs!(kernel.feature_map, @view X[i, :])
        apply!(statevectors[i], kernel.feature_map.circuit)
    end
    
    # Compute kernel matrix using cached statevectors
    @inbounds for i in 1:n_samples
        K[i, i] = 1.0  # Diagonal
        for j in (i+1):n_samples
            K[i, j] = compute_kernel_value_cached(statevectors[i], statevectors[j])
            K[j, i] = K[i, j]  # Symmetry
        end
    end
end

"""
    evaluate_symmetric_nocache!(K::Matrix, kernel::FidelityKernel, X::Matrix)

Original memory-efficient implementation without caching.
"""
function evaluate_symmetric_nocache!(K::Matrix, kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 1)
    num_qubits = n_qubits(kernel.feature_map)
    
    # Workspace registers
    x_statevec = zero_state(num_qubits)
    workspace = zero_state(num_qubits)
    
    @inbounds for i in 1:n_samples
        # Reset and compute statevector for row i
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
        
        xi = @view X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        K[i, i] = 1.0  # Diagonal
        
        for j in (i+1):n_samples
            xj = @view X[j, :]
            K[i, j] = compute_kernel_value!(kernel, x_statevec, xj, workspace)
            K[j, i] = K[i, j]  # Symmetry
        end
    end
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)

Compute the asymmetric kernel matrix K(X,Y) with memory-aware caching.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    @assert size(K) == (n_x, n_y) "K must be $n_x × $n_y"
    
    # Calculate memory requirements for Y statevectors
    num_qubits = n_qubits(kernel.feature_map)
    bytes_per_statevec = (2^num_qubits) * 16  # Complex{Float64}
    required_memory_bytes = n_y * bytes_per_statevec
    memory_budget_bytes = memory_budget_gb * (1024^3)
    
    if required_memory_bytes <= memory_budget_bytes
        # Use cached strategy
        evaluate_asymmetric_cached!(K, kernel, X, Y)
    else
        # Use original memory-efficient strategy
        evaluate_asymmetric_nocache!(K, kernel, X, Y)
    end
    
    return K
end

"""
    evaluate_asymmetric_cached!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)

Compute asymmetric kernel matrix with pre-cached Y statevectors.
"""
function evaluate_asymmetric_cached!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    num_qubits = n_qubits(kernel.feature_map)
    
    # Pre-compute all Y statevectors
    y_statevectors = Vector{ArrayReg}(undef, n_y)
    for j in 1:n_y
        y_statevectors[j] = zero_state(num_qubits)
        map_inputs!(kernel.feature_map, @view Y[j, :])
        apply!(y_statevectors[j], kernel.feature_map.circuit)
    end
    
    # Compute kernel matrix
    x_statevec = zero_state(num_qubits)
    @inbounds for i in 1:n_x
        # Reset and compute X statevector
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
        
        map_inputs!(kernel.feature_map, @view X[i, :])
        apply!(x_statevec, kernel.feature_map.circuit)
        
        # Compute against all cached Y statevectors
        for j in 1:n_y
            K[i, j] = compute_kernel_value_cached(x_statevec, y_statevectors[j])
        end
    end
end

"""
    evaluate_asymmetric_nocache!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)

Original memory-efficient implementation without caching.
"""
function evaluate_asymmetric_nocache!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    num_qubits = n_qubits(kernel.feature_map)
    
    # Workspace registers
    x_statevec = zero_state(num_qubits)
    workspace = zero_state(num_qubits)
    
    @inbounds for i in 1:n_x
        # Reset and compute statevector for row i of X
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
        
        xi = @view X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        for j in 1:n_y
            yj = @view Y[j, :]
            K[i, j] = compute_kernel_value!(kernel, x_statevec, yj, workspace)
        end
    end
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)

Compute the kernel matrix K(X,X) for training data.
Allocates and returns a new matrix.
"""
function evaluate(kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)
    n_samples = size(X, 1)
    K = zeros(n_samples, n_samples)
    evaluate!(K, kernel, X; memory_budget_gb=memory_budget_gb)
    return K
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)

Compute the kernel matrix K(X,Y) between two datasets.
Allocates and returns a new matrix.
"""
function evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    K = zeros(n_x, n_y)
    evaluate!(K, kernel, X, Y; memory_budget_gb=memory_budget_gb)
    return K
end

"""
    evaluate(kernel::FidelityKernel, x::Vector, y::Vector)

Computes the kernel between two data points.
"""
function evaluate(kernel::FidelityKernel, x::Vector, y::Vector)
    fm = kernel.feature_map
    map_inputs!(kernel.feature_map, x)
    x_statevec = apply!(zero_state(n_qubits(fm)), fm.circuit)
    workspace = zero_state(n_qubits(fm))
    
    return compute_kernel_value!(kernel, x_statevec, y, workspace)
end

"""
    clear_cache!(kernel::FidelityKernel)

Clear the kernel cache.
"""
function clear_cache!(kernel::FidelityKernel)
    if kernel.use_cache && !isnothing(kernel.cache)
        empty!(kernel.cache)
    end
end
