using ArgCheck
using LRUCache
using LinearAlgebra
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
    compute_kernel_value(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64})

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Arguments
- `kernel`: FidelityKernel 
- `x_statevec`: Precomputed statevector for the U(x) circuit
- `y`: Data vector that needs to be mapped to U(y)
- `workspace`: Preallocated statevector for computing U(y)
"""
function compute_kernel_value!(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64}, workspace::ArrayReg)
    # Cache lookup commented out for now
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     cache_key = (state(x_statevec), y)
    #     if haskey(kernel.cache, cache_key)
    #         return kernel.cache[cache_key]
    #     end
    # end

    # Verify workspace is in |0⟩ state
    @argcheck all(iszero, state(workspace)[2:end]) "Workspace was not properly zeroed out!"

    # Apply uncompute circuit: U†(y)|0⟩
    map_inputs!(kernel.feature_map, y)
    apply!(workspace, kernel.feature_map.circuit)
    
    # Compute |⟨0|U†(x)U(y)|0⟩|² = |⟨ψ_x|ψ_y⟩|²
    kernel_value = abs2(x_statevec'workspace)
    
    # Cache result (commented out for now)
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     kernel.cache[cache_key] = kernel_value
    # end

    
    return kernel_value
end

"""
    evaluate(kernel::FidelityKernel, x::Vector, y::Vector) 

Computes the kernel between these two data points.
> NOTE: If you are doing several kernel evaluations it will be more efficient
        to use the row-caching logic in the matrix evaluation functions

"""
function evaluate(kernel::FidelityKernel, x::Vector, y::Vector)
    fm = kernel.feature_map
    map_inputs!(kernel.feature_map, x)
    x_statevec = apply!(zero_state(n_qubits(fm)), fm.circuit)
    workspace = zero_state(n_qubits(fm))

    return compute_kernel_value!(kernel, x_statevec, y, workspace)
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix)

Compute the kernel matrix K(X,X) for training data in-place.
Exploits symmetry for efficiency.

# Arguments
- `K`: Pre-allocated kernel matrix to fill (must be n_samples × n_samples)
- `kernel`: FidelityKernel object
- `X`: Data matrix where rows are samples
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 1)
    @argcheck size(K) == (n_samples, n_samples) "K must be $n_samples × $n_samples"
    
    # Workspace for precomputed statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))
    workspace = zero_state(n_qubits(kernel.feature_map))
    
    # Sequential row-by-row computation
    for i in 1:n_samples
        # Compute statevector for row i
        xi = @view X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        # Compute kernel values for this row
        for j in i:n_samples
            xj = @view X[j, :]
            K[i, j] = compute_kernel_value!(kernel, x_statevec, xj, workspace)
            if i != j
                K[j, i] = K[i, j]  # Exploit symmetry
            end

            # Reset workspace to |0⟩
            workspace_state = state(workspace)
            workspace_state .= 0
            workspace_state[1] = 1.0
        end
        
        # Reset x_statevec for next row
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
    end
    
    return K
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix)

Compute the kernel matrix K(X,X) for training data.
Allocates and returns a new matrix. For repeated calls, use `evaluate!` instead.

# Arguments
- `kernel`: FidelityKernel object
- `X`: Data matrix where rows are samples

# Returns
- Kernel matrix K where K[i,j] = kernel(X[i,:], X[j,:])
"""
function evaluate(kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 1)
    K = zeros(n_samples, n_samples)
    evaluate!(K, kernel, X)
    return K
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)

Compute the kernel matrix K(X,Y) between two datasets in-place.

# Arguments
- `K`: Pre-allocated kernel matrix to fill (must be size(X,1) × size(Y,1))
- `kernel`: FidelityKernel object
- `X`: First data matrix (e.g., training data)
- `Y`: Second data matrix (e.g., test data)
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    @assert size(K) == (n_x, n_y) "K must be $n_x × $n_y"
    
    # Workspace for precomputed statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))
    workspace = zero_state(n_qubits(kernel.feature_map))

    # Sequential row-by-row computation
    for i in 1:n_x
        # Compute statevector for row i of X
        xi = @view X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        # Compute kernel values against all rows of Y
        for j in 1:n_y
            yj = @view Y[j, :]
            K[i, j] = compute_kernel_value!(kernel, x_statevec, yj, workspace)

            # Reset workspace to |0⟩
            workspace_state = state(workspace)
            workspace_state .= 0
            workspace_state[1] = 1.0
        end
        
        # Reset x_statevec for next row
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
    end
    
    return K
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix)

Compute the kernel matrix K(X,Y) between two datasets.
Allocates and returns a new matrix. For repeated calls, use `evaluate!` instead.

# Arguments
- `kernel`: FidelityKernel object
- `X`: First data matrix (e.g., training data)
- `Y`: Second data matrix (e.g., test data)

# Returns
- Kernel matrix K where K[i,j] = kernel(X[i,:], Y[j,:])
"""
function evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    K = zeros(n_x, n_y)
    evaluate!(K, kernel, X, Y)
    return K
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
