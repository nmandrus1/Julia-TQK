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
- `_workspace`: Statevector that is allocated once and reused
"""
mutable struct FidelityKernel
    feature_map::AbstractQuantumFeatureMap
    use_cache::Bool
    cache::Union{LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}, Nothing}
    parallel::Bool
    _workspace::ArrayReg
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
    workspace = zero_state(n_qubits(feature_map))
    return FidelityKernel(feature_map, use_cache, cache, parallel, workspace)
end

"""
    compute_kernel_value(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64})

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Arguments
- `kernel`: FidelityKernel 
- `x_statevec`: Precomputed statevector for the U(x) circuit
- `y`: Data vector that needs to be mapped to U(y)
"""
function compute_kernel_value(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64})
    # Cache lookup commented out for now
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     cache_key = (state(x_statevec), y)
    #     if haskey(kernel.cache, cache_key)
    #         return kernel.cache[cache_key]
    #     end
    # end

    # Verify workspace is in |0⟩ state
    @assert all(iszero, state(kernel._workspace)[2:end]) "Workspace was not properly zeroed out!"

    # Apply uncompute circuit: U†(y)|0⟩
    map_inputs!(kernel.feature_map, y)
    apply!(kernel._workspace, kernel.feature_map.circuit)
    
    # Compute |⟨0|U†(x)U(y)|0⟩|² = |⟨ψ_x|ψ_y⟩|²
    kernel_value = abs2(x_statevec'kernel._workspace)
    
    # Cache result (commented out for now)
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     kernel.cache[cache_key] = kernel_value
    # end

    # Reset workspace to |0⟩
    workspace_state = state(kernel._workspace)
    workspace_state .= 0
    workspace_state[1] = 1.0
    
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

    return compute_kernel_value(kernel, x_statevec, y)
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix)

Compute the kernel matrix K(X,X) for training data.
Exploits symmetry for efficiency.

# Arguments
- `X`: Data matrix where rows are samples
"""
function evaluate(kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 1)
    K = zeros(n_samples, n_samples)

    # Parallel implementation commented out for now
    # if kernel.parallel && nthreads() > 1       
    #     # Compute upper triangle in parallel
    #     @threads for idx in 1:div(n_samples * (n_samples + 1), 2)
    #         # Convert linear index to (i,j) coordinates
    #         i = ceil(Int, (-1 + sqrt(1 + 8*idx)) / 2)
    #         j = idx - div(i * (i - 1), 2)
    #         
    #         xi = X[i, :]
    #         xj = X[j, :]
    #         
    #         K[i, j] = compute_kernel_value(kernel, xi, xj)
    #         if i != j
    #             K[j, i] = K[i, j]  # Symmetry
    #         end           
    #     end
    # else
    
    # Workspace for precomputed statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))
    
    # Sequential row-by-row computation
    for i in 1:n_samples
        # Compute statevector for row i
        xi = @view X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        # Compute kernel values for this row
        for j in i:n_samples
            xj = @view X[j, :]
            K[i, j] = compute_kernel_value(kernel, x_statevec, xj)
            if i != j
                K[j, i] = K[i, j]  # Exploit symmetry
            end
        end
        
        # Reset x_statevec for next row
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
    end
    # end
    
    return K
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix)

Compute the kernel matrix K(X,Y) between two datasets.

# Arguments
- `X`: First data matrix (e.g., training data)
- `Y`: Second data matrix (e.g., test data)
"""
function evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    K = zeros(n_x, n_y)

    
    # Parallel implementation commented out for now
    # if kernel.parallel && nthreads() > 1
    #     @threads for idx in 1:(n_x * n_y)
    #         i = div(idx - 1, n_y) + 1
    #         j = mod(idx - 1, n_y) + 1
    #         
    #         xi = X[i, :]
    #         yj = Y[j, :]
    #         
    #         K[i, j] = compute_kernel_value(kernel, xi, yj)           
    #     end
    # else
    
    # Workspace for precomputed statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))

    # Sequential row-by-row computation
    for i in 1:n_x
        # Compute statevector for row i of X
        xi = X[i, :]
        map_inputs!(kernel.feature_map, xi)
        apply!(x_statevec, kernel.feature_map.circuit)
        
        # Compute kernel values against all rows of Y
        for j in 1:n_y
            yj = Y[j, :]
            K[i, j] = compute_kernel_value(kernel, x_statevec, yj)
        end
        
        # Reset x_statevec for next row
        state(x_statevec) .= 0
        state(x_statevec)[1] = 1.0
    end
    # end
    
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
