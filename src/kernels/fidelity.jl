
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
- `n_qubits`: Number of qubits in the circuit
- `params`: Parameters for the feature map circuit
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
    FidelityKernel(feature_map, n_qubits, params; 
                  use_cache=true, cache_size=10000, parallel=true)

Construct a quantum kernel.

# Arguments
- `feature_map`: Circuit that builds feature map
- `n_qubits`: Number of qubits
- `params`: Parameters for the feature map
- `use_cache`: Enable caching of kernel evaluations
- `cache_size`: Maximum number of cached entries
- `parallel`: Enable parallel computation
"""
function FidelityKernel(feature_map::AbstractQuantumFeatureMap;
                      use_cache::Bool=true, cache_size::Int=10000, parallel::Bool=true)
    cache = use_cache ? LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}(maxsize=cache_size) : nothing
    return FidelityKernel(feature_map, use_cache, cache, parallel, zero_state(n_qubits(feature_map)))
end


"""
    compute_kernel_value(kernel::FidelityKernel, x::Vector, y::Vector)

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Arguments
- `kernel`: FidelityKernel 
- `x_statevec`: Precomputed statevector for the U(x) circuit
- `y`: Datavector that needs to be mapped to U(y)' 
"""
function compute_kernel_value(kernel::FidelityKernel, x_statevec::ArrayReg, y::Vector)
    # Check cache first
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     cache_key = (x, y)
    #     if haskey(kernel.cache, cache_key)
    #         return kernel.cache[cache_key]
    #     end
    # end

    # invariant -- assume workspace has been reset
    @assert all(iszero, state(kernel._workspace)[2:end]) "Workspace was not properly zeroed out!"

    fm = kernel.feature_map

    # apply uncompute circuit
    map_inputs!(fm, y)
    kernel._workspace |> fm.circuit'
    
    
    # Get probability of measuring |0...0⟩
    # This is |⟨0|ψ⟩|² where |ψ⟩ = U(y)U†(x)|0⟩
    kernel_value = abs2(x_statevec'kernel._workspace)
    
    # Cache the result
    # if kernel.use_cache && !isnothing(kernel.cache)
    #     kernel.cache[(x, y)] = kernel_value
    # end

    # reset workspace
    workspace_state = state(kernel._workspace)
    workspace_state .= 0
    workspace_state[1] = 1.0 # -> |0...0⟩
    
    return kernel_value
end

"""
    evaluate(kernel::FidelityKernel, x::Vector, y::Vector)

Evaluate the quantum kernel between two data points.
"""
function evaluate(kernel::FidelityKernel, x::Vector, y::Vector)
    return compute_kernel_value(kernel, x, y)
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

    # workspace to avoid reallocating statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))
    
    # if kernel.parallel && nthreads() > 1       
    #     # Compute upper triangle in parallel
    #     @threads for idx in 1:div(n_samples * (n_samples + 1), 2)
    #         # Convert linear index to (i,j) coordinates
    #         i = ceil(Int, (-1 + sqrt(1 + 8*idx)) / 2)
    #         j = idx - div(i * (i - 1), 2)
            
    #         xi = X[i, :]
    #         xj = X[j, :]
            
    #         K[i, j] = compute_kernel_value(kernel, xi, xj)
    #         if i != j
    #             K[j, i] = K[i, j]  # Symmetry
    #         end           
    #     end
    # else
        # Sequential computation
        for i in 1:n_samples

            # compute x_statevec for this row 
            xi = X[i, :]
            map_inputs!(kernel.feature_map, xi)
            x_statevec |> kernel.feature_map.circuit # FIX: nice generic way of applying feature map to quantum state

            for j in i:n_samples
                xj = X[j, :]
                
                K[i, j] = compute_kernel_value(kernel, x_statevec, xj)
                if i != j
                    K[j, i] = K[i, j]  # Symmetry
                end
            end
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
    n_x = size(X, 2)
    n_y = size(Y, 2)
    K = zeros(n_x, n_y)

    # workspace to avoid reallocating statevector
    x_statevec = zero_state(n_qubits(kernel.feature_map))
    
    # if kernel.parallel && nthreads() > 1
    #     @threads for idx in 1:(n_x * n_y)
    #         i = div(idx - 1, n_y) + 1
    #         j = mod(idx - 1, n_y) + 1
            
    #         xi = X[i, :]
    #         yj = Y[j, :]
            
    #         K[i, j] = compute_kernel_value(kernel, xi, yj)           
    #     end
    # else
        # Sequential computation
        for i in 1:n_x

            # compute x_statevec for this row 
            xi = X[i, :]
            map_inputs!(kernel.feature_map, xi)
            x_statevec |> kernel.feature_map.circuit # FIX: nice generic way of applying feature map to quantum state

            for j in 1:n_y
                yj = Y[j, :]
                
                K[i, j] = compute_kernel_value(kernel, x_statevec, yj)
            end
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
