
using LRUCache
using LinearAlgebra
using ProgressMeter
using Yao
using YaoBlocks

"""
    FidelityKernel

A quantum kernel implementation using the uncomputation method.
Computes kernel values as K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Fields
- `feature_map_builder`: Function that builds the feature map circuit (x, params) -> Circuit
- `n_qubits`: Number of qubits in the circuit
- `params`: Parameters for the feature map circuit
- `use_cache`: Whether to cache kernel evaluations
- `cache`: LRU cache for kernel values
- `parallel`: Whether to use parallel computation
"""
mutable struct FidelityKernel
    feature_map_builder::Function
    n_qubits::Int
    params::Any
    use_cache::Bool
    cache::Union{LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}, Nothing}
    parallel::Bool
end

"""
    FidelityKernel(feature_map_builder, n_qubits, params; 
                  use_cache=true, cache_size=10000, parallel=true)

Construct a quantum kernel.

# Arguments
- `feature_map_builder`: Function (x, params) -> Circuit that builds feature map
- `n_qubits`: Number of qubits
- `params`: Parameters for the feature map
- `use_cache`: Enable caching of kernel evaluations
- `cache_size`: Maximum number of cached entries
- `parallel`: Enable parallel computation
"""
function FidelityKernel(feature_map_builder::Function, n_qubits::Int, params;
                      use_cache::Bool=true, cache_size::Int=10000, parallel::Bool=true)
    cache = use_cache ? LRU{Tuple{Vector{Float64}, Vector{Float64}}, Float64}(maxsize=cache_size) : nothing
    return FidelityKernel(feature_map_builder, n_qubits, params, use_cache, cache, parallel)
end

"""
    construct_kernel_circuit(kernel::FidelityKernel, x::Vector, y::Vector)

Construct the kernel circuit U(y)U†(x) for computing K(x,y).
"""
function construct_kernel_circuit(kernel::FidelityKernel, x::Vector, y::Vector)
    # Build U(y)
    U_y = kernel.feature_map_builder(y, kernel.params)
    
    # Build U†(x) by taking adjoint
    U_x = kernel.feature_map_builder(x, kernel.params)
    U_x_dag = Daggered(U_x)
    
    # Combine: U(y)U†(x)
    return chain(U_y, U_x_dag)
end

"""
    compute_kernel_value(kernel::FidelityKernel, x::Vector, y::Vector)

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².
"""
function compute_kernel_value(kernel::FidelityKernel, x::Vector, y::Vector)
    # Check cache first
    if kernel.use_cache && !isnothing(kernel.cache)
        cache_key = (x, y)
        if haskey(kernel.cache, cache_key)
            return kernel.cache[cache_key]
        end
    end
    
    # Construct circuit
    circuit = construct_kernel_circuit(kernel, x, y)
    
    # Prepare initial state |0...0⟩
    reg = zero_state(kernel.n_qubits)
    
    # Apply circuit
    reg |> circuit
    
    # Get probability of measuring |0...0⟩
    # This is |⟨0|ψ⟩|² where |ψ⟩ = U(y)U†(x)|0⟩
    kernel_value = abs2(reg[1])  # First amplitude is for |0...0⟩
    
    # Cache the result
    if kernel.use_cache && !isnothing(kernel.cache)
        kernel.cache[(x, y)] = kernel_value
    end
    
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
- `X`: Data matrix where columns are samples
"""
function evaluate(kernel::FidelityKernel, X::Matrix)
    n_samples = size(X, 2)
    K = zeros(n_samples, n_samples)
    
    if kernel.parallel && nthreads() > 1
        # Parallel computation with progress bar
        progress = Progress(div(n_samples * (n_samples + 1), 2), desc="Computing kernel matrix...")
        progress_lock = ReentrantLock()
        
        # Compute upper triangle in parallel
        @threads for idx in 1:div(n_samples * (n_samples + 1), 2)
            # Convert linear index to (i,j) coordinates
            i = ceil(Int, (-1 + sqrt(1 + 8*idx)) / 2)
            j = idx - div(i * (i - 1), 2)
            
            xi = X[:, i]
            xj = X[:, j]
            
            K[i, j] = compute_kernel_value(kernel, xi, xj)
            if i != j
                K[j, i] = K[i, j]  # Symmetry
            end
            
            lock(progress_lock) do
                next!(progress)
            end
        end
    else
        # Sequential computation
        @showprogress desc="Computing kernel matrix..." for i in 1:n_samples
            for j in i:n_samples
                xi = X[:, i]
                xj = X[:, j]
                
                K[i, j] = compute_kernel_value(kernel, xi, xj)
                if i != j
                    K[j, i] = K[i, j]  # Symmetry
                end
            end
        end
    end
    
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
    
    if kernel.parallel && nthreads() > 1
        # Parallel computation
        progress = Progress(n_x * n_y, desc="Computing kernel matrix...")
        progress_lock = ReentrantLock()
        
        @threads for idx in 1:(n_x * n_y)
            i = div(idx - 1, n_y) + 1
            j = mod(idx - 1, n_y) + 1
            
            xi = X[:, i]
            yj = Y[:, j]
            
            K[i, j] = compute_kernel_value(kernel, xi, yj)
            
            lock(progress_lock) do
                next!(progress)
            end
        end
    else
        # Sequential computation
        @showprogress desc="Computing kernel matrix..." for i in 1:n_x
            for j in 1:n_y
                xi = X[:, i]
                yj = Y[:, j]
                
                K[i, j] = compute_kernel_value(kernel, xi, yj)
            end
        end
    end
    
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
