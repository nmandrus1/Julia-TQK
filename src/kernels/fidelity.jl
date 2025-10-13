using ArgCheck
using LRUCache
using LinearAlgebra
using LinearAlgebra.BLAS
using Yao
using YaoBlocks
using Logging
using Zygote

"""
    FidelityKernel

A quantum kernel implementation using the uncomputation method.
Computes kernel values as K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Fields
- `feature_map`: A quantum circuit that maps input data to a quantum state
"""
mutable struct FidelityKernel
    feature_map::AbstractQuantumFeatureMap
end

# --- Core Helper Functions ---

"""
    In-place modification of a Yao statevector to the |0⟩ state
"""
@inline function zero!(statevec::ArrayReg)
    s = state(statevec)
    fill!(s, 0)
    s[1] = 1.0
end

"""
    create_statevec!(register::ArrayReg, feature_map, x::AbstractVector)

Computes the statevector |ψ(x)⟩ = U(x)|0⟩ in-place in the provided register.
"""
@inline function create_statevec!(register::ArrayReg, feature_map, x::AbstractVector)
    zero!(register)
    map_inputs!(feature_map, x)
    apply!(register, feature_map.circuit)
    return register
end

"""
    compute_kernel_value!(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector{Float64}, workspace::ArrayReg)

Compute the quantum kernel value K(x,y) = |⟨0|U†(x)U(y)|0⟩|².
"""
function compute_kernel_value!(kernel::FidelityKernel, x_statevec::ArrayReg, y::AbstractVector, workspace::ArrayReg)
    zero!(workspace)
    map_inputs!(kernel.feature_map, y)
    apply!(workspace, kernel.feature_map.circuit)
    
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
    calculate_tile_size(num_qubits::Int, memory_budget_gb::Float64, split_factor::Float64=1.0)

Calculate how many statevectors can fit in the memory budget.

# Arguments
- `num_qubits`: Number of qubits
- `memory_budget_gb`: Memory budget in GB
- `split_factor`: Fraction of budget to use (e.g., 0.5 for splitting between X and Y)

# Returns
- tile_size: Number of statevectors that fit
"""
function calculate_tile_size(num_qubits::Int, memory_budget_gb::Float64, split_factor::Float64=1.0)
    bytes_per_statevec = (2^num_qubits) * sizeof(ComplexF64)
    memory_budget_bytes = memory_budget_gb * (1024^3) * split_factor
    
    if bytes_per_statevec > memory_budget_bytes
        error("Memory budget of $(memory_budget_gb) GB is insufficient for even a single $(num_qubits)-qubit statevector (requires $(bytes_per_statevec / 1024^3) GB)")
    end
    
    tile_size = floor(Int, memory_budget_bytes / bytes_per_statevec)
    
    if tile_size < 100 && split_factor == 1.0
        @warn "Small tile size: only $tile_size statevectors fit in $(memory_budget_gb) GB. Consider increasing memory budget for better performance."
    end
    
    return tile_size
end

# --- Main Evaluation Functions ---

"""
    evaluate(kernel::FidelityKernel, X::Matrix; workspace::AbstractFidelityWorkspace)

Compute the kernel matrix K(X,X) for training data.
If no workspace is provided, creates a DynamicWorkspace automatically.
"""
function evaluate(kernel::FidelityKernel, X::AbstractMatrix; workspace::AbstractFidelityWorkspace=DynamicWorkspace(n_qubits(kernel.feature_map), n_params(kernel.feature_map)))
    n_samples = size(X, 1)
    K = zeros(n_samples, n_samples)
    evaluate!(K, kernel, X, workspace)
    return K
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix; workspace::AbstractFidelityWorkspace)

Compute the kernel matrix K(X,Y) between two datasets.
If no workspace is provided, creates a DynamicWorkspace automatically.
"""
function evaluate(kernel::FidelityKernel, X::AbstractMatrix, Y::AbstractMatrix; workspace::AbstractFidelityWorkspace=DynamicWorkspace(n_qubits(kernel.feature_map), n_params(kernel.feature_map)))
    n_x = size(X, 1)
    n_y = size(Y, 1)
    K = zeros(n_x, n_y)
    evaluate!(K, kernel, X, Y, workspace)
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
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, workspace::AbstractFidelityWorkspace)

Compute the symmetric kernel matrix K(X,X) with hybrid tiled evaluation.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::AbstractMatrix, workspace::AbstractFidelityWorkspace)
    n_samples = size(X, 1)
    @argcheck size(K) == (n_samples, n_samples) "K must be $n_samples × $n_samples"
    
    num_qubits = n_qubits(kernel.feature_map)
    tile_size = min(get_forward_tile_size(workspace), n_samples)
    
    # Process tiles for the upper triangle
    @inbounds for i_start in 1:tile_size:n_samples
        i_end = min(i_start + tile_size - 1, n_samples)
        i_size = i_end - i_start + 1
        X_i_view = @view X[i_start:i_end, :]

        # new operation so we reset the workspace 
        reset!(workspace)
        
        # Get views for row tile
        views_i = get_vectors!(workspace, i_size)
        
        # Compute diagonal block (X_i vs X_i)
        K_diag_view = @view K[i_start:i_end, i_start:i_end]
        evaluate_symmetric_cached!(K_diag_view, kernel, X_i_view, views_i)

        # mark current offset
        j_block_start = workspace.offset[]
        
        # Compute off-diagonal blocks (X_i vs X_j for j > i)
        for j_start in (i_end + 1):tile_size:n_samples
            j_end = min(j_start + tile_size - 1, n_samples)
            j_size = j_end - j_start + 1
            X_j_view = @view X[j_start:j_end, :]
            
            # set current workspace offset to reuse views_j AND AVOID OVERWRITING views_i
            workspace.offset[] = j_block_start

            # Get views for column tile
            views_j = get_vectors!(workspace, j_size)
            
            K_offdiag_view = @view K[i_start:i_end, j_start:j_end]
            evaluate_asymmetric_cached!(K_offdiag_view, kernel, X_i_view, X_j_view, views_i.statevecs, views_j)
            
            # Exploit symmetry
            K[j_start:j_end, i_start:i_end] .= transpose(K_offdiag_view)
        end
    end
    
    return K
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix, workspace::AbstractFidelityWorkspace)

Compute the asymmetric kernel matrix K(X,Y) with hybrid tiled evaluation.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::AbstractMatrix, Y::AbstractMatrix, workspace::AbstractFidelityWorkspace)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    @assert size(K) == (n_x, n_y) "K must be $n_x × $n_y"
    
    num_qubits = n_qubits(kernel.feature_map)
    
    # For asymmetric case, we can split the forward capacity
    max_tile = get_forward_tile_size(workspace)
    tile_size = min(div(max_tile, 2), max(n_x, n_y))
    
    # Process in tiles
    @inbounds for i_start in 1:tile_size:n_x
        i_end = min(i_start + tile_size - 1, n_x)
        i_size = i_end - i_start + 1
        X_view = @view X[i_start:i_end, :]
        
        # Get X statevectors
        reset!(workspace)
        x_statevectors = get_vectors!(workspace, i_size)
        
        # Compute X statevectors for this tile
        for (idx, i) in enumerate(1:i_size)
            create_statevec!(x_statevectors[idx], kernel.feature_map, @view X_view[i, :])
        end

        # checkpoint workspace 
        j_block_start = workspace.offset[]
        
        for j_start in 1:tile_size:n_y
            j_end = min(j_start + tile_size - 1, n_y)
            j_size = j_end - j_start + 1
            Y_view = @view Y[j_start:j_end, :]

            # avoid overwriting views_i by resetting workspace offset
            workspace.offset[] = j_block_start             

            # Get Y statevectors
            y_statevectors = get_vectors!(workspace, j_size)
            
            K_view = @view K[i_start:i_end, j_start:j_end]
            evaluate_asymmetric_cached!(K_view, kernel, X_view, Y_view, x_statevectors, y_statevectors)
        end
    end
    
    return K
end

# --- Core Cached Implementations ---

"""
    evaluate_symmetric_cached!(K_view, kernel, X_view, statevectors)

Computes one symmetric tile with all statevectors cached in memory.
Only computes upper triangle and uses symmetry.
"""
function evaluate_symmetric_cached!(K_view::AbstractMatrix, kernel::FidelityKernel, 
                                  X_view::AbstractMatrix, statevectors::AbstractVector{T}) where {T<:AbstractArrayReg}
    n_samples = size(X_view, 1)
    
    # Pre-compute all statevectors
    for i in 1:n_samples
        create_statevec!(statevectors[i], kernel.feature_map, @view X_view[i, :])
    end
    
    # Compute kernel values
    @inbounds for i in 1:n_samples
        K_view[i, i] = 1.0  # Diagonal
        for j in (i+1):n_samples
            val = compute_kernel_value_cached(statevectors[i], statevectors[j])
            K_view[i, j] = val
            K_view[j, i] = val  # Symmetry
        end
    end
end

"""
    evaluate_asymmetric_cached!(K_view, kernel, X_view, Y_view, x_statevectors, y_statevectors)

Computes one asymmetric tile with all statevectors cached in memory.
"""
function evaluate_asymmetric_cached!(K_view::AbstractMatrix, kernel::FidelityKernel, 
                                   X_view::AbstractMatrix, Y_view::AbstractMatrix,
                                   x_statevectors::AbstractVector{T1}, 
                                   y_statevectors::AbstractVector{T2}) where {T1<:AbstractArrayReg, T2<:AbstractArrayReg}   

    n_x = size(X_view, 1)
    n_y = size(Y_view, 1)

    # Compute Y statevectors
    for j in 1:n_y
        create_statevec!(y_statevectors[j], kernel.feature_map, @view Y_view[j, :])
    end
    
    # Compute kernel values
    @inbounds for i in 1:n_x
        for j in 1:n_y
            K_view[i, j] = compute_kernel_value_cached(x_statevectors[i], y_statevectors[j])
        end
    end
end

# --- Gradient Implementations ---

"""
    kernel_gradient_parameter_shift(kernel, X, workspace) -> (grad_w, grad_b)

Compute ∂K/∂wₖ and ∂K/∂bₖ using parameter-shift rule.

# Returns
- `grad_w::Vector{Float64}`: Weight gradients (n_params,)
- `grad_b::Vector{Float64}`: Bias gradients (n_params,)
"""
function kernel_gradient_parameter_shift(
    kernel::FidelityKernel,
    X::AbstractMatrix,
    workspace::AbstractFidelityWorkspace
)
    n_samples = size(X, 1)
    n_p = n_params(kernel.feature_map)
    fm = kernel.feature_map
    
    # Accumulator for final gradients
    grad_w = zeros(n_p)
    grad_b = zeros(n_p)
    
    # Store and copy original parameters
    original_weights, original_biases = get_params(fm)
    weights = copy(original_weights)
    biases = copy(original_biases)
    
    # Cache base statevectors
    reset!(workspace)
    base_statevecs = get_vectors!(workspace, n_samples)
    for i in 1:n_samples
        create_statevec!(base_statevecs[i], fm, @view X[i, :])
    end
    
    shifted_statevec = zero_state(n_qubits(fm))
    
    @inbounds for k in 1:n_p
        feat_idx = fm.gate_features[k]
        
        # Weight gradients
        for i in 1:n_samples
            xᵢₖ = X[i, feat_idx]
            if abs(xᵢₖ) < 1e-9 continue end
            
            shift_w = π / (2 * xᵢₖ)
            
            # Positive shift
            weights[k] = original_weights[k] + shift_w
            assign_params!(fm, weights, biases)
            create_statevec!(shifted_statevec, fm, @view X[i, :])
            K_plus_row = [compute_kernel_value_cached(shifted_statevec, base_statevecs[j]) 
                          for j in 1:n_samples]
            
            # Negative shift
            weights[k] = original_weights[k] - shift_w
            assign_params!(fm, weights, biases)
            create_statevec!(shifted_statevec, fm, @view X[i, :])
            K_minus_row = [compute_kernel_value_cached(shifted_statevec, base_statevecs[j]) 
                           for j in 1:n_samples]
            
            # Accumulate: ∂K/∂w = (∂K/∂θ) × x, summed over all pairs
            dK_dtheta = (K_plus_row .- K_minus_row) ./ 2.0
            grad_w[k] += 2 * sum(dK_dtheta) * xᵢₖ  # Factor of 2 for symmetry
            
            weights[k] = original_weights[k]
        end
        
        # Bias gradients
        shift_b = π / 2
        for i in 1:n_samples
            # Positive shift
            biases[k] = original_biases[k] + shift_b
            assign_params!(fm, weights, biases)
            create_statevec!(shifted_statevec, fm, @view X[i, :])
            K_plus_row = [compute_kernel_value_cached(shifted_statevec, base_statevecs[j]) 
                          for j in 1:n_samples]
            
            # Negative shift
            biases[k] = original_biases[k] - shift_b
            assign_params!(fm, weights, biases)
            create_statevec!(shifted_statevec, fm, @view X[i, :])
            K_minus_row = [compute_kernel_value_cached(shifted_statevec, base_statevecs[j]) 
                           for j in 1:n_samples]
            
            dK_dtheta = (K_plus_row .- K_minus_row) ./ 2.0
            grad_b[k] += 2 * sum(dK_dtheta)
            
            biases[k] = original_biases[k]
        end
    end
    
    assign_params!(fm, original_weights, original_biases)
    return grad_w, grad_b
end

# Add to src/kernels/fidelity.jl

"""
    loss_gradient_finite_diff(kernel, K_cache, loss_fn, X, workspace; ε=1e-5)

Compute gradients using finite differences on parameters.
Simple and robust, requires 2P kernel evaluations.
"""
function loss_gradient_finite_diff(
    kernel::FidelityKernel,
    K_cache::AbstractMatrix,
    loss_fn::Function,
    X::AbstractMatrix,
    workspace::AbstractFidelityWorkspace;
    ε::Float64=1e-5
)
    fm = kernel.feature_map
    n_p = n_params(fm)
    grad_w = zeros(n_p)
    grad_b = zeros(n_p)
    
    weights, biases = get_params(fm)
    
    # Gradients w.r.t. weights
    for i in 1:n_p
        w_plus = copy(weights)
        w_plus[i] += ε
        assign_params!(fm, w_plus, biases)
        K_plus = evaluate!(K_cache, kernel, X, workspace)
        loss_plus = loss_fn(K_plus)
        
        w_minus = copy(weights)
        w_minus[i] -= ε
        assign_params!(fm, w_minus, biases)
        K_minus = evaluate!(K_cache, kernel, X, workspace)
        loss_minus = loss_fn(K_minus)
        
        grad_w[i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Gradients w.r.t. biases
    for i in 1:n_p
        b_plus = copy(biases)
        b_plus[i] += ε
        assign_params!(fm, weights, b_plus)
        K_plus = evaluate!(K_cache, kernel, X, workspace)
        loss_plus = loss_fn(K_plus)
        
        b_minus = copy(biases)
        b_minus[i] -= ε
        assign_params!(fm, weights, b_minus)
        K_minus = evaluate!(K_cache, kernel, X, workspace)
        loss_minus = loss_fn(K_minus)
        
        grad_b[i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Restore and compute final loss
    assign_params!(fm, weights, biases)
    evaluate!(K_cache, kernel, X, workspace)
    loss = loss_fn(K_cache)
    
    return loss, (grad_w, grad_b)
end
"""
    compute_angle_gradients!(psi, adjoint, x, feature_map, grad_collector, real_components)

Backpropagate a single adjoint state to get gradients with respect to gate angles.
"""
function compute_angle_gradients!(
    psi::ArrayReg,
    adjoint::ArrayReg,
    x::AbstractVector,
    feature_map::AbstractQuantumFeatureMap,
    grad_collector::AbstractVector{ComplexF64},
    real_components::AbstractVector{Float64}
)
    # Set feature map inputs for the specific data point x
    map_inputs!(feature_map, x)
    
    # Collect angle gradients via backpropagation
    # `apply_back!` calculates ∂L/∂θ and puts it in grad_collector

    result = apply_back!((copy(psi), adjoint), feature_map.circuit, grad_collector)
    @debug result

    # map!(real, real_components, grad_collector)
    map!(x-> 2* real(x), real_components, grad_collector)
end
