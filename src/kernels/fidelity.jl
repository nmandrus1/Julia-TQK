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
    loss_gradient(kernel::FidelityKernel, loss_fn::Function, X::AbstractMatrix, 
                 workspace::AbstractFidelityWorkspace=DynamicWorkspace(n_qubits(kernel.feature_map), n_params(kernel.feature_map)))

Compute the gradient of a loss function with respect to kernel parameters.

# Arguments
- `kernel`: The FidelityKernel instance
- `loss_fn`: Loss function that takes a kernel matrix and returns a scalar
- `X`: Data matrix (n_samples × n_features)
- `workspace`: Workspace for memory management

# Returns
- `loss`: The loss value
- `(d_weights, d_biases)`: Gradient vectors
"""
function loss_gradient(
    kernel::FidelityKernel, 
    K::AbstractMatrix,
    loss_fn::Function, 
    X::AbstractMatrix,
    workspace::AbstractFidelityWorkspace=DynamicWorkspace(n_qubits(kernel.feature_map), n_params(kernel.feature_map));
    # loss_kwargs::Dict{Symbol, Any},
)
    n_samples = size(X, 1)
    
    # loss, grad = Zygote.withgradient(loss_fn, K, loss_kwargs...)
    loss, grad = Zygote.withgradient(loss_fn, K)


    @debug "Loss" loss
    @debug "Gradient norm" norm(grad[1])
    @debug "Symmetric?" isapprox(grad[1], grad[1]', atol=1e-10)
    @debug "Diagonal" diag(grad[1])[1:5]
    @debug "Off-diagonal" (grad[1][1,2], grad[1][2,1])    

    # Backward pass - choose path based on workspace capacity
    if n_samples <= get_backward_tile_size(workspace)
        # All samples fit in memory with 2-way split
        return loss, loss_gradient_with_workspace!(kernel, grad[1], X, workspace)
    else
        # Need tiling with 3-way split
        return loss, loss_gradient_tiled_with_workspace!(kernel, grad[1], X, workspace)
    end
end

"""
    loss_gradient_with_workspace!(kernel, dL_dK, X, workspace)

Compute gradients using workspace memory without tiling (2-way memory split).
"""
function loss_gradient_with_workspace!(
    kernel::FidelityKernel,
    dL_dK::AbstractMatrix,
    X::AbstractMatrix,
    workspace::AbstractFidelityWorkspace
)
    n_samples = size(X, 1)
    fm = kernel.feature_map
    
    # Get gradient buffers and reset
    reset!(workspace)
    grad_collector, grad_angles, _ = get_grad_buffers!(workspace)
    
    # Get views for backward pass (2-way split: statevecs + adjoints)
    views = get_backward_views(workspace, n_samples)
    
    # Pre-compute all statevectors
    for i in 1:n_samples
        create_statevec!(views.statevecs[i], fm, @view X[i, :])
    end
    
    # Initialize adjoints to zero
    for adjoint in views.adjoints
        zero!(adjoint)
    end
    
    # Process each data point
    for i in 1:n_samples
        # Accumulate adjoints
        accumulate_adjoints!(
            views.adjoints[i],
            views.statevecs[i],
            views.statevecs,
            dL_dK,
            i,
            1:n_samples
        )
        
        # Backpropagate
        compute_angle_gradients!(
            views.statevecs[i],
            views.adjoints[i],
            @view(X[i, :]),
            fm,
            grad_collector,
            grad_angles
        )
        
        _, _, grad_params = get_grad_buffers!(workspace)
        gradient_chain_rule!(grad_params, fm, grad_angles, @view X[i, :])
    end
    
    return extract_gradients(workspace)
end

"""
    loss_gradient_tiled_with_workspace!(kernel, dL_dK, X, workspace)

Compute gradients using workspace memory with tiling (3-way memory split).
"""
function loss_gradient_tiled_with_workspace!(
    kernel::FidelityKernel,
    dL_dK::AbstractMatrix,
    X::AbstractMatrix,
    workspace::AbstractFidelityWorkspace
)
    n_samples = size(X, 1)
    fm = kernel.feature_map
    
    # Get gradient buffers and reset
    reset!(workspace)
    grad_collector, grad_angles, _ = get_grad_buffers!(workspace)
    
    # Get tile size for 3-way split
    tile_size = min(get_backward_tiled_tile_size(workspace), n_samples)
    
    # Process in row tiles
    for i_start in 1:tile_size:n_samples
        i_end = min(i_start + tile_size - 1, n_samples)
        row_tile_size = i_end - i_start + 1
        
        # Get views for this tile (3-way split for tiling)
        # Note: col_size may be different from row_tile_size
        max_col_size = tile_size
        views = get_backward_tiled_views(workspace, row_tile_size, max_col_size)
        
        # Compute row statevectors
        for (idx, i) in enumerate(i_start:i_end)
            create_statevec!(views.statevecs[idx], fm, @view X[i, :])
            zero!(views.adjoints[idx])
        end
        
        # Accumulate contributions from all column tiles
        for j_start in 1:tile_size:n_samples
            j_end = min(j_start + tile_size - 1, n_samples)
            col_size = j_end - j_start + 1
            
            if i_start == j_start
                # Reuse row statevectors for diagonal block
                col_statevecs = views.statevecs
            else
                # Use the dedicated column statevector region
                col_statevecs = @view views.col_statevecs[1:col_size]
                
                for (idx, j) in enumerate(j_start:j_end)
                    create_statevec!(col_statevecs[idx], fm, @view X[j, :])
                end
            end
            
            # Accumulate adjoints for this tile combination
            accumulate_adjoints_tiled!(
                views.adjoints,
                views.statevecs,
                col_statevecs,
                dL_dK,
                i_start:i_end,
                j_start:j_end
            )
        end
        
        # Backpropagate completed adjoints for this row tile
        for (i_local, i_global) in enumerate(i_start:i_end)
            compute_angle_gradients!(
                views.statevecs[i_local],
                views.adjoints[i_local],
                @view(X[i_global, :]),
                fm,
                grad_collector,
                grad_angles
            )
            
            _, _, grad_params = get_grad_buffers!(workspace)
            gradient_chain_rule!(grad_params, fm, grad_angles, @view X[i_global, :])
        end
    end
    
    return extract_gradients(workspace)
end

# --- Workhorse Functions (No Memory Allocation) ---

"""
    accumulate_adjoints!(adjoint, psi_i, all_statevecs, dL_dK, i, j_indices)

Accumulate adjoint contributions for a single data point.
Modifies adjoint in-place. Allocates no memory.
"""
function accumulate_adjoints!(
    adjoint::ArrayReg,
    psi_i::ArrayReg,
    all_statevecs::AbstractVector{<:ArrayReg},
    dL_dK::AbstractMatrix,
    i::Int,
    j_indices::AbstractVector{Int}
)
    @inbounds for (j_local, j) in enumerate(j_indices)
        # Factor of 2 accounts for symmetric kernel contributions
        # gradient_factor = (i == j ? 1 : 2) * dL_dK[i, j]
         
        # skipping diagonal contribution to gradient since diagonal is constant
         
        if i == j
            continue
        end
        gradient_factor = 2 * dL_dK[i, j]
        
        if isapprox(gradient_factor, 0.0; atol=1e-9)
            continue
        end
        
        psi_j = all_statevecs[j_local]
        c_ij = BLAS.dotc(length(state(psi_i)), state(psi_i), 1, state(psi_j), 1)
        
        # adjoint += (gradient_factor * conj(c_ij)) * psi_j
        axpy!(gradient_factor * conj(c_ij), state(psi_j), state(adjoint))
    end
end

"""
    accumulate_adjoints_tiled!(adjoints, row_statevecs, col_statevecs, dL_dK, row_indices, col_indices)

Accumulate adjoint contributions for a tile of data points.
Modifies adjoints in-place. Allocates no memory.
"""
function accumulate_adjoints_tiled!(
    adjoints::AbstractVector{<:ArrayReg},
    row_statevecs::AbstractVector{<:ArrayReg},
    col_statevecs::AbstractVector{<:ArrayReg},
    dL_dK::AbstractMatrix,
    row_indices::AbstractVector{Int},
    col_indices::AbstractVector{Int}
)
    @inbounds for (i_local, i_global) in enumerate(row_indices)
        psi_i = row_statevecs[i_local]
        adjoint_i = adjoints[i_local]
        
        for (j_local, j_global) in enumerate(col_indices)
            
            # Skip diagonal - K[i,i] = 1 always, gradient is zero
            if i_global == j_global
                continue
            end

            # Factor of 2 accounts for symmetric kernel contributions
            # gradient_factor = (i_global == j_global ? 1 : 2) * dL_dK[i_global, j_global]
            gradient_factor = 2 * dL_dK[i_global, j_global]
            
            if isapprox(gradient_factor, 0.0; atol=1e-9)
                continue
            end
            
            psi_j = col_statevecs[j_local]
            c_ij = dot(psi_i, psi_j)
            
            # adjoint_i += (gradient_factor * conj(c_ij)) * psi_j
            axpy!(gradient_factor * conj(c_ij), psi_j, adjoint_i)
        end
    end
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
    apply_back!((copy(psi), adjoint), feature_map.circuit, grad_collector)
    
    map!(real, real_components, grad_collector)
end
