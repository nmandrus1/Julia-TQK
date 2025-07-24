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
- `parallel`: Whether to use parallel computation
"""
mutable struct FidelityKernel
    feature_map::AbstractQuantumFeatureMap
    parallel::Bool
end

"""
    FidelityKernel(feature_map; use_cache=true, cache_size=10000, parallel=true)

Construct a quantum kernel.

# Arguments
- `feature_map`: Circuit that builds feature map
- `parallel`: Enable parallel computation
"""
function FidelityKernel(feature_map::AbstractQuantumFeatureMap; parallel::Bool=true)
    return FidelityKernel(feature_map, parallel)
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
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)

Compute the symmetric kernel matrix K(X,X) with hybrid tiled evaluation.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix; memory_budget_gb::Float64=4.0)
    n_samples = size(X, 1)
    @argcheck size(K) == (n_samples, n_samples) "K must be $n_samples × $n_samples"
    
    num_qubits = n_qubits(kernel.feature_map)
    tile_size = calculate_tile_size(num_qubits, memory_budget_gb)

    # compute workspaces and allocate memory for reuse
    xi_statevectors = [zero_state(num_qubits) for _ in 1:n_samples]
    xj_statevectors = [zero_state(num_qubits) for _ in 1:n_samples]
    
    # Process tiles for the upper triangle
    @inbounds for i_start in 1:tile_size:n_samples
        i_end = min(i_start + tile_size - 1, n_samples)
        X_i_view = @view X[i_start:i_end, :]
        
        # Compute diagonal block (X_i vs X_i)
        # NOTE: This function computes all the xi statevectors already so we can just use them!
        K_diag_view = @view K[i_start:i_end, i_start:i_end]
        evaluate_symmetric_cached!(K_diag_view, kernel, X_i_view, xi_statevectors)

        
        # Compute off-diagonal blocks (X_i vs X_j for j > i)
        for j_start in (i_end + 1):tile_size:n_samples
            j_end = min(j_start + tile_size - 1, n_samples)
            X_j_view = @view X[j_start:j_end, :]
            
            K_offdiag_view = @view K[i_start:i_end, j_start:j_end]
            evaluate_asymmetric_cached!(K_offdiag_view, kernel, X_i_view, X_j_view, xi_statevectors, xj_statevectors)
            
            # Exploit symmetry
            K[j_start:j_end, i_start:i_end] .= transpose(K_offdiag_view)
        end
    end
    
    return K
end

"""
    evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)

Compute the asymmetric kernel matrix K(X,Y) with hybrid tiled evaluation.
"""
function evaluate!(K::Matrix, kernel::FidelityKernel, X::Matrix, Y::Matrix; memory_budget_gb::Float64=4.0)
    n_x = size(X, 1)
    n_y = size(Y, 1)
    @assert size(K) == (n_x, n_y) "K must be $n_x × $n_y"
    
    num_qubits = n_qubits(kernel.feature_map)
    # Split budget 50/50 between X and Y tiles
    tile_size = calculate_tile_size(num_qubits, memory_budget_gb, 0.5)

    # compute workspaces and allocate memory for reuse
    x_statevectors = [zero_state(num_qubits) for _ in 1:n_x]
    y_statevectors = [zero_state(num_qubits) for _ in 1:n_y]
    
    # Efficient loop order: cache X tile and reuse against all Y tiles
    @inbounds for i_start in 1:tile_size:n_x
        i_end = min(i_start + tile_size - 1, n_x)
        X_view = @view X[i_start:i_end, :]

        for i in 1:n_x
            # sets statevectors to zero and applys feature map
            create_statevec!(x_statevectors[i], kernel.feature_map, @view X_view[i, :])
        end
        
        for j_start in 1:tile_size:n_y
            j_end = min(j_start + tile_size - 1, n_y)
            Y_view = @view Y[j_start:j_end, :]
            
            K_view = @view K[i_start:i_end, j_start:j_end]
            evaluate_asymmetric_cached!(K_view, kernel, X_view, Y_view, x_statevectors, y_statevectors)
        end
    end
    
    return K
end

# --- Core Cached Implementations ---

"""
    evaluate_symmetric_cached!(K_view, kernel, X_view)

Computes one symmetric tile with all statevectors cached in memory.
Only computes upper triangle and uses symmetry.
"""
function evaluate_symmetric_cached!(K_view::AbstractMatrix, kernel::FidelityKernel, X_view::AbstractMatrix, statevectors::AbstractVector{T}) where {T<:AbstractArrayReg}
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
    evaluate_asymmetric_cached!(K_view, kernel, X_view, Y_view)

Computes one asymmetric tile with all statevectors cached in memory.
"""
function evaluate_asymmetric_cached!(K_view::AbstractMatrix, kernel::FidelityKernel, X_view::AbstractMatrix, Y_view::AbstractMatrix,
                                   x_statevectors::AbstractVector{T1}, workspace::AbstractVector{T2}) where {T1<:AbstractArrayReg, T2<:AbstractArrayReg}   

    n_x = size(X_view, 1)
    n_y = size(Y_view, 1)

    for j in 1:n_y
        # sets statevectors to zero and applys feature map
        create_statevec!(workspace[j], kernel.feature_map, @view Y_view[j, :])
    end
    
    # Compute kernel values
    @inbounds for i in 1:n_x
        for j in 1:n_y
            K_view[i, j] = compute_kernel_value_cached(x_statevectors[i], workspace[j])
        end
    end
end


# --- Gradient Implementations ---

"""
    loss_gradient(kernel::FidelityKernel, loss_fn::Function, X::AbstractMatrix; memory_budget_gb::Float64=4.0)

Compute the gradient of a loss function with respect to kernel parameters.
Automatically selects between in-memory and tiled computation based on available memory.

# Arguments
- `kernel`: The FidelityKernel instance
- `loss_fn`: Loss function that takes a kernel matrix and returns a scalar
- `X`: Data matrix (n_samples × n_features)
- `memory_budget_gb`: Memory budget in GB

# Returns
- `d_theta`: Gradient vector of length n_parameters
"""
function loss_gradient(
    kernel::FidelityKernel, 
    loss_fn::Function, 
    X::AbstractMatrix; 
    memory_budget_gb::Float64=4.0
)
    n_samples = size(X, 1)
    fm = kernel.feature_map
    num_qubits = n_qubits(fm)
    
    # Check if we can fit all statevectors in memory
    bytes_per_statevec = (2^num_qubits) * sizeof(ComplexF64)
    total_bytes_needed = n_samples * bytes_per_statevec
    memory_budget_bytes = memory_budget_gb * (1024^3)
    
    # Forward pass - always use existing tiled evaluation
    K = evaluate(kernel, X; memory_budget_gb=memory_budget_gb)
    loss, grad = Zygote.withgradient(loss_fn, K)
    
    # Choose computation path based on memory constraints
    if total_bytes_needed <= memory_budget_bytes
        # In-memory path: all statevectors fit
        # have to take the index because this is a tuple for some reason
        return loss, loss_gradient_inmemory!(kernel, grad[1], X)
    else
        # Tiled path: need to work in chunks
        # have to take the index because this is a tuple for some reason
        return loss, loss_gradient_tiled!(kernel, grad[1], X; memory_budget_gb=memory_budget_gb)
    end
end

"""
    loss_gradient_inmemory!(kernel, dL_dK, X)

Compute gradients when all statevectors fit in memory.
"""
function loss_gradient_inmemory!(
    kernel::FidelityKernel,
    dL_dK::AbstractMatrix,
    X::AbstractMatrix
)
    n_samples = size(X, 1)
    fm = kernel.feature_map
    num_params = n_params(fm)
    num_qubits = n_qubits(fm)
    
    # init angle gradient collection
    grad_collector = zeros(ComplexF64, num_params)
    # init real componenet collection
    grad_angles = zeros(Float64, num_params)

    # parameter_gradients -- interleaved vector 
    grad_params = zeros(Float64, num_params * 2)
    
    # Pre-compute all statevectors
    statevecs = [create_statevec!(zero_state(num_qubits), fm, @view X[i, :]) for i in 1:n_samples]
    
    # Pre-allocate adjoint accumulator for reuse
    adjoint_accumulator = zero_state(ComplexF64, num_qubits)
    
    # Process each data point
    for i in 1:n_samples
        # Build complete adjoint for data point i
        zero!(adjoint_accumulator)
        accumulate_adjoints!(
            adjoint_accumulator,
            statevecs[i],
            statevecs,
            dL_dK,
            i,
            1:n_samples
        )
        
        # Backpropagate the completed adjoint
        compute_angle_gradients!(
            statevecs[i],
            adjoint_accumulator,
            @view(X[i, :]),
            fm,
            grad_collector,
            grad_angles
        )

        gradient_chain_rule!(grad_params, fm, grad_angles, @view X[i, :])
    end

    # unpack gradients into their own vectors
    d_weights = [grad_params[i] for i in findall(isodd, eachindex(grad_params))]
    d_biases = [grad_params[i] for i in findall(iseven, eachindex(grad_params))]
    
    return d_weights, d_biases
end

"""
    loss_gradient_tiled!(kernel, dL_dK, X; memory_budget_gb)

Compute gradients using tiled computation for memory-constrained scenarios.
"""
function loss_gradient_tiled!(
    kernel::FidelityKernel,
    dL_dK::AbstractMatrix,
    X::AbstractMatrix;
    memory_budget_gb::Float64=4.0
)
    n_samples = size(X, 1)
    fm = kernel.feature_map
    num_params = n_params(fm)
    num_qubits = n_qubits(fm)
    
    # init angle gradient collection
    grad_collector = zeros(ComplexF64, num_params)
    # init real componenet collection
    grad_angles = zeros(Float64, num_params)

    # parameter_gradients -- interleaved vector 
    grad_params = zeros(Float64, num_params * 2)
    
    # Calculate tile size - split budget between row states, col states, and adjoints
    tile_size = calculate_tile_size(num_qubits, memory_budget_gb, 1/3)
    
    # Process in row tiles
    for i_start in 1:tile_size:n_samples
        i_end = min(i_start + tile_size - 1, n_samples)
        row_tile_size = i_end - i_start + 1
        
        # Compute row statevectors
        row_statevecs = [create_statevec!(zero_state(num_qubits), fm, @view X[i, :]) 
                        for i in i_start:i_end]
        
        # Initialize adjoint accumulators for this tile
        adjoint_accumulators = [zero_state(ComplexF64, num_qubits) for _ in 1:row_tile_size]
        
        # Accumulate contributions from all column tiles
        for j_start in 1:tile_size:n_samples
            j_end = min(j_start + tile_size - 1, n_samples)
            
            # Compute or reuse column statevectors
            if i_start == j_start
                col_statevecs = row_statevecs
            else
                col_statevecs = [create_statevec!(zero_state(num_qubits), fm, @view X[j, :]) 
                               for j in j_start:j_end]
            end
            
            # Accumulate adjoints for this tile combination
            accumulate_adjoints_tiled!(
                adjoint_accumulators,
                row_statevecs,
                col_statevecs,
                dL_dK,
                i_start:i_end,
                j_start:j_end
            )
        end
        
        # Backpropagate completed adjoints for this row tile
        for (i_local, i_global) in enumerate(i_start:i_end)
            compute_angle_gradients!(
                row_statevecs[i_local],
                adjoint_accumulators[i_local],
                @view(X[i_global, :]),
                fm,
                grad_collector,
                grad_angles
            )

            gradient_chain_rule!(grad_params, fm, grad_angles, @view X[i_global, :])
        end

    end
    # unpack gradients into their own vectors
    d_weights = [grad_params[i] for i in findall(isodd, eachindex(grad_params))]
    d_biases = [grad_params[i] for i in findall(iseven, eachindex(grad_params))]
    
    return d_weights, d_biases
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
            # Factor of 2 accounts for symmetric kernel contributions
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
    compute_angle_gradients(psi, adjoint, x, feature_map)

Backpropagate a single adjoint state to get gradients with respect to gate angles.

# Returns
- A `Vector{Float64}` of gradients ∂L/∂θ for the given data point `x`.
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
