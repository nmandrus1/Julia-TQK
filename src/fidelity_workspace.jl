using Yao 

"""
    FidelityWorkspace

Memory-efficient workspace for quantum kernel gradient computation.
Pre-allocates statevector memory that can be reused between forward and backward passes.

# Fields
- `statevec_pool::Vector{ArrayReg}`: Pre-allocated pool of quantum registers
- `forward_capacity::Int`: Maximum samples for forward pass (uses entire pool)
- `backward_capacity::Int`: Maximum samples for backward pass (splits pool for adjoints)
- `grad_collector::Vector{ComplexF64}`: Buffer for collecting angle gradients
- `grad_angles::Vector{Float64}`: Buffer for real components of gradients
- `grad_params::Vector{Float64}`: Buffer for parameter gradients (interleaved weights/biases)
"""
struct FidelityWorkspace
    statevec_pool::Vector{ArrayReg}
    forward_capacity::Int
    backward_capacity::Int
    grad_collector::Vector{ComplexF64}
    grad_angles::Vector{Float64}
    grad_params::Vector{Float64}
end

"""
    WorkspaceViews

Abstraction for accessing different views of the workspace memory pool.
Provides clean interface for forward/backward pass memory access patterns.

# Fields
- `statevecs::SubArray`: View of statevectors
- `adjoints::Union{SubArray,Nothing}`: View of adjoint vectors (backward pass only)
"""
struct WorkspaceViews
    statevecs::SubArray{ArrayReg,1}
    adjoints::Union{SubArray{ArrayReg,1},Nothing}
end

"""
    create_workspace(feature_map::AbstractQuantumFeatureMap, max_samples::Int; 
                    memory_budget_gb::Float64=4.0) -> FidelityWorkspace

Create a workspace with memory pool sized according to the budget.

# Arguments
- `feature_map`: Feature map to determine qubit count and parameter count
- `max_samples`: Maximum number of samples that will be processed
- `memory_budget_gb`: Memory budget in gigabytes

# Returns
- `FidelityWorkspace` with pre-allocated memory pools

# Errors
- Throws error if memory budget is insufficient for even a single statevector
"""
function create_workspace(
    feature_map::ReuploadingCircuit,
    max_samples::Int;
    memory_budget_gb::Float64=4.0
)
    num_qubits = n_qubits(feature_map)
    num_params = n_params(feature_map)
    
    # Calculate memory requirements
    bytes_per_statevec = (2^num_qubits) * sizeof(ComplexF64)
    memory_budget_bytes = memory_budget_gb * (1024^3)
           
    if bytes_per_statevec > memory_budget_bytes
        error("Memory budget of $(memory_budget_gb) GB is insufficient for even a single " *
              "$(num_qubits)-qubit statevector (requires $(bytes_per_statevec / 1024^3) GB)")
    end
    
    # Compute capacities
    forward_capacity = floor(Int, memory_budget_bytes / bytes_per_statevec)
    backward_capacity = floor(Int, memory_budget_bytes / (2 * bytes_per_statevec))
    
    # Don't allocate more than needed
    forward_capacity = min(forward_capacity, max_samples)
    backward_capacity = min(backward_capacity, max_samples)
    
    # Allocate the pool (sized for forward capacity, which is larger)
    statevec_pool = [zero_state(ComplexF64, num_qubits) for _ in 1:forward_capacity]
    
    # Allocate gradient buffers
    grad_collector = zeros(ComplexF64, num_params)
    grad_angles = zeros(Float64, num_params)
    grad_params = zeros(Float64, num_params * 2)  # interleaved weights and biases
    
    return FidelityWorkspace(
        statevec_pool,
        forward_capacity,
        backward_capacity,
        grad_collector,
        grad_angles,
        grad_params
    )
end

"""
    get_forward_views(workspace::FidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for forward pass computation.

# Arguments
- `workspace`: The workspace containing the memory pool
- `n_samples`: Number of samples to process (must not exceed forward_capacity)

# Returns
- `WorkspaceViews` with statevecs pointing to the full pool, adjoints = nothing
"""
function get_forward_views(workspace::FidelityWorkspace, n_samples::Int)
    @assert n_samples <= workspace.forward_capacity "Requested $n_samples samples exceeds forward capacity $(workspace.forward_capacity)"
    
    return WorkspaceViews(
        view(workspace.statevec_pool, 1:n_samples),
        nothing
    )
end

"""
    get_backward_views(workspace::FidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for backward pass computation.
Splits the pool in half: first half for statevectors, second half for adjoints.

# Arguments
- `workspace`: The workspace containing the memory pool
- `n_samples`: Number of samples to process (must not exceed backward_capacity)

# Returns
- `WorkspaceViews` with statevecs and adjoints each using half the pool
"""
function get_backward_views(workspace::FidelityWorkspace, n_samples::Int)
    @assert n_samples <= workspace.backward_capacity "Requested $n_samples samples exceeds backward capacity $(workspace.backward_capacity)"
    
    return WorkspaceViews(
        view(workspace.statevec_pool, 1:n_samples),
        view(workspace.statevec_pool, n_samples+1:2*n_samples)
    )
end

"""
    get_statevectors(workspace::Union{FidelityWorkspace,Nothing}, n_qubits::Int, 
                    count::Int, offset::Int=0)

Helper function that returns either workspace views or newly allocated statevectors.

# Arguments
- `workspace`: Optional workspace (if nothing, allocates new memory)
- `n_qubits`: Number of qubits for statevector size
- `count`: Number of statevectors needed
- `offset`: Starting offset in workspace pool (ignored if workspace is nothing)

# Returns
- Collection of ArrayReg objects (either views or newly allocated)
"""
function get_statevectors(
    workspace::Union{FidelityWorkspace,Nothing},
    n_qubits::Int,
    count::Int,
    offset::Int=0
)
    if workspace !== nothing
        return @view workspace.statevec_pool[offset+1:offset+count]
    else
        return [zero_state(ComplexF64, n_qubits) for _ in 1:count]
    end
end

"""
    get_or_allocate_statevecs(views::Union{WorkspaceViews,Nothing}, n_qubits::Int, count::Int)

Get statevectors from workspace views or allocate new ones.

# Arguments
- `views`: Optional workspace views
- `n_qubits`: Number of qubits
- `count`: Number of statevectors needed

# Returns
- Collection of statevectors
"""
function get_or_allocate_statevecs(
    views::Union{WorkspaceViews,Nothing},
    n_qubits::Int,
    count::Int
)
    if views !== nothing
        @assert count <= length(views.statevecs) "Requested $count statevecs exceeds available $(length(views.statevecs))"
        return @view views.statevecs[1:count]
    else
        return [zero_state(ComplexF64, n_qubits) for _ in 1:count]
    end
end

"""
    get_or_allocate_adjoints(views::Union{WorkspaceViews,Nothing}, n_qubits::Int, count::Int)

Get adjoint vectors from workspace views or allocate new ones.

# Arguments
- `views`: Optional workspace views (must have adjoints field if not nothing)
- `n_qubits`: Number of qubits
- `count`: Number of adjoints needed

# Returns
- Collection of adjoint vectors
"""
function get_or_allocate_adjoints(
    views::Union{WorkspaceViews,Nothing},
    n_qubits::Int,
    count::Int
)
    if views !== nothing && views.adjoints !== nothing
        @assert count <= length(views.adjoints) "Requested $count adjoints exceeds available $(length(views.adjoints))"
        return @view views.adjoints[1:count]
    else
        return [zero_state(ComplexF64, n_qubits) for _ in 1:count]
    end
end

"""
    compute_tile_sizes(workspace::FidelityWorkspace, n_samples::Int) -> (forward_tile_size, backward_tile_size)

Compute optimal tile sizes for forward and backward passes given workspace capacity.

# Arguments
- `workspace`: The workspace with capacity information
- `n_samples`: Total number of samples to process

# Returns
- Tuple of (forward_tile_size, backward_tile_size)
"""
function compute_tile_sizes(workspace::FidelityWorkspace, n_samples::Int)
    # Forward pass can use larger tiles
    forward_tile_size = min(workspace.forward_capacity, n_samples)
    
    # Backward pass uses smaller tiles due to adjoint storage
    backward_tile_size = min(workspace.backward_capacity, n_samples)
    
    return (forward_tile_size, backward_tile_size)
end

"""
    check_workspace_capacity(workspace::Union{FidelityWorkspace,Nothing}, n_samples::Int, 
                           pass_type::Symbol=:forward) -> Bool

Check if workspace has sufficient capacity for the requested operation.

# Arguments
- `workspace`: Optional workspace
- `n_samples`: Number of samples
- `pass_type`: Either :forward or :backward

# Returns
- true if workspace can handle n_samples without tiling, false otherwise
"""
function check_workspace_capacity(
    workspace::Union{FidelityWorkspace,Nothing},
    n_samples::Int,
    pass_type::Symbol=:forward
)
    workspace === nothing && return false
    
    if pass_type == :forward
        return n_samples <= workspace.forward_capacity
    elseif pass_type == :backward
        return n_samples <= workspace.backward_capacity
    else
        error("Unknown pass type: $pass_type. Must be :forward or :backward")
    end
end

"""
    reset_gradient_buffers!(workspace::FidelityWorkspace)

Reset all gradient accumulation buffers to zero.
"""
function reset_gradient_buffers!(workspace::FidelityWorkspace)
    fill!(workspace.grad_collector, 0.0)
    fill!(workspace.grad_angles, 0.0)
    fill!(workspace.grad_params, 0.0)
    return nothing
end

"""
    extract_gradients(workspace::FidelityWorkspace) -> (weights, biases)

Extract weight and bias gradients from the interleaved gradient buffer.

# Returns
- Tuple of (weight_gradients, bias_gradients) as separate vectors
"""
function extract_gradients(workspace::FidelityWorkspace)
    # Gradients are interleaved as [w1, b1, w2, b2, ...]
    d_weights = workspace.grad_params[1:2:end]
    d_biases = workspace.grad_params[2:2:end]
    
    return copy(d_weights), copy(d_biases)
end
