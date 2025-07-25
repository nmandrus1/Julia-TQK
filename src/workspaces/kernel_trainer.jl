"""
Abstract interface for quantum kernel workspaces.
All workspace implementations must provide memory management for statevectors and gradients.
"""
abstract type AbstractFidelityWorkspace end

# Required interface functions
function get_vectors! end
function get_grad_buffers! end
function reset! end
function get_forward_tile_size end
function get_backward_tile_size end

"""
    PreallocatedWorkspace <: AbstractFidelityWorkspace

Memory-efficient workspace that pre-allocates all memory based on a budget.
Provides zero-allocation computation after initialization.

# Fields
- `statevec_pool::Vector{ArrayReg}`: Pre-allocated pool of quantum registers
- `forward_capacity::Int`: Maximum samples for forward pass (uses entire pool)
- `backward_capacity::Int`: Maximum samples for backward pass (splits pool for adjoints)
- `backward_tiled_capacity::Int`: Maximum samples for tiled backward pass (3-way split)
- `grad_collector::Vector{ComplexF64}`: Buffer for collecting angle gradients
- `grad_angles::Vector{Float64}`: Buffer for real components of gradients
- `grad_params::Vector{Float64}`: Buffer for parameter gradients (interleaved weights/biases)
- `offset::Ref{Int}`: Current position in the memory pool
"""
struct PreallocatedWorkspace <: AbstractFidelityWorkspace
    statevec_pool::Vector{ArrayReg}
    forward_capacity::Int
    backward_capacity::Int
    backward_tiled_capacity::Int  # For 3-way split in tiled gradients
    grad_collector::Vector{ComplexF64}
    grad_angles::Vector{Float64}
    grad_params::Vector{Float64}
    offset::Ref{Int}
end

"""
    DynamicWorkspace <: AbstractFidelityWorkspace

Workspace that lazily allocates memory on demand and reuses it.
Starts with empty pools and grows as needed.

# Fields
- `n_qubits::Int`: Number of qubits for statevector size
- `statevec_pool::Vector{ArrayReg}`: Dynamically growing pool
- `grad_collector::Vector{ComplexF64}`: Buffer for angle gradients
- `grad_angles::Vector{Float64}`: Buffer for real gradients
- `grad_params::Vector{Float64}`: Buffer for parameter gradients
- `offset::Ref{Int}`: Current position in the memory pool
"""
mutable struct DynamicWorkspace <: AbstractFidelityWorkspace
    n_qubits::Int
    statevec_pool::Vector{ArrayReg}
    grad_collector::Vector{ComplexF64}
    grad_angles::Vector{Float64}
    grad_params::Vector{Float64}
    offset::Ref{Int}
end

"""
    create_preallocated_workspace(feature_map::AbstractQuantumFeatureMap, max_samples::Int; 
                                 memory_budget_gb::Float64=4.0) -> PreallocatedWorkspace

Create a workspace with memory pool sized according to the budget.

# Arguments
- `feature_map`: Feature map to determine qubit count and parameter count
- `max_samples`: Maximum number of samples that will be processed
- `memory_budget_gb`: Memory budget in gigabytes

# Returns
- `PreallocatedWorkspace` with pre-allocated memory pools

# Errors
- Throws error if memory budget is insufficient for even a single statevector
"""
function create_preallocated_workspace(
    feature_map::AbstractQuantumFeatureMap,
    max_samples::Int;
    memory_budget_gb::Float64=4.0
)
    num_qubits = n_qubits(feature_map)
    num_params = n_params(feature_map)
    
    # Calculate memory requirements
    bytes_per_statevec = (2^num_qubits) * sizeof(ComplexF64)
    memory_budget_bytes = memory_budget_gb * (1024^3)
    
    # Reserve small amount for gradient buffers
    gradient_buffer_size = (num_params * sizeof(ComplexF64) +     # grad_collector
                           num_params * sizeof(Float64) +         # grad_angles  
                           2 * num_params * sizeof(Float64))      # grad_params
    
    available_for_statevecs = memory_budget_bytes - gradient_buffer_size
    
    if bytes_per_statevec > available_for_statevecs
        error("Memory budget of $(memory_budget_gb) GB is insufficient for even a single " *
              "$(num_qubits)-qubit statevector (requires $(bytes_per_statevec / 1024^3) GB)")
    end
    
    # Compute capacities
    forward_capacity = floor(Int, available_for_statevecs / bytes_per_statevec)
    backward_capacity = floor(Int, available_for_statevecs / (2 * bytes_per_statevec))
    backward_tiled_capacity = floor(Int, available_for_statevecs / (3 * bytes_per_statevec))
    
    # Don't allocate more than needed
    forward_capacity = min(forward_capacity, max_samples)
    backward_capacity = min(backward_capacity, max_samples)
    backward_tiled_capacity = min(backward_tiled_capacity, max_samples)
    
    # Allocate the pool (sized for forward capacity, which is largest)
    statevec_pool = [zero_state(ComplexF64, num_qubits) for _ in 1:forward_capacity]
    
    # Allocate gradient buffers
    grad_collector = zeros(ComplexF64, num_params)
    grad_angles = zeros(Float64, num_params)
    grad_params = zeros(Float64, num_params * 2)  # interleaved weights and biases
    
    return PreallocatedWorkspace(
        statevec_pool,
        forward_capacity,
        backward_capacity,
        backward_tiled_capacity,
        grad_collector,
        grad_angles,
        grad_params,
        Ref(0)
    )
end

"""
    DynamicWorkspace(n_qubits::Int, n_params::Int) -> DynamicWorkspace

Create a dynamic workspace that allocates memory on demand.

# Arguments
- `n_qubits`: Number of qubits for statevector size
- `n_params`: Number of parameters in the feature map
"""
function DynamicWorkspace(n_qubits::Int, n_params::Int)
    DynamicWorkspace(
        n_qubits,
        ArrayReg[],  # Empty pool, grows on demand
        zeros(ComplexF64, n_params),
        zeros(Float64, n_params),
        zeros(Float64, n_params * 2),
        Ref(0)
    )
end

# --- Interface implementations for PreallocatedWorkspace ---

function get_vectors!(ws::PreallocatedWorkspace, count::Int)
    start_idx = ws.offset[] + 1
    end_idx = ws.offset[] + count
    @assert end_idx <= length(ws.statevec_pool) "Workspace capacity exceeded: requested $(count) vectors starting at offset $(ws.offset[]), but pool size is $(length(ws.statevec_pool))"
    ws.offset[] = end_idx
    return @view ws.statevec_pool[start_idx:end_idx]
end

function get_grad_buffers!(ws::PreallocatedWorkspace)
    return (ws.grad_collector, ws.grad_angles, ws.grad_params)
end

function reset!(ws::PreallocatedWorkspace)
    ws.offset[] = 0
    fill!(ws.grad_params, 0.0)
    return nothing
end

get_forward_tile_size(ws::PreallocatedWorkspace) = ws.forward_capacity
get_backward_tile_size(ws::PreallocatedWorkspace) = ws.backward_capacity
get_backward_tiled_tile_size(ws::PreallocatedWorkspace) = ws.backward_tiled_capacity

# --- Interface implementations for DynamicWorkspace ---

function get_vectors!(ws::DynamicWorkspace, count::Int)
    # Check if we need to allocate more vectors
    num_needed = (ws.offset[] + count) - length(ws.statevec_pool)
    if num_needed > 0
        # Allocate only the missing vectors and add them to the pool
        new_vectors = [zero_state(ComplexF64, ws.n_qubits) for _ in 1:num_needed]
        append!(ws.statevec_pool, new_vectors)
    end
    
    # Return the view and advance the offset
    start_idx = ws.offset[] + 1
    end_idx = ws.offset[] + count
    ws.offset[] = end_idx
    return @view ws.statevec_pool[start_idx:end_idx]
end

function get_grad_buffers!(ws::DynamicWorkspace)
    return (ws.grad_collector, ws.grad_angles, ws.grad_params)
end

function reset!(ws::DynamicWorkspace)
    ws.offset[] = 0
    fill!(ws.grad_params, 0.0)
    return nothing
end

# Dynamic workspace has no fixed capacity - it grows as needed
get_forward_tile_size(ws::DynamicWorkspace) = typemax(Int)
get_backward_tile_size(ws::DynamicWorkspace) = typemax(Int)
get_backward_tiled_tile_size(ws::DynamicWorkspace) = typemax(Int)

# --- Helper functions for memory views ---

"""
    WorkspaceViews

Convenience struct for managing different memory regions during computation.
"""
struct WorkspaceViews{T<:AbstractVector{<:ArrayReg}}
    statevecs::T
    adjoints::Union{T,Nothing}
    col_statevecs::Union{T,Nothing}  # For tiled operations
end

"""
    get_forward_views(workspace::AbstractFidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for forward pass computation.
"""
function get_forward_views(workspace::AbstractFidelityWorkspace, n_samples::Int)
    reset!(workspace)
    statevecs = get_vectors!(workspace, n_samples)
    return WorkspaceViews(statevecs, nothing, nothing)
end

"""
    get_backward_views(workspace::AbstractFidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for backward pass computation (2-way split).
"""
function get_backward_views(workspace::AbstractFidelityWorkspace, n_samples::Int)
    reset!(workspace)
    statevecs = get_vectors!(workspace, n_samples)
    adjoints = get_vectors!(workspace, n_samples)
    return WorkspaceViews(statevecs, adjoints, nothing)
end

"""
    get_backward_tiled_views(workspace::AbstractFidelityWorkspace, row_size::Int, col_size::Int) -> WorkspaceViews

Get memory views configured for tiled backward pass (3-way split).
"""
function get_backward_tiled_views(workspace::AbstractFidelityWorkspace, row_size::Int, col_size::Int)
    reset!(workspace)
    row_statevecs = get_vectors!(workspace, row_size)
    adjoints = get_vectors!(workspace, row_size)
    col_statevecs = get_vectors!(workspace, col_size)
    return WorkspaceViews(row_statevecs, adjoints, col_statevecs)
end

"""
    extract_gradients(workspace::AbstractFidelityWorkspace) -> (weights, biases)

Extract weight and bias gradients from the interleaved gradient buffer.
"""
function extract_gradients(workspace::AbstractFidelityWorkspace)
    _, _, grad_params = get_grad_buffers!(workspace)
    # Slicing already creates a copy, no need for additional copy()
    d_weights = grad_params[1:2:end]
    d_biases = grad_params[2:2:end]
    return d_weights, d_biases
end

# --- Legacy compatibility functions ---

"""
    create_workspace(args...; kwargs...) 

Legacy function name - redirects to create_preallocated_workspace.
"""
create_workspace(args...; kwargs...) = create_preallocated_workspace(args...; kwargs...)

# Alias for backward compatibility
const FidelityWorkspace = PreallocatedWorkspace
