using Yao

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
        
    if bytes_per_statevec > memory_budget_bytes
        error("Memory budget of $(memory_budget_gb) GB is insufficient for even a single " *
              "$(num_qubits)-qubit statevector (requires $(bytes_per_statevec / 1024^3) GB)")
    end
    
    # Compute capacities
    forward_capacity = floor(Int, memory_budget_bytes / bytes_per_statevec)
    backward_capacity = floor(Int, memory_budget_bytes / (2 * bytes_per_statevec))
    backward_tiled_capacity = floor(Int, memory_budget_bytes / (3 * bytes_per_statevec))

    println("   Forward Capacity: $forward_capacity")
    println("   Backward Capacity: $backward_capacity")
    println("   Backward (Tiled) Capacity: $backward_tiled_capacity")

    # if we can fit a full backward pass in memory allocate what we need for that
    max_neeed_statevectors_in_memory = 2 * max_samples # adjoints + full dataset 
    
    # We need at LEAST enough for backward
    forward_capacity = min(forward_capacity, max_neeed_statevectors_in_memory)
    backward_capacity = min(backward_capacity, max_neeed_statevectors_in_memory)
    backward_tiled_capacity = min(backward_tiled_capacity, max_neeed_statevectors_in_memory)

    
    # Allocate the pool (sized for full backward capacity, which is largest)
    statevec_pool = [zero_state(ComplexF64, num_qubits) for _ in 1:backward_capacity]
    
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
