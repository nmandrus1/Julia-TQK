using Yao

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

