using Yao

"""
    PreallocatedWorkspace <: AbstractFidelityWorkspace

Simplified workspace for statevector caching only.

# Fields
- `statevec_pool::Vector{ArrayReg}`: Pre-allocated quantum registers
- `capacity::Int`: Maximum statevectors that fit in memory
- `offset::Ref{Int}`: Current position in pool
"""
struct PreallocatedWorkspace <: AbstractFidelityWorkspace
    statevec_pool::Vector{ArrayReg}
    capacity::Int
    offset::Ref{Int}
end

"""
    create_preallocated_workspace(feature_map, max_samples; memory_budget_gb=4.0)

Create workspace sized to memory budget.
"""
function create_preallocated_workspace(
    feature_map::AbstractQuantumFeatureMap,
    max_samples::Int;
    memory_budget_gb::Float64=4.0
)
    num_qubits = n_qubits(feature_map)
    
    bytes_per_statevec = (2^num_qubits) * sizeof(ComplexF64)
    memory_budget_bytes = memory_budget_gb * (1024^3)
        
    if bytes_per_statevec > memory_budget_bytes
        error("Memory budget insufficient for $(num_qubits)-qubit statevector")
    end
    
    capacity = min(floor(Int, memory_budget_bytes / bytes_per_statevec), max_samples)
    statevec_pool = [zero_state(ComplexF64, num_qubits) for _ in 1:capacity]
    
    println("Workspace capacity: $capacity statevectors")
    
    return PreallocatedWorkspace(statevec_pool, capacity, Ref(0))
end

# Interface implementations
function get_vectors!(ws::PreallocatedWorkspace, count::Int)
    start_idx = ws.offset[] + 1
    end_idx = ws.offset[] + count
    @assert end_idx <= length(ws.statevec_pool) "Workspace capacity exceeded"
    ws.offset[] = end_idx
    return @view ws.statevec_pool[start_idx:end_idx]
end

function reset!(ws::PreallocatedWorkspace)
    ws.offset[] = 0
    return nothing
end

get_capacity(ws::PreallocatedWorkspace) = ws.capacity
