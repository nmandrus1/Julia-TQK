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
    grad_buffer::Vector{Float64}
    K_cache::Matrix{Float64}  # Pre-allocated kernel matrix
end

"""
    create_preallocated_workspace(feature_map, max_samples)

Create workspace sized to memory budget.
"""
function create_preallocated_workspace(
    feature_map::AbstractQuantumFeatureMap,
    max_samples::Int,
    K_dims::Tuple{Int, Int},
)
    num_qubits = n_qubits(feature_map)
    num_params = n_params(feature_map)
    
    statevec_pool = [zero_state(ComplexF64, num_qubits) for _ in 1:max_samples]
    grad_buffer = zeros(Float64, 2 * num_params)
    K_cache = zeros(Float64, K_dims)
    
    return PreallocatedWorkspace(statevec_pool, grad_buffer, K_cache)
end
