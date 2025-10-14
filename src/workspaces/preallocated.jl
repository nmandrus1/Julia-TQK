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
    # for plus and minus computations of weights
    w_buffer::Vector{Float64}
    b_buffer::Vector{Float64}
    S_matrix::Matrix{ComplexF64}
    K_complex::Matrix{ComplexF64}
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
    
    # --- THIS IS THE CORRECTED PART ---
    # 1. Allocate the S_matrix directly. It will be our primary storage.
    S_matrix = zeros(ComplexF64, 2^num_qubits, max_samples)

    # 2. Create the statevec_pool as a vector of ArrayRegs that are 
    #    VIEWS into the columns of S_matrix. This is highly memory-efficient
    #    and ensures data consistency.
    statevec_pool = [ArrayReg(@view S_matrix[:, i]) for i in 1:max_samples]
    
    # ... (rest of the allocations are the same) ...
    grad_buffer = zeros(Float64, 2 * num_params)
    w_buffer = zeros(Float64, num_params)
    b_buffer = zeros(Float64, num_params)
    K_cache = zeros(Float64, K_dims)
    K_complex = zeros(ComplexF64, K_dims)
    
    return PreallocatedWorkspace(
        statevec_pool,
        grad_buffer,
        K_cache,
        w_buffer,
        b_buffer,
        S_matrix,
        K_complex
    )
end

function get_statevectors(ws::PreallocatedWorkspace)
    ws.statevec_pool
end
