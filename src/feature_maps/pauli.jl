using Yao
using YaoAPI
using ArgCheck

struct PauliConfig <: AbstractFeatureMapConfig
    n_features::Int             # Corresponds to 'feature_dimension' in Qiskit
    reps::Int
    paulis::Vector{String}      # e.g., ["Z", "ZZ"]
    ent::EntanglementStrategy
    alpha::Float64
end

# Default constructor to match Qiskit defaults
function PauliConfig(n::Int, paulis::Vector{String}=["Z", "ZZ"]; 
                     reps::Int=2, 
                     ent::EntanglementStrategy=FullEntanglement, 
                     alpha::Real=2.0)
    return PauliConfig(n, reps, paulis, ent, alpha)
end

n_qubits(c::PauliConfig) = c.n_features
n_trainable_params(c::PauliConfig) = 0 # Pauli Feature Map has no trainable weights

# --- Pure Helper: Entanglement Index Generation ---
# Matches Qiskit's `entanglement.get_entanglement` logic
function get_entanglement_indices(n_qubits::Int, block_size::Int, strategy::EntanglementStrategy)
    # If block_size is 1, it always applies to every qubit individually
    if block_size == 1
        return [[i] for i in 1:n_qubits]
    end

    # If block_size == n_qubits, only one block covers everything
    if block_size == n_qubits
        return [collect(1:n_qubits)]
    end

    indices = Vector{Vector{Int}}()

    if strategy == LinearEntanglement
        # [1, 2], [2, 3], ...
        for i in 1:(n_qubits - 1)
            # Generate block [i, i+1, ... i+block_size-1]
            # Must check bounds (Qiskit linear usually just does nearest neighbor chains if block_size=2)
            # For general block_size k: i, i+1, ..., i+k-1
            if i + block_size - 1 <= n_qubits
                push!(indices, collect(i:(i + block_size - 1)))
            end
        end

    elseif strategy == CircularEntanglement
        # Linear + wrapping boundaries
        for i in 1:n_qubits
            block = Int[]
            for j in 0:(block_size - 1)
                # 1-based wrapping: (i + j - 1) % n + 1
                val = ((i + j - 1) % n_qubits) + 1
                push!(block, val)
            end
            push!(indices, block)
        end

    elseif strategy == FullEntanglement
        # All combinations of size `block_size`
        # Using Julia's Combinatorics would be standard, but let's do a simple 2-qubit case
        # (which is 99% of use cases) manually to stay dependency-free if possible.
        if block_size == 2
            for i in 1:n_qubits
                for j in (i + 1):n_qubits
                    push!(indices, [i, j])
                end
            end
        else
             # Fallback for >2 (Simple sliding window or similar if Combinatorics not avail)
             # Qiskit 'full' for k>2 implies all k-subsets. 
             # For simplicity in this snippet, we default to Linear if k > 2 to avoid huge complexity 
             # without external libs, or assume standard Z/ZZ usage.
             # *Strictly* for Z/ZZ (size 1 and 2), the above covers it.
             return get_entanglement_indices(n_qubits, block_size, LinearEntanglement)
        end
    end
    
    return indices
end


# --- Pure Helper: Data Mapping ---
# Matches Rust `_default_reduce`
function qiskit_data_map(x::AbstractVector, indices::Vector{Int})
    if length(indices) == 1
        return x[indices[1]]
    else
        return mapreduce(i -> (π - x[i]), *, indices; init=1.0)
    end
end

# --- Pure Helper: Native Gates ---
function make_1q_gate(n::Int, idx::Int, gate_char::Char, theta::Real)
    if gate_char == 'X'
        return put(n, idx => Rx(theta))
    elseif gate_char == 'Y'
        return put(n, idx => Ry(theta))
    elseif gate_char == 'Z'
        return put(n, idx => Rz(theta))
    else
        return chain(n)
    end
end

function make_2q_gate(n::Int, idx1::Int, idx2::Int, gate_str::String, theta::Real)
    if gate_str == "XX"
        return rot(kron(n, idx1=>X, idx2=>X), theta)
    elseif gate_str == "YY"
        return rot(kron(n, idx1=>Y, idx2=>Y), theta)
    elseif gate_str == "ZZ"
        return rot(kron(n, idx1=>Z, idx2=>Z), theta)
    # elseif gate_str == "ZX"
    #     return rot(kron(n, idx1=>Z, idx2=>X), theta)
    # elseif gate_str == "XZ"
    #     return rot(kron(n, idx1=>X, idx2=>Z), theta)
    else
        return nothing 
    end
end


# --- Pure Helper: Basis Changes (CORRECTED) ---
function basis_change_layer(n::Int, indices::Vector{Int}, gates::Vector{Char})
    # Filter and create ONLY the active gates
    ops = AbstractBlock[]
    for (idx, g) in zip(indices, gates)
        if g == 'X'
            push!(ops, put(n, idx => H))
        elseif g == 'Y'
            push!(ops, put(n, idx => Rx(π/2))) # SX
        end
        # Ignore 'Z' or others (Identity)
    end
    
    # If ops is empty (e.g. all Z), chain(n) returns global Identity
    if isempty(ops)
        return chain(n)
    end
    
    return chain(n, ops...)
end

function inv_basis_change_layer(n::Int, indices::Vector{Int}, gates::Vector{Char})
    ops = AbstractBlock[]
    for (idx, g) in zip(indices, gates)
        if g == 'X'
            push!(ops, put(n, idx => H))
        elseif g == 'Y'
            push!(ops, put(n, idx => Rx(-π/2))) # SXdg
        end
    end

    if isempty(ops)
        return chain(n)
    end
    
    return chain(n, ops...)
end

# --- Pure Helper: Evolution Block (Fixed CNOT Direction) ---

# --- Pure Helper: Evolution Block (Fixed CNOT Direction & Endianness) ---
function pauli_evolution_block(n::Int, indices::Vector{Int}, pauli_type::String, theta::Real)
    
    k = length(indices)
    
    # FIX: Reverse the string to match Qiskit's Little-Endian (Right-to-Left) parsing
    # "XZ" becomes ['Z', 'X']. zipped with [1, 2] -> Z on 1, X on 2.
    gates = reverse(collect(pauli_type)) 

    # Case 1: Single Qubit (No ladder needed)
    if k == 1
        return make_1q_gate(n, indices[1], gates[1], theta)
    end

    # Case 2: Multi-Qubit (Generic Ladder Logic)
    
    # A. Basis Change
    basis = basis_change_layer(n, indices, gates)
    
    # B. CNOT Ladder (High -> Low)
    # Qiskit Rust: (0..k-1).map(|i| CNOT(qubits[i+1], qubits[i]))
    ladder_ops = [cnot(indices[i+1], indices[i]) for i in 1:(k-1)]
    ladder = chain(n, ladder_ops...)
    
    # C. Rotation
    # Qiskit rotates the FIRST qubit in the chain
    target_qubit = indices[1]
    rot_op = put(n, target_qubit => Rz(theta))
    
    # D. Inverse Basis
    inv_basis = inv_basis_change_layer(n, indices, gates)

    return chain(
        basis,
        ladder,
        rot_op,
        ladder', # Uncompute
        inv_basis
    )
end

function pauli_feature_map(n_qubits::Int, x::AbstractVector, paulis::Vector{String}, 
                           reps::Int, ent::EntanglementStrategy, alpha::Real)
    
    # Pure map over repetitions
    layers = map(1:reps) do rep_idx
        
        # 1. Hadamard Layer
        h_layer = repeat(n_qubits, H, 1:n_qubits)
        
        # 2. Evolution Layer
        # Iterate over Pauli Types (e.g., "Z", then "ZZ")
        type_blocks = map(paulis) do p_type
            
            # A. Get indices for this type and strategy
            # Note: Qiskit allows rotation of entanglement per rep, but standard is static.
            # We use the static logic here for simplicity.
            active_indices_list = get_entanglement_indices(n_qubits, length(p_type), ent)
            
            # B. Build blocks for each set of indices (e.g., Z on 1, Z on 2...)
            blocks = map(active_indices_list) do indices
                
                # Calculate Angle using Data Map on specific indices
                phi_val = qiskit_data_map(x, indices)
                theta = alpha * phi_val
                
                # Create Block
                pauli_evolution_block(n_qubits, indices, p_type, theta)
            end
            
            # Chain all blocks for this Pauli Type together
            chain(blocks...)
        end
        
        # Combine H layer + All Evolution blocks for this rep
        chain(h_layer, type_blocks...)
    end

    return chain(layers...)
end

"""
    build_circuit(config, params, x)
"""
function build_circuit(config::PauliConfig, params::AbstractVector, x::AbstractVector)
    return pauli_feature_map(config.n_features, x, config.paulis, config.reps, config.ent, config.alpha)
end
