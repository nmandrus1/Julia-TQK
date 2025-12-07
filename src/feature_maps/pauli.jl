using Yao

"""
    PauliConfig
    
Immutable blueprint for a PauliFeatureMap Circuit.
Stores the architecture and the 'wiring' of features to gates.
"""
struct PauliConfig <: AbstractFeatureMapConfig
    n_features::Int
    reps::Int
    paulis::Vector{String}
    ent::EntanglementStrategy
end

n_qubits(c::PauliConfig) = c.n_features
n_trainable_params(c::PauliConfig) = 0

# --- Helper: The Pauli Evolution Circuit (from previous step) ---
"""
    pauli_evolution_circuit(n::Int, pauli_string::String, theta::Real)

Generates the circuit for exp(-i * theta/2 * P).
Matches Qiskit's `PauliEvolutionGate` logic.
"""
function pauli_evolution_circuit(n::Int, pauli_string::String, theta::Real)
    # Parse string: Qiskit strings are reversed (index 0 is rightmost).
    # We map string index i (from right) to Julia qubit index i+1.
    active_ops = []
    reversed_s = reverse(pauli_string) # Handle Qiskit's little-endian string
    
    for (i, char) in enumerate(reversed_s)
        if char != 'I'
            push!(active_ops, (i, char)) # i is 1-based index in Julia
        end
    end
    
    if isempty(active_ops)
        return chain(n)
    end
    
    indices = [x[1] for x in active_ops]
    gates   = [x[2] for x in active_ops]
    
    # 1. Basis Change
    basis_change = chain(n)
    for (idx, gate_char) in zip(indices, gates)
        if gate_char == 'X'
            push!(basis_change, put(n, idx => H))
        elseif gate_char == 'Y'
            push!(basis_change, put(n, idx => Rx(π/2))) # Y -> Z basis
        end
    end
    
    # 2. CNOT Ladder (Parity)
    parity_ladder = chain(n)
    for k in 1:(length(indices) - 1)
        push!(parity_ladder, cnot(indices[k], indices[k+1]))
    end
    
    # 3. Rotation
    # The rotation Rz(theta) implements exp(-i * theta/2 * Z)
    target_qubit = indices[end]
    central_rot = put(n, target_qubit => Rz(theta))
    
    # 4. Construct Full Block (Uncompute automatically via adjoint)
    return chain(
        basis_change,
        parity_ladder,
        central_rot,
        parity_ladder',
        basis_change'
    )
end

# --- Helper: Default Data Mapping Function ---
"""
    qiskit_default_data_map(x::Vector, indices::Vector{Int})

Replicates Qiskit's default non-linear data map:
- If |S| = 1: x_i
- If |S| > 1: Π (π - x_j) for j in S
"""
function qiskit_default_data_map(x::AbstractVector, indices::Vector{Int})
    if length(indices) == 1
        return x[indices[1]]
    else
        val = 1.0
        for idx in indices
            val *= (π - x[idx])
        end
        return val
    end
end

# --- Main Function: PauliFeatureMap ---
"""
    pauli_feature_map(n_qubits::Int, x::Vector, paulis::Vector{String}; 
                      reps::Int=2, alpha::Real=2.0)

Constructs a Yao.jl block identical to Qiskit's PauliFeatureMap.

Arguments:
- `n_qubits`: Number of qubits (feature dimension).
- `x`: The classical data vector (should match n_qubits).
- `paulis`: List of Pauli strings (e.g. ["Z", "ZZ"]). 
            Note: Uses Qiskit string ordering (rightmost char is q0).
- `reps`: Number of circuit repetitions (default: 2).
- `alpha`: Scaling factor for rotations (default: 2.0).
"""
function pauli_feature_map(n_qubits::Int, x::AbstractVector, paulis::Vector{String}; 
                           reps::Int=2, alpha::Real=2.0)
    
    # Initialize the circuit chain
    # Standard Qiskit structure: [H layer] -> [Evolutions] -> [H layer] -> [Evolutions] ...
    full_circuit = chain(n_qubits)
    
    for _ in 1:reps
        # 1. Hadamard Layer (applied at start of every repetition)
        push!(full_circuit, repeat(n_qubits, H, 1:n_qubits))
        
        # 2. Evolution Layer
        for p_str in paulis
            # Identify active indices for this pauli string
            # (Need to parse again to get indices for data mapping)
            rev_s = reverse(p_str)
            active_indices = [i for (i, c) in enumerate(rev_s) if c != 'I']
            
            # Skip Identity strings
            if isempty(active_indices)
                continue
            end
            
            # Calculate rotation angle using data map
            # Qiskit logic: angle = alpha * data_map(x)
            phi_val = qiskit_default_data_map(x, active_indices)
            theta = alpha * phi_val
            
            # Generate and push the evolution block
            evol_block = pauli_evolution_circuit(n_qubits, p_str, theta)
            push!(full_circuit, evol_block)
        end
    end
    
    return full_circuit
end


"""
    build_circuit(config, params, x)

The 'Circuit Factory'. 
Pure function: Input -> Output. No side effects. 
Zygote can differentiate this because it just sees array math and struct construction.

NOTE: PauliFeatureMap has no trainable parameters, so `params` argument is ignored
"""
function build_circuit(config::PauliConfig, params::AbstractVector, x::AbstractVector)
    return pauli_feature_map(config.n_features, x, config.paulis; reps=config.reps)
end

# --- Usage Example for Research ---

# 1. Define inputs
n = 3
x_data = [0.1, 0.2, 0.3]
# Standard "ZZFeatureMap" equivalent: Z on each, then ZZ interactions
pauli_list = ["ZII", "IZI", "IIZ", "ZZI", "ZIZ", "IZZ"] 

# 2. Build the circuit
# This returns a specific circuit encoded with the data `x_data`
qc = pauli_feature_map(n, x_data, pauli_list, reps=2)

# 3. Verify parameters or use in Kernel
println("Circuit Depth (Blocks): ", length(qc))
println(qc)
