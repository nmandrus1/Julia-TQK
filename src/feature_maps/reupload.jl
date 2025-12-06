using YaoAPI
using Yao
using ArgCheck

# Keep your Enum for clarity
@enum EntanglementStrategy LinearEntanglement CircularEntanglement FullEntanglement NoEntanglement

"""
    ReuploadingConfig
    
Immutable blueprint for a Data Reuploading Circuit.
Stores the architecture and the 'wiring' of features to gates.
"""
struct ReuploadingConfig <: AbstractFeatureMapConfig
    n_qubits::Int
    n_features::Int
    n_layers::Int
    entanglement::EntanglementStrategy
    
    # Pre-computed layout: which feature index (1..M) maps to which gate (1..P)
    # This avoids logic inside the hot loop.
    gate_feature_indices::Vector{Int}
    
    # Total trainable parameters (weights + biases)
    total_params::Int
end

# --- Constructor ---
function ReuploadingConfig(n_qubits::Int, n_features::Int, n_layers::Int; entanglement::EntanglementStrategy=LinearEntanglement)
    gate_feature_indices = Int[]
    
    # 1. Pre-calculate the wiring layout (exact same logic as your old code)
    n_chunks = div(n_features, 3)
    remaining_encodings = n_features - 3 * n_chunks
    
    # Simulate the loop to capture indices
    for layer in 1:n_layers
        for q in 1:n_qubits
            # Full chunks (Rz, Ry, Rz)
            for k in 0:n_chunks-1
                push!(gate_feature_indices, k*3 + 3) # Rz
                push!(gate_feature_indices, k*3 + 1) # Ry
                push!(gate_feature_indices, k*3 + 2) # Rz
            end
            
            # Remaining features
            if remaining_encodings == 2
                push!(gate_feature_indices, n_chunks*3 + 1)
                push!(gate_feature_indices, n_chunks*3 + 2)
            elseif remaining_encodings == 1
                push!(gate_feature_indices, n_chunks*3 + 1)
            end
        end
    end
    
    # 2. Calculate Parameter Count
    # Each gate that has a feature mapping gets 1 weight and 1 bias.
    num_encoding_gates = length(gate_feature_indices)
    total_params = 2 * num_encoding_gates 

    return ReuploadingConfig(
        n_qubits, 
        n_features, 
        n_layers, 
        entanglement, 
        gate_feature_indices,
        total_params
    )
end

# --- Interface Implementation ---

n_qubits(c::ReuploadingConfig) = c.n_qubits
n_trainable_params(c::ReuploadingConfig) = c.total_params

"""
    build_circuit(config, params, x)

The 'Circuit Factory'. 
Pure function: Input -> Output. No side effects. 
Zygote can differentiate this because it just sees array math and struct construction.
"""

function build_circuit(config::ReuploadingConfig, params::AbstractVector, x::AbstractVector)
    @argcheck length(params) == config.total_params
    @argcheck length(x) == config.n_features
    
    # 1. Calculate all angles at once (Vectorized)
    n_gates_total = length(config.gate_feature_indices)
    weights = view(params, 1:n_gates_total)
    biases = view(params, n_gates_total+1:2*n_gates_total)
    
    relevant_features = x[config.gate_feature_indices]
    angles = weights .* relevant_features .+ biases
    
    # 2. Define geometry for slicing 'angles'
    n_chunks = div(config.n_features, 3)
    rem = config.n_features - 3 * n_chunks
    
    # Gates per qubit = 3 * chunks + remainder gates
    gates_per_qubit = 3 * n_chunks + (rem == 2 ? 2 : (rem == 1 ? 1 : 0))
    gates_per_layer = config.n_qubits * gates_per_qubit

    # 3. Construct Layers using 'map' (No mutation!)
    layers = map(1:config.n_layers) do l
        # A. Calculate slice indices for this layer
        layer_start_idx = (l - 1) * gates_per_layer
        
        # B. Construct gates for each qubit in this layer
        qubit_chains = map(1:config.n_qubits) do q
             # Slice angles for this specific qubit
             q_offset = (q-1) * gates_per_qubit
             base = layer_start_idx + q_offset
             
             # Extract angles safely
             q_angles = view(angles, base+1 : base+gates_per_qubit)
             
             # Build the chain for this qubit
             build_qubit_gates(q, q_angles, n_chunks, rem)
        end
        
        # C. Combine qubit chains into a layer block
        # vcat flattens the list of lists
        layer_block = chain(config.n_qubits, vcat(qubit_chains...))
        
        # D. Add entanglement if needed
        if l < config.n_layers && config.n_qubits > 1
             ent = make_entanglement(config.n_qubits, config.entanglement)
             chain(config.n_qubits, layer_block, ent)
        else
             layer_block
        end
    end

    # 4. Chain all layers together
    return chain(config.n_qubits, layers...)
end

"""
    build_qubit_gates(q, angles, n_chunks, rem)

Helper to construct the sequence of gates for a single qubit.
Uses mapping based on index to determine Gate Type (Ry vs Rz).
"""
function build_qubit_gates(q::Int, angles::AbstractVector, n_chunks::Int, rem::Int)
    # We map over the indices of the angles vector to create gates.
    # Logic: 
    #   Chunks (indices 1 to 3*n_chunks): Rz, Ry, Rz pattern
    #   Remainder: Ry, then Rz (if exists)
    
    num_chunk_gates = n_chunks * 3
    
    map(1:length(angles)) do i
        if i <= num_chunk_gates
            # Chunk Logic: 1->Rz, 2->Ry, 3->Rz...
            mod_val = (i-1) % 3
            if mod_val == 0
                return put(q => Rz(angles[i]))
            elseif mod_val == 1
                return put(q => Ry(angles[i]))
            else
                return put(q => Rz(angles[i]))
            end
        else
            # Remainder Logic
            rem_idx = i - num_chunk_gates
            if rem_idx == 1
                return put(q => Ry(angles[i]))
            else
                return put(q => Rz(angles[i]))
            end
        end
    end
end

function make_entanglement(n::Int, strategy::EntanglementStrategy)
    if strategy == LinearEntanglement
        return chain(n, [control(i, i+1 => X) for i in 1:n-1])
    elseif strategy == CircularEntanglement
        c = chain(n, [control(i, i+1 => X) for i in 1:n-1])
        return chain(c, control(n, 1 => X))
    elseif strategy == FullEntanglement
        return chain(n, [control(i, j => X) for i in 1:n for j in 1:n if i!=j])
    else
        return chain(n)
    end
end
