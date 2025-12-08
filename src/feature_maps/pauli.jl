using Yao
using YaoAPI
using ArgCheck

# --- 1. The Blueprint Structs ---

"""
    PauliTermBlueprint

Represents the pre-computed structure of a single Pauli evolution term exp(-i * theta * P).
The constant parts (Basis Change, CNOT ladders) are built once and stored here.
"""
struct PauliTermBlueprint{B1<:AbstractBlock, B2<:AbstractBlock}
    # The constant block to apply BEFORE the rotation (Basis Change + CNOT Ladder)
    pre_block::B1
    
    # The constant block to apply AFTER the rotation (Uncompute Ladder + Uncompute Basis)
    post_block::B2
    
    # The qubit index that receives the Rz(theta) rotation
    target_qubit::Int
    
    # The indices of the input vector 'x' needed to compute theta
    data_indices::Vector{Int}
end

"""
    PauliConfig

Stores the configuration and the "compiled" blueprint of the circuit layers.
"""
struct PauliConfig <: AbstractFeatureMapConfig
    n_features::Int
    reps::Int
    paulis::Vector{String}
    ent::EntanglementStrategy # Restored for parity
    alpha::Float64
    
    # We organize blueprints by layer (repetition)
    # Each repetition is a Vector of PauliTermBlueprint
    layers::Vector{Vector{PauliTermBlueprint}}
end

n_qubits(c::PauliConfig) = c.n_features
n_trainable_params(c::PauliConfig) = 0 # No trainable weights in this version

# --- 2. Structural Logic (The "Compiler") ---
# This runs ONCE during construction. Zygote NEVER sees this code.

function get_entanglement_indices(n_qubits::Int, block_size::Int, strategy::EntanglementStrategy)
    if block_size == 1
        return [[i] for i in 1:n_qubits]
    end
    if block_size == n_qubits
        return [collect(1:n_qubits)]
    end
    
    indices = Vector{Vector{Int}}()
    if strategy == LinearEntanglement
        for i in 1:(n_qubits - 1)
            if i + block_size - 1 <= n_qubits
                push!(indices, collect(i:(i + block_size - 1)))
            end
        end
    elseif strategy == CircularEntanglement
        for i in 1:n_qubits
            block = Int[]
            for j in 0:(block_size - 1)
                val = ((i + j - 1) % n_qubits) + 1
                push!(block, val)
            end
            push!(indices, block)
        end
    elseif strategy == FullEntanglement
        if block_size == 2
            for i in 1:n_qubits
                for j in (i + 1):n_qubits
                    push!(indices, [i, j])
                end
            end
        else
            return get_entanglement_indices(n_qubits, block_size, LinearEntanglement)
        end
    end
    return indices
end

# Helpers to build the static blocks
function build_basis_change(n::Int, indices::Vector{Int}, gates::Vector{Char})
    ops = map(zip(indices, gates)) do (idx, g)
        if g == 'X'
            put(n, idx => H)
        elseif g == 'Y'
            put(n, idx => Rx(π/2))
        else
            put(n, idx => Rz(0.0)) # Identity
        end
    end
    return chain(n, ops...)
end

function build_inv_basis_change(n::Int, indices::Vector{Int}, gates::Vector{Char})
    ops = map(zip(indices, gates)) do (idx, g)
        if g == 'X'
            put(n, idx => H)
        elseif g == 'Y'
            put(n, idx => Rx(-π/2))
        else
            put(n, idx => Rz(0.0))
        end
    end
    return chain(n, ops...)
end

function build_ladder(n::Int, indices::Vector{Int})
    length(indices) < 2 && return chain(n)
    return chain(n, [cnot(indices[i+1], indices[i]) for i in 1:length(indices)-1]...)
end


function PauliConfig(n::Int, paulis::Vector{String}=["Z", "ZZ"]; 
                     reps::Int=2, 
                     ent::EntanglementStrategy=FullEntanglement, 
                     alpha::Real=2.0)
    
    # Wrap the entire construction logic in Zygote.ignore.
    # This executes normally but tells Zygote: "This output is constant w.r.t gradients."
    layers_blueprints = Zygote.ignore() do
        
        blueprints = Vector{Vector{PauliTermBlueprint}}()
        
        for _ in 1:reps
            current_layer_terms = Vector{PauliTermBlueprint}()
            
            for p_str in paulis
                gates = reverse(collect(p_str))
                p_len = length(p_str)
                active_indices_list = get_entanglement_indices(n, p_len, ent)
                
                for indices in active_indices_list
                    basis = build_basis_change(n, indices, gates)
                    ladder = build_ladder(n, indices)
                    pre_block = chain(n, basis, ladder)
                    post_block = chain(n, ladder', build_inv_basis_change(n, indices, gates))
                    
                    target_qubit = indices[1]
                    
                    push!(current_layer_terms, PauliTermBlueprint(pre_block, post_block, target_qubit, indices))
                end
            end
            push!(blueprints, current_layer_terms)
        end
        
        blueprints # Return the constructed blueprints
    end

    return PauliConfig(n, reps, paulis, ent, alpha, layers_blueprints)
end

# --- 3. The Circuit Factory (The Hot Path) ---
# This is the function Zygote differentiates.
# It iterates the blueprints and performs math. No structural logic.

function build_circuit(config::PauliConfig, params::AbstractVector, x::AbstractVector)
    
    # 1. Map over the layers (reps)
    reps_circuits = map(config.layers) do layer_terms
        
        # A. Create the Hadamard layer (Constant structure)
        h_layer = repeat(config.n_features, H, 1:config.n_features)
        
        # B. Map over the Pauli terms in this layer
        evolutions = map(layer_terms) do bp # bp = blueprint
            
            # --- Differentiable Math Part ---
            # Calculate the data-dependent angle.
            phi_val = if length(bp.data_indices) == 1
                x[bp.data_indices[1]]
            else
                mapreduce(i -> (π - x[i]), *, bp.data_indices; init=1.0)
            end
            
            theta = config.alpha * phi_val
            
            # --- Assembly Part ---
            # Sandwich the variable rotation between the pre-computed blocks.
            # Zygote sees: chain(ConstantBlock, VariableBlock, ConstantBlock)
            rot_gate = put(config.n_features, bp.target_qubit => Rz(theta))
            
            chain(bp.pre_block, rot_gate, bp.post_block)
        end
        
        chain(h_layer, evolutions...)
    end
    
    # 2. Chain all repetitions together
    return chain(config.n_features, reps_circuits...)
end
