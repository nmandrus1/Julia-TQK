using SymEngine, Yao, Yao.AD, LinearAlgebra, ArgCheck
 
# Don't include types.jl here - it's already loaded by the parent module!
# include("../types.jl")  # REMOVE THIS LINE

@enum EntanglementBlock linear alternating all_to_all

"""
    create_entanglement_block(n_qubits::Int, strategy::Symbol = :linear)

Creates an entangling layer.

# Arguments
- `n_qubits`: The total number of qubits.
- `strategy`: A symbol indicating the entanglement pattern (:linear, :circular, etc.).

# Returns
- A `ChainBlock` containing the entangling gates.
"""
function create_entanglement_block(n_qubits::Int, strategy::EntanglementBlock = linear)
    if n_qubits <= 1
        return chain() # No entanglement for 1 qubit
    end

    if strategy == linear
        # Entangles qubit i with qubit i+1
        return chain(n_qubits, [control(i, i+1 => X) for i in 1:n_qubits-1])
    elseif strategy == all_to_all
        # Entangles every qubit with every other qubit
         gates = [control(i, j => X) for i in 1:n_qubits for j in 1:n_qubits if i != j]
         return chain(n_qubits, gates)
    elseif strategy == alternating
        # As seen in the paper's 4-qubit example
        # This would need the layer index to alternate properly.
        # For simplicity, we'll use a fixed example here.
        cz_set1 = chain(n_qubits, control(1, 2 => Z), control(3, 4 => Z))
        cz_set2 = chain(n_qubits, control(2, 3 => Z), control(1, 4 => Z))
        return chain(cz_set1, cz_set2) # Example for a single layer
    else
        error("Unknown entanglement strategy: $strategy")
    end
end

"""
    collect_parameterized_gates(block::AbstractBlock)

Recursively traverse the circuit and collect the gates into
a vector for efficient parameter updates. 
"""

function collect_parameterized_gates(block::AbstractBlock)
    gates = []
    if nparameters(block) > 0 && block isa PrimitiveBlock
        push!(gates, block)
    elseif block isa CompositeBlock
        for sub in subblocks(block)
            append!(gates, collect_parameterized_gates(sub))
        end
    end
    return gates
end

"""
    ReuploadingCircuit

Manages a quantum circuit with w*x + θ parameterization.

# Fields
- `circuit`: The Yao circuit with placeholder parameters
- `weights`: Vector of weight parameters
- `biases`: Vector of bias parameters  
- `gate_features`: Feature index used by each parameterized gate
- `n_features`: The feature dimension of the data being mapped by this circuit
- `gates`: collection of gates with parameters for efficient updates
"""
struct ReuploadingCircuit <: AbstractQuantumFeatureMap
    circuit::ChainBlock
    weights::Vector{Float64}
    biases::Vector{Float64}
    angles::Vector{Float64}
    gate_features::Vector{Int64}
    n_features::Int64
    parameterized_gates::Vector{PrimitiveBlock}
end

"""
    ReuploadingCircuit(n_qubits::Int, n_features::Int, n_layers::Int, entanglement::EntanglementBlock)

Build a data-reuploading circuit with parameterized gates.

# Arguments
- `n_qubits`: Number of qubits
- `n_features`: Number of input features
- `n_layers`: Number of circuit layers
- `entanglement`: Entanglement between layers

# Returns
- `ReuploadingCircuit` object with tracked parameter mappings
"""

function ReuploadingCircuit(n_qubits::Int, n_features::Int, n_layers::Int, entanglement::EntanglementBlock)
    all_gates = []
    gate_features = Int[]

    n_chunks = div(n_features, 3)
    remaining_encodings = n_features - 3*n_chunks
    
    for layer in 1:n_layers
        # Single-qubit rotation layer
        for q in 1:n_qubits
            for k in 0:n_chunks-1
                # Rz(2) Ry(1) Rz(3)
                push!(all_gates, put(q => Rz(0.0)))
                push!(all_gates, put(q => Ry(0.0)))
                push!(all_gates, put(q => Rz(0.0)))
                push!(gate_features, k*3 + 3)
                push!(gate_features, k*3 + 1)
                push!(gate_features, k*3 + 2)
            end           
            if remaining_encodings == 2
                push!(all_gates, put(q => Rz(0.0)))
                push!(gate_features, n_chunks*3 + 2)
                push!(all_gates, put(q => Ry(0.0)))
                push!(gate_features, n_chunks*3 + 1)
            elseif remaining_encodings == 1
                push!(all_gates, put(q => Ry(0.0)))
                push!(gate_features, n_chunks*3 + 1)
            end
           
        end
        
        # Entangling layer
        if layer < n_layers && n_qubits > 1
            entanglement_block = create_entanglement_block(n_qubits, entanglement)
            if !isempty(entanglement_block.blocks)
                # Append the gates from the entanglement block to the flat list
                append!(all_gates, entanglement_block.blocks)
            end
        end
    end
    
    circuit = chain(n_qubits, all_gates...)
    n_params = nparameters(circuit)

    parameterized_gates = collect_parameterized_gates(circuit)

    return ReuploadingCircuit(circuit, zeros(n_params), zeros(n_params), zeros(n_params), gate_features, n_features, parameterized_gates)
end

"""
    compute_angles!(pc::ParameterizedCircuit, weights::Vector{Float64}, 
                   biases::Vector{Float64}, x::Vector{Float64})

Compute gate angles in place using formula: angle[i] = w[i] * x[feature[i]] + b[i]
using the weights and biases stored in the circuit

# Arguments
- `pc`: ParameterizedCircuit object
- `x`: Input feature vector

# Returns
- Vector of angles for dispatch!
"""
function compute_angles!(pc::ReuploadingCircuit, x::AbstractVector)   
    @argcheck length(x) == pc.n_features

    for i in eachindex(pc.weights)
        feature_idx = pc.gate_features[i]
        pc.angles[i] = pc.weights[i] * x[feature_idx] + pc.biases[i]
    end
end


"""
    gradient_chain_rule(grad_angles::Vector{Float64}, x::Vector{Float64}, 
                       gate_features::Vector{Int}, n_params::Int)

Apply chain rule to convert angle gradients to parameter gradients.

# Mathematical formulation
For each gate i with angle = w[i] * x[j] + b[i]:
- ∂L/∂w[i] = ∂L/∂angle[i] * x[j]
- ∂L/∂b[i] = ∂L/∂angle[i]

# Returns
- Vector [grad_w1, grad_b1, grad_w2, grad_b2, ...]
"""
function gradient_chain_rule(grad_angles::Vector{Float64}, x::Vector{Float64}, 
                           gate_features::Vector{Int}, n_params::Int)
    grad_params = Vector{Float64}(undef, 2 * n_params)
    
    @inbounds for i in 1:n_params
        feature_idx = gate_features[i]
        grad_params[2i-1] = grad_angles[i] * x[feature_idx]  # ∂L/∂w
        grad_params[2i] = grad_angles[i]                     # ∂L/∂b
    end
    
    return grad_params
end


"""
    expectation_and_gradient(pc::ParameterizedCircuit, params::Vector{Float64}, 
                           x::Vector{Float64}, observable)

Compute expectation value and parameter gradients using Yao's AD.

# Arguments
- `pc`: ParameterizedCircuit object
- `params`: Flattened parameters [w1, b1, w2, b2, ...]
- `x`: Input feature vector
- `observable`: Yao observable to measure

# Returns
- `(expectation_value, gradient_vector)`
"""
function expectation_and_gradient(reup_circ::ReuploadingCircuit, params::Vector{Float64}, 
                                x::Vector{Float64}, observable)
    # Split parameters
    weights = params[1:2:end]
    biases = params[2:2:end]
    
    # Compute and dispatch angles
    compute_angles!(reup_circ, x)
    dispatch!(reup_circ.circuit, reup_circ.angles)
    
    # Get expectation and angle gradients from Yao
    reg = zero_state(nqubits(reup_circ.circuit))
    val, grad_angles = expect'(observable, reg => reup_circ.circuit)
    
    # Apply chain rule
    grad_params = gradient_chain_rule(grad_angles, x, reup_circ.gate_features, reup_circ.n_weights)
    
    return real(val), grad_params
end


# ================================= 
# AbstractQuantumFeatureMap interface
# ================================= 


n_qubits(fm::ReuploadingCircuit) = nqubits(fm.circuit)
n_features(fm::ReuploadingCircuit) = fm.n_features
get_params(fm::ReuploadingCircuit) = (fm.weights, fm.biases)

"""
  n_parameters(fm::ReuploadingCircuit)  

Returns the number of **angles** calculated by the circuit.
Each angle is calculated by angle = w * x_feature + b
"""
n_parameters(fm::ReuploadingCircuit) = nparameters(fm.circuit)


function map_inputs!(fm::ReuploadingCircuit, x::AbstractVector)
    compute_angles!(fm, x)
    for (gate, angle) in zip(fm.parameterized_gates, fm.angles)
        setiparams!(gate, angle)
    end
end



"""
    assign_params!(fm::ReuploadingCircuit, weights::Vector{Float64}, biases::Vector{Float64})       

    Takes two vectors of weights and biases parameters and stores them to be used in the final computation of
    the rotation gates.

"""
function assign_params!(fm::ReuploadingCircuit, weights::Vector{Float64}, biases::Vector{Float64})
    @assert length(weights) == length(fm.weights)
    @assert length(biases) == length(fm.biases)
    fm.weights .= weights  # Use .= for in-place copy
    fm.biases .= biases
end


"""
    assign_random_params!(fm::ReuploadingCircuit, range::Tuple{Float64, Float64} = (-π, π); seed = 11)       

    Assigns the parameters of the circuit to random values distributed uniformly in the given range.
"""
function assign_random_params!(fm::ReuploadingCircuit, range::Tuple{<:Real, <:Real} = (-π, π); seed = 11)
    Random.seed!(seed)
    n_params = nparameters(fm.circuit)
    
    # Correct random generation using range
    fm.weights .= range[1] .+ (range[2] - range[1]) .* rand(n_params)
    fm.biases .= range[1] .+ (range[2] - range[1]) .* rand(n_params)
end
