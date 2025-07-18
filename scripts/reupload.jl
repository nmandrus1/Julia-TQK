using DrWatson
@quickactivate "TrainableQuantumKernel" # <- project name

using SymEngine, Yao

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

# Modified function to track parameter ordering
function create_single_qubit_layer(d_features::Int, layer_idx::Int, qubit_idx::Int)
    num_rotation_triplets = ceil(Int, d_features / 3)
    gates = []
    w_params = Dict{String, Basic}()
    θ_params = Dict{String, Basic}()
    w_keys = String[]
    θ_keys = String[]
    
    # Create symbolic variables for the entire data vector x
    x_vars = [symbols("x_$i") for i in 1:d_features]
    
    for i in 1:num_rotation_triplets
        # Feature indices for this triplet
        idx_z = (i - 1) * 3 + 1
        idx_y = (i - 1) * 3 + 2
        idx_x = (i - 1) * 3 + 3
        
        # Create unique symbolic weights and biases
        axes = ["z", "y", "x"]
        indices = [idx_z, idx_y, idx_x]
        
        for (axis, idx) in zip(axes, indices)
            w_key = "w_q$(qubit_idx)_l$(layer_idx)_r$(i)_$(axis)"
            θ_key = "θ_q$(qubit_idx)_l$(layer_idx)_r$(i)_$(axis)"
            
            w_params[w_key] = symbols(w_key)
            θ_params[θ_key] = symbols(θ_key)
            
            push!(w_keys, w_key)
            push!(θ_keys, θ_key)
            
            # Define rotation angle
            angle = (idx > d_features ? 0 : x_vars[idx]) * w_params[w_key] + θ_params[θ_key]
            
            # Add appropriate rotation gate
            if axis == "z"
                push!(gates, Rz(angle))
            elseif axis == "y"
                push!(gates, Ry(angle))
            else
                push!(gates, Rx(angle))
            end
        end
    end
    
    return chain(gates...), w_params, θ_params, x_vars, w_keys, θ_keys
end


struct ReuploadBuilderOpts
    n_qubits::Int
    feature_dim::Int
    n_layers::Int
    # in reupload.jl
    entanglement::EntanglementBlock
end

# Build circuit with structured parameters
function build_reuploading_circuit(params::ReuploadBuilderOpts)
    n_qubits = params.n_qubits
    n_layers = params.n_layers 
    d_features = params.feature_dim
    entangler = params.entanglement
    
    full_circuit_gates = []
    all_w = Dict{String, Basic}()
    all_θ = Dict{String, Basic}()
    all_w_keys = String[]
    all_θ_keys = String[]
    x_vars = nothing
    
    for l in 1:n_layers
        qubit_layers = []
        for q in 1:n_qubits
            single_q_layer, w_params, θ_params, x_vars_new, w_keys, θ_keys = create_single_qubit_layer(d_features, l, q)
            push!(qubit_layers, put(n_qubits, q => single_q_layer))
            
            merge!(all_w, w_params)
            merge!(all_θ, θ_params)
            append!(all_w_keys, w_keys)
            append!(all_θ_keys, θ_keys)
            
            if isnothing(x_vars)
                x_vars = x_vars_new
            end
        end
        
        data_layer = chain(qubit_layers)
        push!(full_circuit_gates, data_layer)
        
        if l < n_layers
            entanglement_block = create_entanglement_block(n_qubits, entangler)
            if !isempty(entanglement_block)
                push!(full_circuit_gates, entanglement_block)
            end
        end
    end
    
    circuit = chain(full_circuit_gates...)
    params = CircuitParameters(all_w, all_θ, x_vars, all_w_keys, all_θ_keys)
    
    return circuit, params
end


# Parameter storage structure
struct CircuitParameters
    w::Dict{String, Basic}  # weights
    θ::Dict{String, Basic}  # biases
    x::Vector{Basic}        # input features
    w_keys::Vector{String}  # ordered keys for weights
    θ_keys::Vector{String}  # ordered keys for biases
end


# Helper functions for parameter management
function set_weights!(params::CircuitParameters, values::Vector{<:Real})
    @assert length(values) == length(params.w_keys) "Number of values must match number of weights"
    return Dict(params.w_keys[i] => values[i] for i in 1:length(values))
end

function set_biases!(params::CircuitParameters, values::Vector{<:Real})
    @assert length(values) == length(params.θ_keys) "Number of values must match number of biases"
    return Dict(params.θ_keys[i] => values[i] for i in 1:length(values))
end

function set_inputs!(params::CircuitParameters, x_values::Vector{<:Real})
    @assert length(x_values) == length(params.x) "Input dimension mismatch"
    return Dict("x_$i" => x_values[i] for i in 1:length(x_values))
end

# Combine all parameter values for circuit evaluation
function get_all_parameter_values(params::CircuitParameters, w_values::Vector{<:Real}, 
                                 θ_values::Vector{<:Real}, x_values::Vector{<:Real})
    param_dict = Dict{String, Real}()
    
    # Add weights
    for (i, key) in enumerate(params.w_keys)
        param_dict[key] = w_values[i]
    end
    
    # Add biases
    for (i, key) in enumerate(params.θ_keys)
        param_dict[key] = θ_values[i]
    end
    
    # Add inputs
    for i in 1:length(x_values)
        param_dict["x_$i"] = x_values[i]
    end
    
    return param_dict
end

# Count parameters by type
function count_parameters(params::CircuitParameters)
    return (weights=length(params.w), biases=length(params.θ), inputs=length(params.x))
end

# Get parameter shapes for initialization
function get_parameter_shapes(params::CircuitParameters)
    return (
        n_weights = length(params.w_keys),
        n_biases = length(params.θ_keys),
        n_inputs = length(params.x)
    )
end

# Example usage for gradient visualization
function evaluate_circuit_at_point(circuit, params::CircuitParameters, 
                                 w_values::Vector{<:Real}, θ_values::Vector{<:Real}, 
                                 x_values::Vector{<:Real})

    # Convert symbolic circuit to numeric with specific parameter values
    param_values = get_all_parameter_values(params, w_values, θ_values, x_values)
    
    # You would need to implement substitution of symbolic values
    # This is a placeholder showing the interface
    # numeric_circuit = substitute_parameters(circuit, param_values)
    
    return param_values  # Return for demonstration
end

# Initialize random parameters
function random_parameters(params::CircuitParameters; seed=42)
    shapes = get_parameter_shapes(params)
    
    Random.seed!(seed)
    w_init = -pi .+ pi .* randn(shapes.n_weights)
    θ_init = -pi .+ pi .* randn(shapes.n_weights)
    
    return w_init, θ_init
end
