using Base.Threads
using SymEngine
using Random

"""
    CircuitParameters

Struct used to store the parameters of a circuit. These can be assigned to
a circuit. Contains symbolic representations of weights, biases, and input features.
"""
struct CircuitParameters
    w::Dict{String, Basic}  # weights
    θ::Dict{String, Basic}  # biases
    x::Vector{Basic}        # input features
    w_keys::Vector{String}  # ordered keys for weights
    θ_keys::Vector{String}  # ordered keys for biases
end

"""
    random_parameters(params::CircuitParameters; seed=42)

Assign the parameters of a circuit to random floats uniformly distributed
between -π and π.

# Arguments
- `params`: CircuitParameters struct containing parameter structure
- `seed`: Random seed for reproducibility (default: 42)

# Returns
- `w_init`: Vector of initialized weights
- `θ_init`: Vector of initialized biases
"""
function random_parameters(params::CircuitParameters; seed=42)
    shapes = get_parameter_shapes(params)
    
    Random.seed!(seed)
    w_init = -pi .+ 2pi .* rand(shapes.n_weights)
    θ_init = -pi .+ 2pi .* rand(shapes.n_biases)
    
    return w_init, θ_init
end

"""
    get_parameter_shapes(params::CircuitParameters)

Get the number of parameters of each type.

# Returns
Named tuple with fields:
- `n_weights`: Number of weight parameters
- `n_biases`: Number of bias parameters  
- `n_inputs`: Number of input features
"""
function get_parameter_shapes(params::CircuitParameters)
    return (
        n_weights = length(params.w_keys),
        n_biases = length(params.θ_keys),
        n_inputs = length(params.x)
    )
end
