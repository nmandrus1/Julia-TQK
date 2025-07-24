"""
Abstract type for quantum feature maps used in kernel methods.
All concrete feature maps should implement the required interface methods.
"""
abstract type AbstractQuantumFeatureMap end

# Required interface methods (these define the "contract")
"""
    n_qubits(fm::AbstractQuantumFeatureMap) -> Int
    
Returns the number of qubits in the feature map.
"""
function n_qubits(fm::AbstractQuantumFeatureMap)
    error("n_qubits not implemented for $(typeof(fm))")
end

"""
    n_features(fm::AbstractQuantumFeatureMap) -> Int
    
Returns the number of input features expected by the feature map.
"""
function n_features(fm::AbstractQuantumFeatureMap)
    error("n_features not implemented for $(typeof(fm))")
end

"""
    n_parameters(fm::AbstractQuantumFeatureMap) -> Int
    
Returns the total number of trainable parameters in the feature map.
"""
function n_params(fm::AbstractQuantumFeatureMap)
    error("n_parameters not implemented for $(typeof(fm))")
end

"""
    map_inputs!(fm::AbstractQuantumFeatureMap, x::Vector{Float64}) -> ChainBlock
    
Builds and modifies the quantum circuit for input features x.
The circuit should have all parameters set according to the current state of fm.
"""
function map_inputs!(fm::AbstractQuantumFeatureMap, x::AbstractVector)
    error("map_inputs! not implemented for $(typeof(fm))")
end

"""
    assign_params!(fm::AbstractQuantumFeatureMap, params...)
    
Updates the parameters of the feature map in-place.
The specific arguments depend on the concrete implementation.
"""
function assign_params!(fm::AbstractQuantumFeatureMap, params...)
    error("assign_params! not implemented for $(typeof(fm))")
end

# Optional interface methods with default implementations
"""
    get_params(fm::AbstractQuantumFeatureMap) -> Tuple
    
Returns the current parameters as a tuple.
Default implementation returns empty tuple.
"""
function get_params(fm::AbstractQuantumFeatureMap)
    return ()
end

"""
    assign_random_params!(fm::AbstractQuantumFeatureMap, range::Tuple{Float64, Float64} = (-π, π); seed = nothing)
    
Assigns random parameters to the feature map.
Default implementation does nothing (for parameter-free feature maps).
"""
function assign_random_params!(fm::AbstractQuantumFeatureMap, range::Tuple{<:Real, <:Real} = (-π, π); seed = nothing)
    return fm
end
