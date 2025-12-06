using YaoAPI

abstract type AbstractFeatureMapConfig end

"""
    n_qubits(config::AbstractFeatureMapConfig) -> Int
Returns the number of qubits.
"""
function n_qubits(config::AbstractFeatureMapConfig)::Int
    error("Not implemented")
end

"""
    n_trainable_params(config::AbstractFeatureMapConfig) -> Int
Returns the total number of trainable parameters (weights + biases) required.
"""
function n_trainable_params(config::AbstractFeatureMapConfig)::Int
    error("Not implemented")
end

"""
    build_circuit(config::AbstractFeatureMapConfig, params::AbstractVector, x::AbstractVector) -> AbstractBlock
Factory function that creates a fresh Yao block (circuit) for a specific data point and parameter set.
"""
function build_circuit(config::AbstractFeatureMapConfig, params::AbstractVector, x::AbstractVector)::AbstractBlock
    error("Not implemented")
end
