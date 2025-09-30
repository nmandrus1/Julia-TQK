# src/quantum_baselines/pauli_feature_map.jl
using PyCall
using Parameters

"""
Configuration for PauliFeatureMap from Qiskit.

Pauli strings can be:
- Single qubit: "I", "X", "Y", "Z"
- Two qubit: "XX", "YY", "ZZ", "XY", "YZ", "XZ"
- Three qubit: "XXX", "XYZ", "ZZZ", etc.
"""
@with_kw struct PauliFeatureMapConfig
    # Core parameters
    n_features::Int              # Input feature dimension
    n_qubits::Int = n_features   # Default: one qubit per feature
    reps::Int = 2                 # Number of repetitions
    
    # Pauli strings (e.g., ["Z", "YY", "ZXZ"])
    paulis::Vector{String} = ["Z", "ZZ"]
    
    # Entanglement structure
    entanglement::String = "full"  # "full", "linear", "circular", or "sca"
    
    # Additional parameters
    alpha::Float64 = 2.0          # Scaling factor
    insert_barriers::Bool = false
    parameter_prefix::String = "x"
    
    # Optional: custom entanglement map (for advanced use)
    entanglement_blocks::Union{Nothing, Vector{Vector{Int}}} = nothing
end
