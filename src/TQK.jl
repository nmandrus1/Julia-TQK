module TQK

# Core types and utilities
include("types.jl")

# Kernel implementations
include("kernels/fidelity.jl")

# Feature map implementations
include("feature_maps/reupload.jl")

# Core type exports
export CircuitParameters, random_parameters, get_parameter_shapes

# Kernel exports
export FidelityKernel, evaluate, clear_cache!

# Feature map exports
export ReuploadingCircuit, assign_random_params, assign_params!
export create_entanglement_block, EntanglementBlock
export linear, alternating, all_to_all


end # module
