module TQK

using Yao
using YaoBlocks
using Yao.AD
using LinearAlgebra
using ArgCheck
using LRUCache
using ProgressMeter
using Random
using Base.Threads

# Include implementation files
include("feature_maps/interface.jl")
include("feature_maps/reupload.jl")
include("kernels/fidelity.jl")

# Export types
export AbstractQuantumFeatureMap, ReuploadingCircuit, FidelityKernel
export EntanglementBlock, linear, alternating, all_to_all

# Export functions for feature maps
export n_qubits, n_features, n_params, map_inputs!
export assign_params!, assign_random_params!, get_params

# Export functions for kernels
export evaluate, evaluate!, evaluate_symmetric_cached!, compute_kernel_value_cached, evaluate_asymmetric_cached!, calculate_tile_size

# gradients
export loss_gradient

# Export utility functions
export compute_angles!

end
