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
include("workspaces/interface.jl")

include("feature_maps/interface.jl")
include("feature_maps/reupload.jl")

include("kernels/fidelity.jl")

include("workspaces/dynamic.jl")
include("workspaces/preallocated.jl")

include("trainer/quantum_kernel_trainer.jl")

# experiment configuration (many types declared here)
include("experiment_configuration.jl")

include("qiskit/qiskit_interface.jl")

# data generation
include("data_generation/quantum_data.jl")
include("data_generation/rbf_data.jl")
include("data_generation/load_data.jl")


# Export types
export AbstractQuantumFeatureMap, ReuploadingCircuit, FidelityKernel
export EntanglementBlock, linear, alternating, all_to_all
export AbstractFidelityWorkspace, PreallocatedWorkspace, DynamicWorkspace
export QuantumKernelTrainer

# Qiskit
export PauliFeatureMapConfig

# Config
export ExperimentConfig, DataConfig, KernelHyperparameterSearchConfig
export RBFKernelHyperparameterSearchConfig, ReuploadingKernelHyperparameterSearchConfig
export PauliKernelHyperparameterSearchConfig, DataParams, QuantumPauliDataParams, RBFDataParams

# Results
export ReuploadingKernelHyperparameters, PauliKernelHyperparameters, KernelHyperparameters, RBFKernelHyperparameters
export KernelResults

# Export functions for feature maps
export n_qubits, n_features, n_params, map_inputs!
export assign_params!, assign_random_params!, get_params

# Export functions for kernels
export evaluate, evaluate!, evaluate_symmetric_cached!, compute_kernel_value_cached, evaluate_asymmetric_cached!, calculate_tile_size

# Export workspace functions
export create_preallocated_workspace
export extract_gradients, get_grad_buffers!

# gradients
export loss_gradient

# Export utility functions
export compute_angles!

# optimization
export create_optimization_function, train!
# basic loss functions
export kernel_target_alignment

# Pauli Feature Map Functions
export create_pauli_feature_map, compute_pauli_kernel_matrix

# data generation
export generate_pauli_expectation_data_grid, generate_pseudo_svm_dataset, prepare_data!, produce_data


end
