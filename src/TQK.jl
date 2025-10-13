module TQK

using Yao
using YaoBlocks
using Yao.AD
using LinearAlgebra
using ArgCheck
using Random
using Base.Threads

# Include implementation files
include("workspaces/interface.jl")

include("feature_maps/interface.jl")
include("feature_maps/reupload.jl")

include("workspaces/preallocated.jl")
include("workspaces/parallel.jl")

include("kernels/fidelity.jl")

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
export AbstractFidelityWorkspace, PreallocatedWorkspace, ThreadAwareWorkspace
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
export evaluate, evaluate!

# Export workspace functions
export create_preallocated_workspace, create_thread_aware_workspace
export extract_gradients, get_grad_buffers!
export get_workspace, get_K_cache, get_grad_buffer, get_feature_map, get_statevectors, get_workspace
# gradients
export loss_gradient_finite_diff

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
