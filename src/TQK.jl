module TQK

using Yao
using YaoBlocks
using Yao.AD
using LinearAlgebra
using ArgCheck
using Random
using Base.Threads

# Include implementation files

include("feature_maps/types.jl")
include("feature_maps/reupload.jl")

include("trainer/quantum_kernel_trainer.jl")

# experiment configuration (many types declared here)
include("experiment_configuration.jl")

include("qiskit/qiskit_interface.jl")
include("qiskit/pauli_search.jl")
include("rbf/rbf_search.jl")
include("reupload_hyperparam_search.jl")

# data generation
include("data_generation/quantum_data.jl")
include("data_generation/rbf_data.jl")
include("data_generation/load_data.jl")
include("data_generation/reupload_data.jl")


# Export types
export AbstractQuantumFeatureMapConfig, ReuploadingConfig
export EntanglementStrategy, LinearEntanglement, CircularEntanglement, FullEntanglement
export QuantumKernelTrainer

# Qiskit
export PauliFeatureMapConfig, PauliSearchConstraints

# Config
export ExperimentConfig, DataConfig, KernelHyperparameterSearchConfig
export RBFKernelHyperparameterSearchConfig, ReuploadingKernelHyperparameterSearchConfig
export PauliKernelHyperparameterSearchConfig, DataParams, QuantumPauliDataParams, RBFDataParams, ReuploadingDataParams

# Results
export ReuploadingKernelHyperparameters, PauliKernelHyperparameters, KernelHyperparameters, RBFHyperparameters
export KernelResults

# Export functions for feature maps
export n_qubits, n_trainable_params, build_circuit

# Export functions for kernels
export evaluate, evaluate!

# Export workspace functions
export create_preallocated_workspace
export extract_gradients, get_grad_buffers!

# optimization
export create_optimization_function, train!

# Pauli Feature Map Functions
export create_pauli_feature_map, compute_pauli_kernel_matrix

# data generation
export generate_pauli_expectation_data_grid, generate_pseudo_svm_dataset, prepare_data!, produce_data, generate_reupload_data

export search_pauli_hyperparameters, tune_svm_C
export search_rbf_hyperparameters 
export evaluate_pauli_config_kta, evaluate_pauli_config_cv       
export evaluate_rbf_kta, evaluate_rbf_cv, tune_rbf_C, pairwise_distances_colmajor

export train_reuploading_kernel_with_cv

end
