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

# SPSA
include("optimizers/spsa.jl")

include("feature_maps/reupload.jl")
include("feature_maps/pauli.jl")

# experiment configuration (many types declared here)
include("experiment_configuration.jl")

include("tuning/interface.jl")
include("tuning/rbf.jl")
include("tuning/pauli.jl")
include("tuning/reuploading.jl")
include("tuning/svm.jl")

include("kernels/pure_fidelity.jl")
include("kernels/compute_uncompute.jl")

include("utils/mock_hardware.jl")


# include("qiskit/qiskit_interface.jl")
# include("qiskit/pauli_search.jl")
# include("rbf/rbf_search.jl")
# include("reupload_hyperparam_search.jl")

# # data generation
include("data_generation/quantum_data.jl")
include("data_generation/rbf_data.jl")
include("data_generation/load_data.jl")
include("data_generation/reupload_data.jl")


# Export types
export AbstractFeatureMapConfig
export ReuploadingConfig, PauliConfig
export EntanglementStrategy, LinearEntanglement, CircularEntanglement, FullEntanglement

# Optimization 
export SPSAConfig

# Qiskit
export PauliSearchConstraints

# Config
export AbstractDataParams, AbstractKernelMethod, AbstractTrainedKernel 
export DataConfig, RBFDataParams, QuantumPauliDataParams, ReuploadingDataParams
export RBFMethod, PauliMethod, ReuploadingMethod
export TrainedRBFKernel, TrainedPauliKernel, TrainedReuploadingKernel
export ExperimentConfig

# Export functions for feature maps
export n_qubits, n_trainable_params, build_circuit

# kernel matrix computatoins
export compute_statevectors
export compute_kernel_matrix_hardware, compute_kernel_matrix_pure, compute_fidelity_hardware_compatible
export rbf_kernel_matrix

export compute_final_matrix

# loss
export kernel_target_alignment
export hardware_compatible_loss, variational_kta_loss

# optimization
export optimize_spsa

export tune_kernel, tune_svm_c

# experiment running
export derive_rng
export produce_data

# Pauli Feature Map Functions
# export create_pauli_feature_map, compute_pauli_kernel_matrix

# # data generation
# export generate_pauli_expectation_data_grid, generate_pseudo_svm_dataset, prepare_data!, produce_data, generate_reupload_data

# export search_pauli_hyperparameters, tune_svm_C
# export search_rbf_hyperparameters 
# export evaluate_pauli_config_kta, evaluate_pauli_config_cv       
# export evaluate_rbf_kta, evaluate_rbf_cv, tune_rbf_C, pairwise_distances_colmajor

# export train_reuploading_kernel_with_cv

end
