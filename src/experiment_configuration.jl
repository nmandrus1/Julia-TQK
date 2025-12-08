# src/experiment_configuration.jl
using DrWatson
using StableRNGs
using Random
using Parameters # for @kwdef

# ==========================================
# 0. Reproducibility Core
# ==========================================

"""
    derive_rng(source_seed::Int, step_salt::Int) -> StableRNG

Creates a new, independent StableRNG for a specific pipeline step.
The 'step_salt' ensures that the Data Generation RNG is different from the Tuning RNG,
even if they share the master seed.
"""
function derive_rng(source_seed::UInt, step_salt::Int)
    # We use a simple hashing strategy to mix the master seed with the step salt
    # This prevents overlap between streams.
    derived_seed = hash(step_salt, source_seed)
    return StableRNG(derived_seed)
end

# defined "Salts" for different pipeline stages to avoid magic numbers
const SALT_DATA_GEN::Int = 1001
const SALT_TUNING::Int   = 2002
const SALT_SVM_CV::Int   = 3003

# ==========================================
# 1. Data Configuration
# ==========================================
abstract type AbstractDataParams end

"""
    DataConfig

Configuration for the 'Teacher' dataset generation.
"""
@kwdef struct DataConfig{P<:AbstractDataParams}
    dataset_name::String
    n_samples::Int = 100
    n_features::Int = 2
    test_size::Float64 = 0.2
    seed::Int = 42  # This seed is specific to the DATA GENERATION step
    params::P
end

DrWatson.default_prefix(c::DataConfig) = c.dataset_name

# --- Data Parameter Structs ---

@kwdef struct RBFDataParams <: AbstractDataParams
    gamma::Float64 = 1.0
    n_support_vectors::Int = 20
    feature_range::Tuple{Float64, Float64} = (0.0, 2π)
    alpha_range::Tuple{Float64, Float64} = (0.1, 2.0)
end

@kwdef struct QuantumPauliDataParams <: AbstractDataParams
    n_qubits::Int
    paulis::Vector{String}
    reps::Int = 2
    entanglement::String = "linear"
    gap::Float64 = 0.1
    grid_points_per_dim::Int = 20
end

@kwdef struct ReuploadingDataParams <: AbstractDataParams
    n_qubits::Int
    n_features::Int
    n_layers::Int
    entanglement::String = "linear"
    n_support_vectors::Int = 20
    alpha_range::Tuple{Float64, Float64} = (0.1, 2.0)
end


# ==========================================
# 2. Kernel Method Blueprints (The "Student")
# ==========================================
abstract type AbstractKernelMethod end

# --- RBF Method ---
@kwdef struct RBFMethod <: AbstractKernelMethod
    name::String = "rbf"
    gamma_grid::Vector{Float64} = [0.01, 0.1, 1.0, 10.0, 100.0]
end

# --- Pauli Method ---
@kwdef struct PauliSearchConstraints
    base_paulis::Vector{String} = ["I", "X", "Y", "Z"]
    max_pauli_order::Int = 2
    min_num_terms::Int = 1
    max_num_terms::Int = 5
end

@kwdef struct PauliMethod <: AbstractKernelMethod
    name::String = "pauli"
    n_features::Int
    n_reps::Int = 2
    search_iterations::Int = 20
    ent::EntanglementStrategy = LinearEntanglement
    constraints::PauliSearchConstraints = PauliSearchConstraints()
end

# --- Reuploading Method ---
@kwdef struct ReuploadingMethod <: AbstractKernelMethod
    name::String = "reuploading"
    circuit_config::ReuploadingConfig # Defined in src/feature_maps/types.jl
    optimizer::SPSAConfig = SPSAConfig(seed=0) # Seed will be overwritten by pipeline
end


# ==========================================
# 3. Trained Kernel Artifacts (The "Learned Knowledge")
# ==========================================

"""
Abstract parent for all trained kernels. 
Any subtype T must support a dispatch:
    compute_kernel_matrix(kernel::T, X)
"""
abstract type AbstractTrainedKernel end

"""
    TrainedRBFKernel

The simplest artifact. It just remembers the optimal Gamma.
"""
struct TrainedRBFKernel <: AbstractTrainedKernel
    gamma::Float64
end

"""
    TrainedPauliKernel

Stores the specific Pauli strings found during the random search.
Constraints are no longer needed here; just the result.
"""
struct TrainedPauliKernel <: AbstractTrainedKernel
    paulis::Vector{String}
    n_qubits::Int
end

"""
    TrainedReuploadingKernel

Crucial: Stores BOTH the learned parameters (thetas) AND the 
circuit configuration (layers, qubits, feature map type) required 
to reconstruct the circuit.
"""
struct TrainedReuploadingKernel <: AbstractTrainedKernel
    config::ReuploadingConfig  # The architecture (Teacher or Student)
    params::Vector{Float64}    # The learned angles
end

# We don't strictly need a generic "TrainedKernel" struct because 
# TuningResult{T} handles the wrapping of these specific types.
  
# ==========================================
# 4. Master Experiment Config
# ==========================================

@kwdef struct ExperimentConfig{M<:AbstractKernelMethod}
    # Identity
    name::String
    
    # The Global Random Seed
    # All sub-RNGs are derived from this single integer.
    master_seed::UInt64 = 42
    
    # Components
    data_config::DataConfig
    method::M
    
    # Global Tuning Settings
    # Passed to the SVM stage
    c_grid::Vector{Float64} = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_folds::Int = 5
    
    # Performance tuning for the experiment itself
    tuning_batch_size::Int = 0 # 0 means full dataset
end

DrWatson.default_prefix(c::ExperimentConfig) = "exp_$(c.name)"

# ==========================================
# 5. Artifact Container (Save Everything)
# ==========================================

struct ExperimentArtifacts
    config::ExperimentConfig
    
    # Method Tuning Results
    tuning_result::Any # Holds TuningResult from interface.jl
    
    # Final Model Analysis
    kernel_matrix_train::Matrix{Float64}
    kernel_matrix_test::Matrix{Float64}
    
    # SVM Results
    best_C::Float64
    svm_cv_acc::Float64
    train_acc::Float64
    test_acc::Float64
    
    # Metadata
    completed_at::String
end
