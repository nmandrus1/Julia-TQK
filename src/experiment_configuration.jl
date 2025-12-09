# src/experiment_configuration.jl
using DrWatson
using StableRNGs
using Random
using Parameters # for @kwdef
using Dates

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
const SALT_DATAGEN::Int    = 1001 
const SALT_SAMPLING::Int    = 2002 # When tuning, this RNG controls how batches are sampled 
const SALT_OPTIMIZER::Int   = 3003 # When tuning, this RNG controls changes to the model
const SALT_SVM_CV::Int      = 4004

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
    master_seed::UInt
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
    n_features::Int
    paulis::Vector{String}
    reps::Int = 2
    ent::EntanglementStrategy = LinearEntanglement
    n_support_vectors::Int = 20
    alpha_range::Tuple{Float64, Float64} = (0.1, 2.0)
end

@kwdef struct ReuploadingDataParams <: AbstractDataParams
    n_qubits::Int
    n_features::Int
    n_layers::Int
    ent::EntanglementStrategy = LinearEntanglement
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
    optimizer::SPSAConfig = SPSAConfig() # Seed will be overwritten by pipeline
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
    config::PauliConfig
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

"""
    ExperimentArtifacts

The master container for a single run. 
Designed to be saved as a .jld2 file by DrWatson.
"""
struct ExperimentArtifacts
    # 1. The Recipe (Crucial for reproducibility)
    config::ExperimentConfig
    
    # 2. The Learning Journey
    tuning_result::TuningResult  # Contains best params and loss history
    
    # 3. The Geometry (For your Hypersphere research)
    # We save the kernel matrices to analyze the distribution of dot products
    K_train::Matrix{Float64}
    K_test::Matrix{Float64}
    
    # 4. The Quantum State (Optional/Heavy)
    # Save raw statevectors for manifold learning (PCA/t-SNE on the quantum state)
    # WARNING: Only populated if n_qubits <= 14 to save disk space
    train_statevectors::Union{Matrix{ComplexF64}, Nothing} 
    
    # 5. The Classical Model
    best_C::Float64
    svm_cv_acc::Float64
    train_acc::Float64
    test_acc::Float64
    
    # 6. Metadata
    timestamp::DateTime
    git_commit::String # DrWatson can fetch this automatically
end
