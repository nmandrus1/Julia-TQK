using DrWatson
using Random
using StructTypes

# ==========================================
# 1. Data Configuration (Reproducibility Source)
# ==========================================
abstract type AbstractDataParams end

@kwdef struct DataConfig{P<:AbstractDataParams}
    dataset_name::String
    n_samples::Int = 100
    n_features::Int = 2
    test_size::Float64 = 0.2
    seed::Int = 42
    params::P
end

# Specific Data Generators
# @kwdef struct TeacherStudentParams <: AbstractDataParams
#     teacher_type::String = "efficiency_su2"
#     depth::Int = 2
# end

# @kwdef struct SyntheticDataParams <: AbstractDataParams
#     type::String = "moons" # or circles, blobs
#     noise::Float64 = 0.1
# end

# DrWatson helpers for smart naming
DrWatson.default_prefix(c::DataConfig) = c.dataset_name

# ==========================================
# 2. Kernel Blueprints ( The "Method" )
# The method defines the search strategy and space
# ==========================================
abstract type AbstractKernelMethod end

# RBF Blueprint: "I will search this gamma grid"
@kwdef struct RBFMethod <: AbstractKernelMethod
    name::String = "rbf"
    gamma_grid::Vector{Float64} = [0.01, 0.1, 1.0, 10.0]
end

@kwdef struct PauliSearchConstraints
    base_paulis::Vector{String} = ["X", "Y", "Z"]
    max_pauli_order::Int = 2
    min_num_terms::Int = 1
    max_num_terms::Int = 3
end

# Pauli Blueprint: "I will random search within these constraints"
@kwdef struct PauliMethod <: AbstractKernelMethod
    name::String = "pauli"
    n_qubits::Int
    n_reps::Int = 2
    search_iterations::Int = 20
    constraints::PauliSearchConstraints = PauliSearchConstraints()
end

# Reuploading Blueprint: "I will gradient descend this architecture"
@kwdef struct ReuploadingMethod <: AbstractKernelMethod
    name::String = "reuploading"
    # The architecture IS the search space definition
    circuit_config::ReuploadingConfig 
    optimizer::SPSAConfig = SPSAConfig()
    initial_params_seed::Int = 42
end

# ==========================================
# 3. Trained Kernel Instances ( The "Result" )
# ==========================================
abstract type AbstractTrainedKernel end

# RBF Instance: Knows its Gamma
struct TrainedRBFKernel <: AbstractTrainedKernel
    gamma::Float64
end

# Pauli Instance: Knows its structure
struct TrainedPauliKernel <: AbstractTrainedKernel
    paulis::Vector{String}
end

# Reuploading Instance: Knows its Architecture + Weights
struct TrainedReuploadingKernel <: AbstractTrainedKernel
    config::ReuploadingConfig
    params::Vector{Float64}
end

# After training a kernel we train the SVM 
struct TrainedSVM{K <: AbstractTrainedKernel}
    K
    C::Float64
end

# ==========================================
# 4. Master Experiment Config
# ==========================================
@kwdef struct ExperimentConfig{M<:AbstractKernelMethod}
    # Identity
    name::String
    
    # Components
    data::DataConfig
    method::M
    
    # Global SVM Settings (The "Fairness" Guarantee)
    # All methods share these exact settings for the final step
    c_grid::Vector{Float64} = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_folds::Int = 5
end

DrWatson.default_prefix(c::ExperimentConfig) = "exp_$(c.name)"
