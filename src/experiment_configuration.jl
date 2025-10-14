using DrWatson
using Random
using PythonCall
using OptimizationOptimJL

using StructTypes

# Deterministic configuration of experiments and data generation
# Instead of regeneration, reload
# Fully typed
# NO STRANGE DICTS -- All data are objects with fields

# ============================================================================
# Base Configuration Types
# ============================================================================

"""
Abstract type for all kernel hyperparameter configurations.
"""
abstract type KernelHyperparameterSearchConfig end

"""
Abstract type for all kernel hyperparameter selection.
"""
abstract type KernelHyperparameters end

"""
Abstract type for all data configurations.
"""
abstract type DataParams end

# ============================================================================
# Data Configuration
# ============================================================================

@kwdef struct DataConfig{P<:DataParams}
    # Size parameters
    n_samples::Int = 10000
    n_features::Int = 2
    test_size::Float64 = 0.2
    
    # Data generation specific
    data_params::P
    
    # Reproducibility
    seed::Int = 42
    
    # File path (set after generation)
    # trying to work without this, not sure it's needed
    # data_path::String = ""
end

DrWatson.default_prefix(c::DataConfig) = savename(c.data_params)

@kwdef struct RBFDataParams <: DataParams
    dataset_type::String="rbf"
    gamma::Float64
    n_support_vectors::Int
    alpha_range::Tuple{Float64, Float64} = (0.1, 2.0)
    bias_range::Tuple{Float64, Float64} = (-1.0, 1.0)
    feature_range::Tuple{Float64, Float64} = (0.0, 2π)
end


DrWatson.allaccess(c::RBFDataParams) = (:gamma, :n_support_vectors, :alpha_range, :bias_range, :feature_range )

@kwdef struct QuantumPauliDataParams <: DataParams
    dataset_type::String="pauli"
    n_qubits::Int
    paulis::Vector{String}
    reps::Int
    entanglement::String = "full"
    gap::Float64 = 0.3
    grid_points_per_dim::Int = 20
end

DrWatson.allaccess(c::QuantumPauliDataParams) = (:n_qubits, :paulis, :reps, :entanglement, :gap, :grid_points_per_dim)
DrWatson.default_prefix(c::QuantumPauliDataParams) = join(["pauli_data", join(c.paulis, "-")], "_")

# ============================================================================
# RBF Kernel Configuration
# ============================================================================

@kwdef struct RBFKernelHyperparameterSearchConfig <: KernelHyperparameterSearchConfig
    kernel_type::Symbol= :rbf
    
    # If "auto", will use cross-validation to find best gamma
    gamma_range::Vector{Float64} = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
end


"""
    Best Hyperparameters found during tuning  

    These will be stored/passed on to SVM training
"""
@kwdef struct RBFHyperparameters <: KernelHyperparameters
    gamma::Float64
    C::Float64

    # Reproducibility
    seed::Int = 42
end

# ============================================================================
# Reuploading Quantum Kernel Configuration
# ============================================================================

@kwdef struct ReuploadingKernelHyperparameterSearchConfig <: KernelHyperparameterSearchConfig
    kernel_type::Symbol= :reuploading
    
    # Circuit architecture
    n_qubits::Int
    n_features::Int = 2
    n_layers::Int
    entanglement::String = "linear"
    
    # Training configuration
    iterations::Vector{Int} = [5, 5, 10]
        
    # Optimization parameter grid search
    learning_rate::Float64 = 0.01
    
    # Memory management
    memory_budget_gb::Float64 = 8.0
    
    # Reproducibility
    seed::Int
end

@kwdef struct ReuploadingKernelHyperparameters <: KernelHyperparameters
    # final angles found by optimization 
    nqubits::Int
    nlayers::Int
    thetas::Vector{Float64}
    biases::Vector{Float64}
    loss::Vector{Float64}
    C::Float64
end
    
# ============================================================================
# Pauli Kernel Configuration with Search
# ============================================================================

@kwdef struct PauliSearchConstraints
    base_paulis::Vector{String} = ["X", "Y", "Z"]
    max_pauli_order::Int = 2
    min_num_terms::Int = 1
    max_num_terms::Int = 3
end

@kwdef struct PauliKernelHyperparameterSearchConfig <: KernelHyperparameterSearchConfig
    kernel_type::Symbol = :pauli
    
    # Circuit parameters
    n_qubits::Int = 2
    reps::Vector{Int} = [1, 2, 3]
    entanglement::Vector{String} = ["full", "linear", "circular"]
    
    # Search configuration
    search_strategy::String = "random"  
    n_search_iterations::Int = 5
    search_constraints::PauliSearchConstraints = PauliSearchConstraints()
          
    # Reproducibility
    seed::Int = 42
end

"""
    Best Hyperparameters found during tuning  

    These will be stored/passed on to SVM training
"""
@kwdef struct PauliKernelHyperparameters <: KernelHyperparameters

    # Circuit parameters
    n_qubits::Int = 2
    reps::Int
    entanglement::String
    paulis::Vector{String}
    C::Float64
             
    # Reproducibility
    seed::Int = 42
end

"""
Generate a random valid Pauli string set respecting constraints
"""
function generate_constrained_pauli_set(
    constraints::PauliSearchConstraints;
    seed::Union{Int, Nothing} = nothing
)
    !isnothing(seed) && Random.seed!(seed)
    
    base_paulis = constraints.base_paulis
    max_order = constraints.max_pauli_order
    max_terms = constraints.max_num_terms
    min_terms = constraints.min_num_terms
    
    # Ensure we can have entanglement
    if max_order < 2
        error("max_pauli_order must be >= 2 for entanglement")
    end
    
    pauli_set = Set{String}()
    
    # Add one mandatory entangling term
    order = rand(2:max_order)
    mandatory_term = join(rand(base_paulis, order))
    push!(pauli_set, mandatory_term)
    
    # Add remaining terms
    num_terms = rand(min_terms:max_terms)
    while length(pauli_set) < num_terms
        order = rand(1:max_order)
        new_term = join(rand(base_paulis, order))
        push!(pauli_set, new_term)
    end
    
    # Return as shuffled vector
    result = collect(pauli_set)
    shuffle!(result)
    return result
end

# ============================================================================
# Master Experiment Configuration
# ============================================================================

@kwdef struct ExperimentConfig{P<:DataParams, K<:KernelHyperparameterSearchConfig}
    experiment_name::String = nothing
    
    # Data configuration
    data_config::DataConfig{P}
    
    # Model configuration (this model learns the data) 
    kernel_config::K
    
    # Learning curve configuration
    learning_curve_sizes::Vector{Int} = collect(100:100:data_config.n_samples)

    # ranges for the value of C in SVM computation
    c_ranges::Vector{Float64} = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_folds::Int = 5
    
    # Experiment tracking
    seed::Int = 42
end

DrWatson.default_prefix(c::ExperimentConfig) = join(savename(c.data_config), savename(c.kernel_config), "_")

"""
Convert experiment config to DrWatson-compatible dict
"""
function config_to_dict(config::ExperimentConfig)
    return @dict(
        experiment_name = config.experiment_name,
        description = config.description,
        n_samples = config.data_config.n_samples,
        n_features = config.data_config.n_features,
        dataset_type = config.data_config.dataset_type,
        kernel_type = config.kernel_config.kernel_type,
        seed = config.seed,
        created_at = config.created_at
    )
end

"""
Constructor to create an ExperimentConfig from a flat dictionary,
typically generated from a DataFrame row.
"""
function ExperimentConfig(d::Dict{Symbol, Any})
    # --- Reconstruct DataConfig ---
    dataset_type = d[:dataset_type]
    n_features = d[:n_features]
    seed = d[:seed]
    
    local data_params, data_name
    if dataset_type == "rbf"
        gamma = d[:rbf_gamma]
        n_sv = d[:n_support_vectors]
        data_params = Dict(:gamma => gamma, :n_support_vectors => n_sv)
        data_name = "rbf_g$(gamma)_sv$(n_sv)"
    else # quantum_pauli
        paulis = split(d[:quantum_paulis], "-")
        reps = d[:quantum_reps]
        data_params = Dict(:n_qubits => n_features, :paulis => paulis, :reps => reps)
        data_name = "quantum_p$(d[:quantum_paulis])_r$(reps)"
    end

    data_config = DataConfig(
        dataset_type = dataset_type,
        dataset_name = data_name,
        n_samples = d[:n_samples],
        n_features = n_features,
        data_params = data_params,
        seed = seed
    )

    # --- Reconstruct KernelConfig Array ---
    kernel_type = Symbol(d[:kernel_type])
    
    kernel_configs = KernelHyperparameterSearchConfig[]

    kernel_type = if kernel_type == :rbf
        RBFKernelHyperparameterSearchConfig(gamma="auto")
    elseif kernel_type == :reuploading
        ReuploadingKernelHyperparameterSearchConfig(
            n_qubits = n_features, n_layers = 2, entanglement = "linear",
            max_iterations = 50, seed = seed
        )
    elseif kernel_type == :pauli
        PauliKernelHyperparameterSearchConfig(
            n_qubits = n_features, reps = 2, n_search_iterations = 20, seed = seed
        )
    end

    # --- Construct the final ExperimentConfig ---
    return ExperimentConfig(
        experiment_name = d[:experiment_name],
        data_config = data_config,
        kernel_configs = kernel_configs,
        learning_curve_sizes = d[:learning_curve_sizes],
        seed = seed
    )
end

# ============================================================================
# Example Usage
# ============================================================================


# Define how to handle the abstract type
StructTypes.StructType(::Type{ExperimentConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{KernelHyperparameterSearchConfig}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{KernelHyperparameterSearchConfig}) = :kernel_type
StructTypes.subtypes(::Type{KernelHyperparameterSearchConfig}) = (
    rbf = RBFKernelHyperparameterSearchConfig,
    reuploading = ReuploadingKernelHyperparameterSearchConfig,
    pauli = PauliKernelHyperparameterSearchConfig
)
