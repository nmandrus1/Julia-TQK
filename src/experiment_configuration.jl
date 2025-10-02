using DrWatson
using Random
using PythonCall
using Dates
using OptimizationOptimJL

using StructTypes

# ============================================================================
# Base Configuration Types
# ============================================================================

"""
Abstract type for all kernel configurations.
Each concrete kernel type must implement:
- produce_kernel(config::KernelConfig, data_config::DataConfig) -> kernel
"""
abstract type KernelConfig end

# ============================================================================
# Data Configuration
# ============================================================================

@kwdef struct DataConfig
    # Dataset identification
    dataset_type::String  # "rbf", "quantum_pauli", "quantum_expectation"
    dataset_name::String = "default"
    
    # Size parameters
    n_samples::Int = 1000
    n_features::Int = 2
    test_size::Float64 = 0.2
    
    # Data generation specific
    data_params::Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    # Reproducibility
    seed::Int = 42
    
    # File path (set after generation)
    data_path::String = ""
end

# ============================================================================
# RBF Kernel Configuration
# ============================================================================

@kwdef struct RBFKernelConfig <: KernelConfig
    kernel_type::Symbol= :rbf
    gamma::Union{Float64, String} = "auto"  # or specific value
    
    # If "auto", will use cross-validation to find best gamma
    cv_folds::Int = 3
    gamma_range::Vector{Float64} = [0.001, 0.01, 0.1, 1.0, 10.0]
end

"""
Produce an RBF kernel (just returns the configuration since sklearn handles it)
"""
function produce_kernel(config::RBFKernelConfig, data_config::DataConfig; n_samples::Union{Int, Nothing}=nothing)
    # For RBF, we don't actually "train" - just return the gamma
    if config.gamma == "auto"
        @info "RBF kernel will use cross-validation to select gamma"
        return Dict(:type => "rbf", :gamma => nothing, :needs_cv => true, 
                   :gamma_range => config.gamma_range, :cv_folds => config.cv_folds)
    else
        @info "RBF kernel using gamma = $(config.gamma)"
        return Dict(:type => "rbf", :gamma => config.gamma, :needs_cv => false)
    end
end

# ============================================================================
# Reuploading Quantum Kernel Configuration
# ============================================================================

#Optimizer configuration? 

@kwdef struct ReuploadingKernelConfig <: KernelConfig
    kernel_type::Symbol= :reuploading
    
    # Circuit architecture
    n_qubits::Int
    n_features::Int
    n_layers::Int
    entanglement::String = "linear"
    
    # Training configuration
    optimizer::String #"Adam" or "LBFGS"
    max_iterations::Int = 100
    loss_function::String = "svm_loss"  # or "svm_loss"
    
    # Optimization parameters
    learning_rate::Float64 = 0.01
    convergence_tol::Float64 = 1e-6
    
    # Memory management
    memory_budget_gb::Float64 = 4.0
    
    # Reproducibility
    seed::Int
    end

"""
Produce a trained reuploading quantum kernel
"""
function produce_kernel(config::ReuploadingKernelConfig, data_config::DataConfig; n_samples::Union{Int, Nothing}=nothing)
    @info "Training Reuploading Quantum Kernel" config.n_qubits config.n_layers config.entanglement
    
    # Load data
    data = load(data_config.data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Use subset if specified
    if !isnothing(n_samples)
        X_train = X_train[1:min(n_samples, size(X_train, 1)), :]
        y_train = y_train[1:min(n_samples, length(y_train))]
        @info "Using data subset" n_samples
    
    # Create feature map
    entanglement_enum = if config.entanglement == "linear"
        TQK.linear
    elseif config.entanglement == "alternating"
        TQK.alternating
    elseif config.entanglement == "all_to_all"
        TQK.all_to_all
    else
        error("Unknown entanglement: $(config.entanglement)")
    end
    
    Random.seed!(config.seed)
    feature_map = ReuploadingCircuit(
        config.n_qubits, 
        data_config.n_features, 
        config.n_layers, 
        entanglement_enum
    )
    assign_random_params!(feature_map, seed=config.seed)
    
    # Create kernel
    kernel = FidelityKernel(feature_map)
    
    # Set up loss function
    loss_fn = if config.loss_function == "kernel_alignment"
        K -> kernel_alignment_loss(K, y_train)
    else
        error("Loss function $(config.loss_function) not implemented")
    end
    
    # Create trainer
    trainer = QuantumKernelTrainer(
        kernel,
        loss_fn,
        X_train,
        y_train,
        memory_budget_gb=config.memory_budget_gb
    )
    
    # Select optimizer
    if config.optimizer == "LBFGS"
        opt = LBFGS()
    elseif config.optimizer == "BFGS"
        opt = BFGS()
    else
        error("Optimizer $(config.optimizer) not implemented")
    end
    
    # Train
    @info "Starting training with $(config.optimizer) for $(config.max_iterations) iterations"
    sol = train!(trainer, 
                optimizer=opt,
                iterations=config.max_iterations,
                callback=(state, loss) -> @info "Iter $(state.iter): loss = $loss")
    
    @info "Training complete. Final loss: $(sol.minimum)"
    
    return Dict(
        :type => "reuploading",
        :kernel => kernel,
        :feature_map => feature_map,
        :final_loss => sol.minimum,
        :training_solution => sol
    )
end
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

@kwdef struct PauliKernelConfig <: KernelConfig
    kernel_type::Symbol = :pauli
    
    # Circuit parameters
    n_qubits::Int = 2
    reps::Vector{Int} = [1, 2, 3]
    entanglement::Vector{String} = ["full", "linear", "circular"]
    
    # Search configuration
    search_strategy::String = "random"  
    n_search_iterations::Int = 50
    search_constraints::PauliSearchConstraints = PauliSearchConstraints()
       
    # Cross-validation for search
    cv_folds::Int = 3
    cv_samples::Int = 100  # Use subset of training data for CV
    
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

"""
Evaluate a single Pauli kernel configuration using cross-validation
"""
function evaluate_pauli_kernel(
    paulis::Vector{String},
    n_qubits::Int,
    reps::Int,
    entanglement::String,
    X_cv::Matrix{Float64},
    y_cv::Vector{Float64},
    cv_folds::Int
)
    # Use PythonCall to create Qiskit feature map
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    feature_map = qiskit_lib.PauliFeatureMap(
        feature_dimension=size(X_cv, 2),
        reps=reps,
        entanglement=entanglement,
        paulis=pylist(paulis)
    )
    
    quantum_kernel = qiskit_kernels.FidelityStatevectorKernel(
        feature_map=feature_map
    )
    
    # Compute kernel matrix
    K = pyconvert(Matrix{Float64}, quantum_kernel.evaluate(x_vec=X_cv))
    
    # Simple KTA score as evaluation metric
    y_outer = y_cv * y_cv'
    K_centered = K .- mean(K)
    y_centered = y_outer .- mean(y_outer)
    
    kta = sum(K_centered .* y_centered) / (sqrt(sum(K_centered.^2)) * sqrt(sum(y_centered.^2)))
    
    return kta
end

"""
Search for the best Pauli kernel configuration
"""
function search_best_pauli_kernel(
    config::PauliKernelConfig,
    X_train::Matrix{Float64},
    y_train::Vector{Float64}
)
    @info "Starting Pauli kernel search" config.search_strategy config.n_search_iterations
    
    # Use subset for cross-validation
    n_cv = min(config.cv_samples, size(X_train, 1))
    cv_indices = randperm(size(X_train, 1))[1:n_cv]
    X_cv = X_train[cv_indices, :]
    y_cv = y_train[cv_indices]
    
    best_score = -Inf
    best_paulis = nothing
    all_results = []
    
    Random.seed!(config.seed)
    
    for i in 1:config.n_search_iterations
        # Generate candidate Pauli set
        paulis = generate_constrained_pauli_set(
            config.search_constraints,
            seed=config.seed + i
        )
        
        try
            # Evaluate this configuration
            score = evaluate_pauli_kernel(
                paulis,
                config.n_qubits,
                config.reps,
                config.entanglement,
                X_cv,
                y_cv,
                config.cv_folds
            )
            
            @info "Iteration $i/$(config.n_search_iterations)" paulis score
            
            push!(all_results, (paulis=paulis, score=score))
            
            if score > best_score
                best_score = score
                best_paulis = paulis
            end
        catch e
            @warn "Iteration $i failed" exception=e
            push!(all_results, (paulis=paulis, score=-Inf))
        end
    end
    
    @info "Search complete" best_score best_paulis
    
    return best_paulis, best_score, all_results
end

"""
Produce a Pauli feature map kernel (with optional search)
"""
function produce_kernel(config::PauliKernelConfig, data_config::DataConfig; n_samples::Union{Int, Nothing}=nothing)
    @info "Producing Pauli Quantum Kernel" config.n_qubits config.reps
    
    # Load data
    data = load(data_config.data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Use subset if specified
    if !isnothing(n_samples)
        X_train = X_train[1:min(n_samples, size(X_train, 1)), :]
        y_train = y_train[1:min(n_samples, length(y_train))]
        @info "Using data subset" n_samples
    end
    
    # Determine Pauli strings (either search or use fixed)
    if !isnothing(config.fixed_paulis)
        @info "Using fixed Pauli strings" config.fixed_paulis
        paulis = config.fixed_paulis
        search_results = nothing
    else
        @info "Searching for optimal Pauli strings"
        paulis, best_score, search_results = search_best_pauli_kernel(
            config, X_train, y_train
        )
    end
    
    # Create the final feature map with best Pauli strings
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    feature_map = qiskit_lib.PauliFeatureMap(
        feature_dimension=data_config.n_features,
        reps=config.reps,
        entanglement=config.entanglement,
        paulis=pylist(paulis)
    )
    
    quantum_kernel = qiskit_kernels.FidelityStatevectorKernel(
        feature_map=feature_map
    )
    
    return Dict(
        :type => "pauli",
        :kernel => quantum_kernel,
        :feature_map => feature_map,
        :paulis => paulis,
        :search_results => search_results
    )
end

# ============================================================================
# Master Experiment Configuration
# ============================================================================

@kwdef struct ExperimentConfig
    experiment_name::String
    description::String = ""
    
    # Data configuration
    data_config::DataConfig
    
    # Model configuration (this model learns the data) 
    kernel_config::KernelConfig
    
    # Learning curve configuration
    learning_curve_sizes::Vector{Int} = Int[]  # Empty means no learning curves
    
    # Experiment tracking
    seed::Int = 42
    created_at::String = string(now())
end

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
    
    kernel_configs = KernelConfig[]

    kernel_type = if kernel_type == :rbf
        RBFKernelConfig(gamma="auto")
    elseif kernel_type == :reuploading
        ReuploadingKernelConfig(
            n_qubits = n_features, n_layers = 2, entanglement = "linear",
            max_iterations = 50, seed = seed
        )
    elseif kernel_type == :pauli
        PauliKernelConfig(
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
# 
# ============================================================================
# Pipeline Functions
# ============================================================================

"""
Step 1: Generate or load data according to data_config
"""
function prepare_data!(config::ExperimentConfig)
    dc = config.data_config
    
    # Create unique filename using DrWatson
    data_params = @dict(
        dataset_type = dc.dataset_type,
        n_samples = dc.n_samples,
        n_features = dc.n_features,
        seed = dc.seed
    )
    
    filename = savename(data_params, "jld2")
    filepath = datadir("sims", dc.dataset_type, filename)
    
    # Check if data already exists
    if isfile(filepath)
        @info "Data already exists, loading..." filepath
        config.data_config.data_path = filepath
        return filepath
    end
    
    @info "Generating new data..." dc.dataset_type
    
    # Generate based on dataset type
    if dc.dataset_type == "rbf"
        # Call your RBF data generation function
        data_dict = generate_pseudo_svm_dataset(
            n_samples=dc.n_samples,
            n_features=dc.n_features,
            seed=dc.seed,
            dc.data_params...
        )
    elseif dc.dataset_type == "quantum_pauli"
        # Call your Pauli data generation function
        data_dict = generate_pauli_expectation_data_grid(
            # your parameters here
            seed=dc.seed,
            dc.data_params...
        )
    else
        error("Unknown dataset type: $(dc.dataset_type)")
    end
    
    # Save data
    wsave(filepath, data_dict)
    config.data_config.data_path = filepath
    
    @info "Data saved" filepath
    return filepath
end

"""
Step 2: Produce all kernels with optional learning curves
"""
function produce_all_kernels(config::ExperimentConfig)
    results = Dict{String, Any}()
    
    for (i, kernel_config) in enumerate(config.kernel_configs)
        kernel_name = "$(kernel_config.kernel_type)_$i"
        @info "Processing kernel $kernel_name"
        
        if isempty(config.learning_curve_sizes)
            # Single run on full training data
            @info "Single training run on full data"
            kernel_result = produce_kernel(kernel_config, config.data_config)
            results[kernel_name] = kernel_result
            
        else
            # Learning curve: Phase 1 (hyperparameter selection) + Phase 2 (learning curves)
            @info "Running learning curves" config.learning_curve_sizes
            
            # Phase 1: Hyperparameter selection on full training data
            @info "Phase 1: Hyperparameter selection on full training data"
            best_config = select_hyperparameters(kernel_config, config.data_config)
            
            # Phase 2: Train on increasing data sizes
            @info "Phase 2: Training on increasing data sizes"
            curve_results = Dict{Int, Any}()
            
            for n_samples in config.learning_curve_sizes
                @info "Training with n=$n_samples samples"
                kernel_result = produce_kernel(best_config, config.data_config, n_samples=n_samples)
                curve_results[n_samples] = kernel_result
            end
            
            results[kernel_name] = Dict(
                :type => kernel_config.kernel_type,
                :best_config => best_config,
                :learning_curves => curve_results
            )
        end
        
        # Save kernel result
        kernel_params = @dict(
            experiment_name = config.experiment_name,
            kernel_type = kernel_config.kernel_type,
            kernel_index = i
        )
        
        kernel_filename = savename(kernel_params, "jld2")
        kernel_filepath = datadir("kernels", config.experiment_name, kernel_filename)
        wsave(kernel_filepath, results[kernel_name])
        @info "Kernel saved" kernel_filepath
    end
    
    return results
end

"""
Phase 1: Select best hyperparameters for a kernel
For RBF: Cross-validate gamma on full training data
For Reuploading: Already optimizes on full data, return same config
For Pauli: Run search on full data, return best config
"""
function select_hyperparameters(config::KernelConfig, data_config::DataConfig)
    if config isa RBFKernelConfig
        if config.gamma == "auto"
            @info "Cross-validating RBF gamma on full training data"
            # Run CV to select best gamma, return updated config
            data = load(data_config.data_path)
            X_train = data["X_train"]
            y_train = data["y_train"]
            
            # Simple CV (you can use a proper CV library)
            best_gamma = config.gamma_range[1]
            best_score = -Inf
            
            for gamma in config.gamma_range
                # Compute KTA as selection metric
                K = exp.(-gamma .* pairwise_squared_distances(X_train))
                y_outer = y_train * y_train'
                kta = sum(K .* y_outer) / (sqrt(sum(K.^2)) * sqrt(sum(y_outer.^2)))
                
                if kta > best_score
                    best_score = kta
                    best_gamma = gamma
                end
            end
            
            @info "Selected gamma" best_gamma best_score
            return RBFKernelConfig(gamma=best_gamma)
        else
            return config  # Gamma already specified
        end
        
    elseif config isa ReuploadingKernelConfig
        # Reuploading already trains on full data in produce_kernel
        # Just return the config as-is
        return config
        
    elseif config isa PauliKernelConfig
        if isnothing(config.fixed_paulis)
            @info "Searching for best Pauli strings on full training data"
            data = load(data_config.data_path)
            X_train = data["X_train"]
            y_train = data["y_train"]
            
            best_paulis, best_score, _ = search_best_pauli_kernel(config, X_train, y_train)
            
            @info "Selected Pauli strings" best_paulis best_score
            return PauliKernelConfig(
                n_qubits = config.n_qubits,
                reps = config.reps,
                entanglement = config.entanglement,
                fixed_paulis = best_paulis,
                seed = config.seed
            )
        else
            return config  # Paulis already specified
        end
    end
end

# Helper for RBF gamma selection
function pairwise_squared_distances(X::Matrix)
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n
        for j in i:n
            d = sum((X[i, :] - X[j, :]).^2)
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

"""
Main pipeline: Run complete experiment
"""
function run_experiment(config::ExperimentConfig)
    @info "Starting experiment" config.experiment_name
    
    # Step 1: Prepare data
    prepare_data!(config)
    
    # Step 2: Produce all kernels
    kernels = produce_all_kernels(config)
    
    # Step 3: Train SVMs and evaluate (implement next)
    # results = train_and_evaluate_svms(config, kernels)
    
    @info "Experiment complete" config.experiment_name
    
    return kernels
end

# ============================================================================
# Example Usage
# ============================================================================


# Define how to handle the abstract type
StructTypes.StructType(::Type{ExperimentConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{KernelConfig}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{KernelConfig}) = :kernel_type
StructTypes.subtypes(::Type{KernelConfig}) = (
    rbf = RBFKernelConfig,
    reuploading = ReuploadingKernelConfig,
    pauli = PauliKernelConfig
)
