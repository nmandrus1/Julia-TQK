
using DrWatson
using Random
using PythonCall
using OptimizationOptimJL

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
    kernel_type::String = "rbf"
    gamma::Union{Float64, String} = "auto"  # or specific value
    
    # If "auto", will use cross-validation to find best gamma
    cv_folds::Int = 3
    gamma_range::Vector{Float64} = [0.001, 0.01, 0.1, 1.0, 10.0]
end

"""
Produce an RBF kernel (just returns the configuration since sklearn handles it)
"""
function produce_kernel(config::RBFKernelConfig, data_config::DataConfig)
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

@kwdef struct ReuploadingKernelConfig <: KernelConfig
    kernel_type::String = "reuploading"
    
    # Circuit architecture
    n_qubits::Int = 4
    n_layers::Int = 2
    entanglement::String = "linear"  # "linear", "alternating", "all_to_all"
    
    # Training configuration
    optimizer::String = "LBFGS"
    max_iterations::Int = 100
    loss_function::String = "svm_loss"
    
    # Optimization parameters
    learning_rate::Float64 = 0.01
    convergence_tol::Float64 = 1e-6
    
    # Memory management
    memory_budget_gb::Float64 = 8.0
    
    # Reproducibility
    seed::Int = 42
end

"""
Produce a trained reuploading quantum kernel
"""
function produce_kernel(config::ReuploadingKernelConfig, data_config::DataConfig)
    @info "Training Reuploading Quantum Kernel" config.n_qubits config.n_layers config.entanglement
    
    # Load data
    data = load(data_config.data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
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
    kernel_type::String = "pauli"
    
    # Circuit parameters
    n_qubits::Int = 4
    reps::Int = 2
    entanglement::String = "full"  # "full", "linear", "circular"
    
    # Search configuration
    search_strategy::String = "random"  # "random", "exhaustive", "bayesian"
    n_search_iterations::Int = 50
    search_constraints::PauliSearchConstraints = PauliSearchConstraints()
    
    # If not searching, specify paulis directly
    fixed_paulis::Union{Nothing, Vector{String}} = nothing
    
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
function produce_kernel(config::PauliKernelConfig, data_config::DataConfig)
    @info "Producing Pauli Quantum Kernel" config.n_qubits config.reps
    
    # Load data
    data = load(data_config.data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    
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
    
    # Model configurations (one or more)
    kernel_configs::Vector{KernelConfig}
    
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
        num_kernels = length(config.kernel_configs),
        seed = config.seed,
        created_at = config.created_at
    )
end

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
Step 2: Produce all kernels
"""
function produce_all_kernels(config::ExperimentConfig)
    kernels = Dict{String, Any}()
    
    for (i, kernel_config) in enumerate(config.kernel_configs)
        kernel_name = "$(kernel_config.kernel_type)_$i"
        @info "Producing kernel $kernel_name"
        
        kernel_result = produce_kernel(kernel_config, config.data_config)
        kernels[kernel_name] = kernel_result
        
        # Save kernel result
        kernel_params = @dict(
            experiment_name = config.experiment_name,
            kernel_type = kernel_config.kernel_type,
            kernel_index = i
        )
        
        kernel_filename = savename(kernel_params, "jld2")
        kernel_filepath = datadir("kernels", config.experiment_name, kernel_filename)
        
        wsave(kernel_filepath, kernel_result)
        @info "Kernel saved" kernel_filepath
    end
    
    return kernels
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

function example_experiment()
    # Define a simple experiment comparing all three kernel types
    
    exp_config = ExperimentConfig(
        experiment_name = "quantum_vs_classical_comparison",
        description = "Compare RBF, Reuploading, and Pauli kernels on RBF-generated data",
        
        data_config = DataConfig(
            dataset_type = "rbf",
            n_samples = 500,
            n_features = 6,
            test_size = 0.2,
            data_params = Dict(
                :gamma => 1.0,
                :n_support_vectors => 20
            ),
            seed = 42
        ),
        
        kernel_configs = [
            RBFKernelConfig(
                gamma = "auto",
                gamma_range = [0.1, 1.0, 10.0]
            ),
            
            ReuploadingKernelConfig(
                n_qubits = 4,
                n_layers = 2,
                entanglement = "linear",
                optimizer = "LBFGS",
                max_iterations = 50,
                seed = 42
            ),
            
            PauliKernelConfig(
                n_qubits = 4,
                reps = 2,
                entanglement = "full",
                search_strategy = "random",
                n_search_iterations = 20,
                search_constraints = PauliSearchConstraints(
                    base_paulis = ["X", "Y", "Z"],
                    max_pauli_order = 2,
                    min_num_terms = 1,
                    max_num_terms = 3
                ),
                seed = 42
            )
        ],
        
        seed = 42
    )
    
    # Run the experiment
    kernels = run_experiment(exp_config)
    
    return exp_config, kernels
end
