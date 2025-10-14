using DrWatson
@quickactivate "TQK"

using DataFrames
using JLD2
using Dates
using Statistics
using PythonCall
using LIBSVM
using Printf
using Distances

# ============================================================================
# KERNEL NAMING & IDENTIFICATION
# ============================================================================

"""
Generate unique identifier for a kernel configuration.
"""
function kernel_identifier(config::KernelHyperparameterSearchConfig)
    if config isa RBFKernelHyperparameterSearchConfig
        return "rbf"
    elseif config isa PauliKernelHyperparameterSearchConfig
        return "pauli"
    elseif config isa ReuploadingKernelHyperparameterSearchConfig
        params = @dict(
            kernel = :reup,
            q = config.n_qubits,
            l = config.n_layers,
            ent = config.entanglement
        )
        return savename(params, connector="_")
    end
end

"""
Parse kernel identifier back to type and params.
"""
function parse_kernel_id(kid::String)
    if kid == "rbf"
        return :rbf, Dict()
    elseif kid == "pauli"
        return :pauli, Dict()
    else
        # Parse "reup_ent_linear_l_2_q_4"
        parts = Dict(split(p, "_")[1] => split(p, "_")[2] 
                    for p in split(kid, "_") if length(split(p, "_")) == 2)
        return :reup, parts
    end
end

# ============================================================================
# REGISTRY MANAGEMENT
# ============================================================================

"""
Load or create master registry.
"""
function load_registry()
    registry_path = datadir("experiments", "master_registry.jld2")
    
    if isfile(registry_path)
        return load(registry_path, "registry")
    else
        # Create empty registry with proper schema
        return DataFrame(
            experiment_id = String[],
            status = Symbol[],
            created_at = DateTime[],
            completed_at = Union{DateTime, Missing}[],
            dataset_type = String[],
            dataset_params = Dict[],
            n_samples = Int[],
            n_features = Int[],
            data_seed = Int[],
            tested_kernels = Vector{String}[],
            learning_curve_sizes = Vector{Int}[],
            data_path = String[],
            result_path = String[]
        )
    end
end

"""
Save registry atomically.
"""
function save_registry!(registry::DataFrame)
    registry_path = datadir("experiments", "master_registry.jld2")
    mkpath(dirname(registry_path))
    wsave(registry_path, @dict registry)
end

"""
Add experiment to registry.
"""

"""
Add experiment to registry.
"""
function register_experiment!(registry::DataFrame, exp_config::ExperimentConfig)
    exp_id = exp_config.experiment_name
    
    # Check if exists
    if exp_id in registry.experiment_id
        @warn "Experiment $exp_id already registered"
        return exp_id
    end
    
    # Create result directory
    result_path = datadir("experiments", "results", exp_id)
    mkpath(result_path)
    
    # Add row
    push!(registry, Dict(
        :experiment_id => exp_id,
        :status => :pending,
        :created_at => now(),
        :completed_at => missing,
        :dataset_type => string(exp_config.data_config.data_params.dataset_type),
        # Convert the struct to a Dict before pushing
        :dataset_params => DrWatson.struct2dict(exp_config.data_config.data_params),
        :n_samples => exp_config.data_config.n_samples,
        :n_features => exp_config.data_config.n_features,
        :data_seed => exp_config.data_config.seed,
        :tested_kernels => [kernel_identifier(exp_config.kernel_config)],
        :learning_curve_sizes => exp_config.learning_curve_sizes,
        :data_path => "",  # Will be filled after data generation
        :result_path => result_path
    ), cols=:union)
    
    save_registry!(registry)
    return exp_id
end

"""
Update experiment status and metrics.
"""
function update_experiment!(registry::DataFrame, exp_id::String; 
                           status::Union{Symbol, Nothing}=nothing,
                           data_path::Union{String, Nothing}=nothing,
                           kernel_metrics::Union{Dict, Nothing}=nothing)
    idx = findfirst(==(exp_id), registry.experiment_id)
    
    if !isnothing(status)
        registry[idx, :status] = status
        if status == :completed
            registry[idx, :completed_at] = now()
        end
    end
    
    if !isnothing(data_path)
        registry[idx, :data_path] = data_path
    end
    
    if !isnothing(kernel_metrics)
        for (kernel_id, metrics) in kernel_metrics
            # Add columns dynamically if needed
            col_test = Symbol("$(kernel_id)_best_test_acc")
            col_train = Symbol("$(kernel_id)_best_train_acc")
            
            if !(col_test in names(registry))
                registry[!, col_test] = Vector{Union{Float64, Missing}}(missing, nrow(registry))
            end
            if !(col_train in names(registry))
                registry[!, col_train] = Vector{Union{Float64, Missing}}(missing, nrow(registry))
            end
            
            registry[idx, col_test] = metrics[:best_test_acc]
            registry[idx, col_train] = metrics[:best_train_acc]
        end
    end
    
    save_registry!(registry)
end

# ============================================================================
# EXPERIMENT GRID GENERATION
# ============================================================================

"""
Generate experiment grid with many reuploading configs.
"""
function generate_experiment_grid(;
    dataset_configs::Vector{<:DataConfig},
    reuploading_grid::Dict,
    test_rbf::Bool=true,
    test_pauli::Bool=true
)
    experiments = ExperimentConfig[]
    
    for data_config in dataset_configs
        # Expand reuploading grid
        reup_configs = KernelHyperparameterSearchConfig[]
        
        if !isempty(reuploading_grid)
            reup_params = dict_list(reuploading_grid)
            
            for params in reup_params
                push!(reup_configs, ReuploadingKernelHyperparameterSearchConfig(
                    n_qubits = params[:n_qubits],
                    n_features = data_config.n_features,
                    n_layers = params[:n_layers],
                    entanglement = params[:entanglement],
                    seed = data_config.seed
                ))
            end
        end
        
        # Add RBF
        if test_rbf
            push!(reup_configs, RBFKernelHyperparameterSearchConfig())
        end
        
        # Add Pauli
        if test_pauli
            push!(reup_configs, PauliKernelHyperparameterSearchConfig(
                n_qubits = data_config.n_features,
                seed = data_config.seed
            ))
        end
        
        # Create experiment for each kernel on this dataset
        for (i, kernel_config) in enumerate(reup_configs)
            kernel_id = kernel_identifier(kernel_config)
            exp_name = "$(savename(data_config))_$(kernel_id)"
            
            push!(experiments, ExperimentConfig(
                experiment_name = exp_name,
                data_config = data_config,
                kernel_config = kernel_config,
                learning_curve_sizes = collect(100:100:min(1000, data_config.n_samples)),
                seed = data_config.seed
            ))
        end
    end
    
    @info "Generated $(length(experiments)) experiments"
    return experiments
end

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================
"""
Dispatch hyperparameter search based on kernel type.
"""
function search_hyperparameters(
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    config::ExperimentConfig
)
    if config.kernel_config isa RBFKernelHyperparameterSearchConfig
        return search_rbf_hyperparameters(config, X_train, y_train)
    elseif config.kernel_config isa PauliKernelHyperparameterSearchConfig
        return search_pauli_hyperparameters(config, X_train, y_train)
    elseif config.kernel_config isa ReuploadingKernelHyperparameterSearchConfig
        return train_reuploading_kernel_with_cv(config, X_train, y_train)
    end
end

# ============================================================================
# LEARNING CURVE EVALUATION
# ============================================================================

"""
Evaluate kernel at specific training size.
"""
function evaluate_at_size(
    hyperparams::KernelHyperparameters,
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    X_test::AbstractMatrix,
    y_test::AbstractVector,
    n_samples::Int
)
    X_sub = X_train[1:n_samples, :]
    y_sub = y_train[1:n_samples]

    local K_train, K_test, C
    
    # Compute kernel
    if hyperparams isa RBFHyperparameters

        # 1. Calculate pairwise squared Euclidean distances.
        #    NOTE: Distances.jl expects features in columns, so we transpose (').
        D_train_sq = pairwise(SqEuclidean(), X_sub', dims=2)
        K_train = exp.(-hyperparams.gamma .* D_train_sq)
        @show size(K_train)
        
        D_test_sq = pairwise(SqEuclidean(), X_test', X_sub', dims=2)
        K_test = exp.(-hyperparams.gamma .* D_test_sq)

        C = hyperparams.C
        
    elseif hyperparams isa PauliKernelHyperparameters
        K_train = compute_pauli_kernel_matrix(hyperparams, X_sub)
        K_test = compute_pauli_kernel_matrix(hyperparams, X_test, X_sub)       
        C = hyperparams.C
        
    elseif hyperparams isa ReuploadingKernelHyperparameters
        ent_map = Dict("linear"=>linear, "alternating"=>alternating, "all_to_all"=>all_to_all)
        
        # Reconstruct feature map with trained params
        nqubits = hyperparams.nqubits
        nlayers = hyperparams.nlayers
        feature_map = ReuploadingCircuit(nqubits, size(X_train, 2), nlayers , linear)
        assign_params!(feature_map, hyperparams.thetas, hyperparams.biases)
        
        kernel = FidelityKernel(feature_map)
        K_train = TQK.evaluate(kernel, X_sub)
        K_test = TQK.evaluate(kernel, X_test, X_sub)
        
        C = hyperparams.C
    end
    
    # Train SVM
    model = svmtrain(K_train, y_sub, kernel=Kernel.Precomputed, cost=C, verbose=false)
    
    train_pred = svmpredict(model, K_train)[1]
    test_pred = svmpredict(model, K_test')[1]
    
    return Dict(
        :train_acc => mean(train_pred .== y_sub),
        :test_acc => mean(test_pred .== y_test),
        :n_samples => n_samples
    )
end

"""
Run learning curve for a kernel.
"""
function run_learning_curve(
    hyperparams::KernelHyperparameters,
    data::Dict,
    sizes::Vector{Int}
)
    X_train = permutedims(data["X_train"])  # To row-major
    X_test = permutedims(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    results = Dict{Int, Dict}()
    
    for size in sizes
        if size > length(y_train)
            continue
        end
        
        @info "  Evaluating at size $size"
        results[size] = evaluate_at_size(hyperparams, X_train, y_train, X_test, y_test, size)
    end
    
    return results
end

# ============================================================================
# MAIN EXPERIMENT EXECUTION
# ============================================================================

"""
Run single experiment: one dataset, one kernel.
"""
function run_experiment!(registry::DataFrame, exp_config::ExperimentConfig)
    exp_id = exp_config.experiment_name
    kernel_id = kernel_identifier(exp_config.kernel_config)
    
    @info "Running experiment: $exp_id"
    
    # Register if new
    if !(exp_id in registry.experiment_id)
        register_experiment!(registry, exp_config)
    end
    
    update_experiment!(registry, exp_id, status=:running)
    
    try
        # 1. Generate/load data
        @info "Loading data..."
        data, data_path = prepare_data!(exp_config)
        update_experiment!(registry, exp_id, data_path=data_path)
        
        X_train = permutedims(data["X_train"])
        y_train = Vector(data["y_train"])
        
        # 2. Hyperparameter search (cached)
        @info "Searching hyperparameters for $kernel_id..."
        result_dir = datadir("experiments", "results", exp_id, kernel_id)
        mkpath(result_dir)
        
        hyperparam_file = joinpath(result_dir, "hyperparameters.jld2")
        
        if isfile(hyperparam_file)
            @info "Loading cached hyperparameters"
            hyperparams = load(hyperparam_file, "hyperparams")
        else
            hyperparams, score = search_hyperparameters(X_train, y_train, exp_config)
            @info "Best hyperparameters found (score: $score)"
            wsave(hyperparam_file, @dict hyperparams score)
        end
        
        # 3. Learning curves
        @info "Running learning curves..."
        learning_curves = run_learning_curve(hyperparams, data, exp_config.learning_curve_sizes)
        
        curves_file = joinpath(result_dir, "learning_curves.jld2")
        wsave(curves_file, @dict learning_curves)
        
        # 4. Update registry with metrics
        test_accs = [lc[:test_acc] for lc in values(learning_curves)]
        train_accs = [lc[:train_acc] for lc in values(learning_curves)]
        
        metrics = Dict(
            kernel_id => Dict(
                :best_test_acc => maximum(test_accs),
                :best_train_acc => maximum(train_accs)
            )
        )
        
        update_experiment!(registry, exp_id, status=:completed, kernel_metrics=metrics)
        
        @info "Experiment $exp_id completed successfully"
        
    catch e
        @error "Experiment $exp_id failed" exception=e
        update_experiment!(registry, exp_id, status=:failed)
        rethrow(e)
    end
end

"""
Run batch of experiments.
"""
function run_experiments!(experiments::Vector{ExperimentConfig})
    registry = load_registry()
    
    for (i, exp_config) in enumerate(experiments)
        @info "Progress: $i/$(length(experiments))"
        run_experiment!(registry, exp_config)
    end
    
    @info "All experiments complete"
end

# ============================================================================
# RESULT LOADING & ANALYSIS
# ============================================================================

"""
Load detailed results for a specific kernel.
"""
function load_kernel_results(exp_id::String, kernel_id::String)
    result_dir = datadir("experiments", "results", exp_id, kernel_id)
    
    if !isdir(result_dir)
        error("Results not found for $exp_id/$kernel_id")
    end
    
    hyperparams = load(joinpath(result_dir, "hyperparameters.jld2"))
    curves = load(joinpath(result_dir, "learning_curves.jld2"))
    
    return Dict(
        :hyperparameters => hyperparams,
        :learning_curves => curves
    )
end

"""
Load all results for an experiment.
"""
function load_experiment_results(exp_id::String)
    registry = load_registry()
    row = registry[findfirst(==(exp_id), registry.experiment_id), :]
    
    results = Dict{String, Any}()
    
    for kernel_id in row.tested_kernels
        results[kernel_id] = load_kernel_results(exp_id, kernel_id)
    end
    
    return results
end

"""
Query registry with filters.
"""
function query_registry(; kwargs...)
    registry = load_registry()
    
    for (key, value) in kwargs
        if key in names(registry)
            if value isa Vector
                registry = filter(row -> row[key] in value, registry)
            else
                registry = filter(row -> row[key] == value, registry)
            end
        end
    end
    
    return registry
end

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

"""
Quick start: generate and run small test.
"""
function quick_test()
    # Single RBF dataset
    data_config = DataConfig(
        n_samples=500,
        n_features=2,
        data_params=RBFDataParams(gamma=1.0, n_support_vectors=20),
        seed=42
    )
    
    # Test 4 reuploading configs
    reup_grid = Dict(
        :n_qubits => [2, 4],
        :n_layers => [2],
        :entanglement => ["linear"],
    )
    
    experiments = generate_experiment_grid(
        dataset_configs=[data_config],
        reuploading_grid=reup_grid,
        test_rbf=true,
        test_pauli=true
    )
    
    run_experiments!(experiments)
    
    return load_registry()
end

"""
Full experiment suite.
"""
function run_full_suite()
    # Multiple datasets
    rbf_configs = [
        DataConfig(
            n_samples=1000,
            n_features=2,
            data_params=RBFDataParams(gamma=g, n_support_vectors=50),
            seed=s
        )
        for g in [0.1, 1.0, 10.0]
        for s in [42, 123]
    ]
    
    quantum_configs = [
        DataConfig(
            n_samples=1000,
            n_features=2,
            data_params=QuantumPauliDataParams(
                n_qubits=2,
                paulis=p,
                reps=2,
                entanglement="linear",
                gap=0.3,
                grid_points_per_dim=100
            ),
            seed=s
        )
        for p in [["Z", "ZZ"], ["X", "XY"]]
        for s in [42, 123]
    ]
    
    # Full reuploading grid
    reup_grid = Dict(
        :n_qubits => [2, 4, 6, 8],
        :n_layers => [2, 4, 6, 8],
        :entanglement => ["linear", "all_to_all"],
        :optimizer => ["LBFGS"],
        :max_iterations => [100]
    )
    
    experiments = generate_experiment_grid(
        dataset_configs=vcat(rbf_configs, quantum_configs),
        reuploading_grid=reup_grid,
        test_rbf=true,
        test_pauli=true
    )
    
    @info "Generated $(length(experiments)) total experiments"
    
    run_experiments!(experiments)
end

# Export
export load_registry, save_registry!, register_experiment!, update_experiment!
export generate_experiment_grid, run_experiment!, run_experiments!
export load_kernel_results, load_experiment_results, query_registry
export quick_test, run_full_suite
