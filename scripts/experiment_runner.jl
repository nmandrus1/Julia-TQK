using DrWatson
@quickactivate "TQK"

using TQK
using LIBSVM
using Distances

include("experiment_registry_refactored.jl")


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
            ent = config.entanglement,
            lr = config.learning_rate
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
                    # learning_rate = params[:learning_rate],
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
        
        exp_name = savename(data_config)

        # Create experiment for each kernel on this dataset
        for (i, kernel_config) in enumerate(reup_configs)
            kernel_id = kernel_identifier(kernel_config)
            
            push!(experiments, ExperimentConfig(
                experiment_name = exp_name,
                data_config = data_config,
                kernel_config = kernel_config,
                learning_curve_sizes = collect(100:250:data_config.n_samples),
                seed = data_config.seed
            ))
        end
    end
    
    @info "Generated $(length(experiments)) experiments"
    return experiments
end

# ============================================================================
# EXPERIMENT EXECUTION (UPDATED FOR LONG FORMAT)
# ============================================================================


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


"""
Run single experiment: one dataset, one kernel.
"""
function run_experiment!(exp_config::ExperimentConfig)
    exp_id = exp_config.experiment_name
    kernel_id = kernel_identifier(exp_config.kernel_config)
    
    @info "Running: $exp_id / $kernel_id"
    
    # Register experiment and kernel
    register_experiment!(exp_config)
    register_kernel_result!(exp_id, kernel_id)
    update_kernel_result!(exp_id, kernel_id, status=:running)
    
    try
        # 1. Generate/load data
        @info "Loading data..."
        data, data_path = prepare_data!(exp_config)
        update_experiment_data_path!(exp_id, data_path)
        
        X_train = permutedims(data["X_train"])
        y_train = Vector(data["y_train"])
        
        # 2. Hyperparameter search (cached)
        @info "Searching hyperparameters..."
        result_dir = datadir("experiments", "results", exp_id, kernel_id)
        hyperparam_file = joinpath(result_dir, "hyperparameters.jld2")
        
        if isfile(hyperparam_file)
            @info "Loading cached hyperparameters"
            hyperparams = load(hyperparam_file, "hyperparams")
        else
            hyperparams, score = search_hyperparameters(X_train, y_train, exp_config)
            @info "Best score: $score"
            wsave(hyperparam_file, @dict hyperparams score)
        end
        
        # 3. Learning curves
        @info "Running learning curves..."
        learning_curves = run_learning_curve(hyperparams, data, exp_config.learning_curve_sizes)
        
        curves_file = joinpath(result_dir, "learning_curves.jld2")
        wsave(curves_file, @dict learning_curves)
        
        # 4. Update with metrics
        test_accs = [lc[:test_acc] for lc in values(learning_curves)]
        train_accs = [lc[:train_acc] for lc in values(learning_curves)]
        sizes = [lc[:n_samples] for lc in values(learning_curves)]
        
        metrics = Dict(
            :best_test_acc => maximum(test_accs),
            :best_train_acc => maximum(train_accs),
            :final_train_size => maximum(sizes)
        )
        
        update_kernel_result!(exp_id, kernel_id, status=:completed, metrics=metrics)
        @info "Completed: $exp_id / $kernel_id"
        
    catch e
        @error "Failed: $exp_id / $kernel_id" exception=e
        update_kernel_result!(exp_id, kernel_id, status=:failed)
        rethrow(e)
    end
end

"""
Run batch of experiments.
"""
function run_experiments!(experiments::Vector{ExperimentConfig})
    for (i, exp_config) in enumerate(experiments)
        @info "Progress: $i/$(length(experiments))"
        run_experiment!(exp_config)
    end
    
    @info "Complete. Summary:"
    display(summarize_experiments())
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Quick demo of query functions.
"""
function demo_queries()
    println("\n=== Experiment Statistics ===")
    display(experiment_stats())
    
    println("\n=== Top 5 Experiments ===")
    display(top_experiments(5))
    
    println("\n=== Kernel Comparison ===")
    display(compare_kernels())
    
    println("\n=== Summary by Experiment ===")
    display(summarize_experiments())
    
    println("\n=== Find RBF Kernels ===")
    display(find_experiments(kernel_id="rbf"))
end

export run_experiment!, run_experiments!, demo_queries


"""
Quick start: generate and run small test.
"""
function quick_test()
    # Single RBF dataset
    data_config1 = DataConfig(
        n_samples=500,
        n_features=2,
        data_params=RBFDataParams(gamma=2.0, n_support_vectors=10),
        seed=42
    )

    data_config2 = DataConfig(
        n_samples=500,
        n_features=2,
        data_params=ReuploadingDataParams(
            n_qubits=4,
            n_features=2,
            n_layers=4,
            entanglement=all_to_all,
            n_support_vectors=10,
        ),
        seed = 53
    )
    
    # Test 4 reuploading configs
    reup_grid = Dict(
        :n_qubits => [2, 4, 8],
        :n_layers => [2, 4, 8],
        :entanglement => ["linear", "all_to_all"],
    )
    
    experiments = generate_experiment_grid(
        dataset_configs=[data_config1, data_config2],
        reuploading_grid=reup_grid,
        test_rbf=true,
        test_pauli=true
    )
    
    run_experiments!(experiments)    

end


"""
Full experiment suite.
"""
function run_full_suite()
    # Multiple datasets
    rbf_configs = [
        DataConfig(
            n_samples=5000,
            n_features=2,
            data_params=RBFDataParams(gamma=2.0, n_support_vectors=nsv),
            seed=s
        )
        for nsv in [50]
        for s in [42]
    ]

    reuploading_2q_2l_configs = [
        DataConfig(
            n_samples=5000,
            n_features=2,
            data_params=ReuploadingDataParams(
                n_qubits=8,
                n_features=2,
                n_layers=4,
                entanglement=all_to_all,
                n_support_vectors=nsv,
            ),
            seed = s
        )
        for nsv in [50]
        for s in [42]
    ]

    # reuploading_8q_8l_configs = [
    #     DataConfig(
    #         n_samples=5000,
    #         n_features=2,
    #         data_params=ReuploadingDataParams(
    #             n_qubits=8,
    #             n_features=2,
    #             n_layers=8,
    #             entanglement=all_to_all,
    #             n_support_vectors=nsv,
    #         ),
    #         seed = s
    #     )
    #     for nsv in [10, 50]
    #     for s in [42]
    # ]    
    # quantum_configs = [
    #     DataConfig(
    #         n_samples=5000,
    #         n_features=2,

    #         data_params=QuantumPauliDataParams(
    #             n_qubits=2,
    #             paulis=p,
    #             reps=2,
    #             entanglement=ent,
    #             gap=0.3,
    #             grid_points_per_dim=100
    #         ),
    #         seed=s
    #     )
    #     for (p, ent) in [(["ZZ"], "linear"), (["X", "XY"], "full")]
    #     for s in [42]
    # ]
    
    # Full reuploading grid
    reup_grid = Dict(
        :n_qubits => [8],
        :n_layers => [4],
        :entanglement => ["linear"],
    )
    
    experiments = generate_experiment_grid(
        # dataset_configs=vcat(rbf_configs, reuploading_2q_2l_configs, reuploading_8q_8l_configs),
        dataset_configs=vcat(rbf_configs, reuploading_2q_2l_configs),
        reuploading_grid=reup_grid,
        test_rbf=true,
        test_pauli=true
    )
    
    @info "Generated $(length(experiments)) total experiments"
    
    run_experiments!(experiments)
end
