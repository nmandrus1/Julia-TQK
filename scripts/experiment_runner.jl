using DrWatson
@quickactivate "TQK"

using TQK
using Dates
using LinearAlgebra
using StableRNGs
using LIBSVM
using Statistics
using JLD2
using Yao

# ============================================================================
# 1. MAIN ENTRY POINT (DrWatson Wrapper)
# ============================================================================

"""
    run_experiment(config::ExperimentConfig; force=false)

The primary entry point. Wraps the experiment logic in DrWatson's `produce_or_load`
to ensure we never re-run identical experiments unless forced.
"""
function run_experiment(config::ExperimentConfig; force=false)
    # 1. Define unique filename based on config
    #    DrWatson.savename uses the struct fields to create a unique string.
    #    "exp_Name_seed=123_..."
    filename = savename(config, "jld2", allowedtypes=(String, Int, Float64, Symbol))
    
    # 2. Output Directory
    #    We organize results by method type for cleaner folders
    method_type = string(typeof(config.method))
    output_dir = datadir("experiments", "results", method_type)
    
    # 3. Produce or Load
    #    - If file exists: Load it.
    #    - If not: Run `execute_pipeline`, save it, return it.
    output, filepath = produce_or_load(
        execute_pipeline, # Function to run
        config,           # Input to function
        output_dir;       # Where to save
        suffix="jld2",
        filename=filename,
        force=force,
        tag=true          # Auto-save git commit info
    )
    
    @info "Experiment artifact available at: $filepath"
    return output["artifacts"]
end

# ============================================================================
# 2. EXECUTION PIPELINE (The Logic)
# ============================================================================

"""
    execute_pipeline(config::ExperimentConfig)

Runs the full TQK pipeline:
1. Data Gen/Load
2. Kernel Tuning (Student learns from Teacher)
3. Matrix Computation (Geometry extraction)
4. Classical Optimization (SVM)
5. Artifact Packaging
"""
function execute_pipeline(config::ExperimentConfig)
    @info "--- Starting Pipeline: $(config.name) ---"
    
    # 1. Reproducibility: Derive RNGs
    #    These are deterministic based on master_seed
    rng_sampling= derive_rng(config.master_seed, SALT_SAMPLING)  
    rng_optim = derive_rng(config.master_seed, SALT_OPTIMIZER)  
    rng_svm = derive_rng(config.master_seed, SALT_SVM_CV)  

    # 2. Data Preparation
    #    produce_data handles its own seed from config.data_config
    @info "Loading Data..."
    data = produce_data(config.data_config)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    # 3. Kernel Tuning
    @info "Tuning Kernel ($(config.method.name))..."
    tune_conf = TQK.TuningConfig(
        sampling_rng = rng_sampling,
        optimizer_rng = rng_optim,
        batch_size = config.tuning_batch_size
    )
    
    tuning_result = tune_kernel(config.method, X_train, y_train, tune_conf)
    best_params = tuning_result.best_params
    @info "Tuning Complete. Best KTA: $(tuning_result.best_score)"

    # 4. Compute Final Geometry (Kernels)
    @info "Computing full kernel matrices..."
    K_train = compute_final_kernel_matrix(best_params, X_train)
    # For test, we need K(X_test, X_train) because X_train are the support vectors
    K_test  = compute_cross_kernel_matrix(best_params, X_test, X_train) 

    # 5. Optional: Compute Statevectors (For Hypersphere Analysis)
    #    Only if system is small enough to fit in memory/disk
    train_statevectors = nothing
    if should_save_statevectors(best_params)
        @info "Computing statevectors for geometric analysis..."
        train_statevectors = compute_final_statevectors(best_params, X_train)
    end

    # 6. Classical Optimization (SVM C-Search)
    @info "Optimizing SVM..."
    best_c, cv_acc, _ = tune_svm_c(K_train, y_train, config.c_grid; 
                                   cv_folds=config.cv_folds, rng=rng_svm)
    
    # Final Evaluation (Refit on full train with best C)
    model = svmtrain(K_train, y_train, kernel=Kernel.Precomputed, cost=best_c, verbose=false)
    
    # Predict (LIBSVM expects (n_features, n_samples), here features=support_vectors)
    # K_train is symmetric, K_test is (n_test, n_train). LIBSVM needs K'.
    train_preds, _ = svmpredict(model, K_train)
    test_preds, _  = svmpredict(model, K_test')
    
    train_acc = mean(train_preds .== y_train)
    test_acc  = mean(test_preds .== y_test)
    
    @info "Results: Train Acc=$train_acc, Test Acc=$test_acc"

    # 7. Package Artifacts
    artifacts = ExperimentArtifacts(
        config,
        tuning_result,
        K_train,
        K_test,
        train_statevectors,
        best_c,
        cv_acc,
        train_acc,
        test_acc,
        now(),
        gitdescribe()
    )

    # DrWatson requires a Dict return
    return Dict("artifacts" => artifacts)
end

# ============================================================================
# 3. HELPER FUNCTIONS (Matrix Dispatch)
# ============================================================================

"""
Computes K(X, X) based on the TrainedKernel type.
"""
function compute_final_kernel_matrix(k::TrainedRBFKernel, X)
    return rbf_kernel_matrix(X, X, k.gamma)
end

function compute_final_kernel_matrix(k::TrainedReuploadingKernel, X)
    # Use pure simulation for the final matrix (exact geometry)
    return compute_kernel_matrix_pure(k.config, k.params, X)
end

function compute_final_kernel_matrix(k::TrainedPauliKernel, X)
    return compute_kernel_matrix_pure(k.config, [], X)
end

"""
Computes K(X_test, X_train) for SVM prediction.
"""
function compute_cross_kernel_matrix(k::TrainedRBFKernel, X_test, X_train)
    return rbf_kernel_matrix(X_test, X_train, k.gamma)
end

function compute_cross_kernel_matrix(k::TrainedReuploadingKernel, X_test, X_train)
    # Ensure compute_kernel_matrix_pure supports 2 args (X, Y)
    return compute_kernel_matrix_pure(k.config, k.params, X_test, X_train)
end

function compute_cross_kernel_matrix(k::TrainedPauliKernel, X_test, X_train)
    return compute_kernel_matrix_pure(k.config, [], X_test, X_train)
end

"""
Computes raw statevectors for analysis.
"""
function compute_final_statevectors(k::TrainedReuploadingKernel, X)
    # Returns Matrix{ComplexF64} of size (2^n_qubits, n_samples)
    states_vec = compute_statevectors(k.config, k.params, X)
    return hcat(state.(states_vec)...)
end


"""
Computes raw statevectors for analysis.
"""
function compute_final_statevectors(k::TrainedPauliKernel, X)
    # Returns Matrix{ComplexF64} of size (2^n_qubits, n_samples)
    states_vec = compute_statevectors(k.config, [], X)
    return hcat(state.(states_vec)...)
end

function compute_final_statevectors(k::Any, X)
    return nothing # Not applicable for RBF
end

# Heuristic to avoid saving massive statevectors
should_save_statevectors(k::TrainedReuploadingKernel) = k.config.n_qubits <= 12
should_save_statevectors(k::TrainedPauliKernel) = k.config.n_features <= 12
should_save_statevectors(k::Any) = false

# ============================================================================
# 4. GRID GENERATION (Config Factories)
# ============================================================================

"""
Generate a list of ExperimentConfig objects for batch execution.
"""
function generate_experiment_grid(;
    dataset_configs::Vector{<:DataConfig},
    reuploading_grid::Dict,
    test_rbf::Bool=true,
    test_pauli::Bool=true,
    master_seed::Int = 42
)
    experiments = ExperimentConfig[]
    
    for data_config in dataset_configs
        # 1. Reuploading Configurations
        if !isempty(reuploading_grid)
            # DrWatson.dict_list expands the grid
            reup_params_list = dict_list(reuploading_grid)
            
            for params in reup_params_list
                # Build the Method Config
                method = ReuploadingMethod(
                    name = savename("reup", params, connector="_"),
                    circuit_config = ReuploadingConfig(
                        params[:n_qubits],
                        data_config.n_features,
                        params[:n_layers],
                        params[:entanglement]
                    ),
                    optimizer = SPSAConfig(
                        max_iter=get(params, :max_iter, 200),
                        n_resamples=get(params, :n_resamples, 1)
                    )
                )
                
                push!(experiments, ExperimentConfig(
                    name = "$(savename(data_config))_$(method.name)",
                    master_seed = master_seed,
                    data_config = data_config,
                    method = method
                ))
            end
        end
        
        # 2. RBF Baseline
        if test_rbf
            method = RBFMethod(name="rbf_baseline")
            push!(experiments, ExperimentConfig(
                name = "$(savename(data_config))_rbf",
                master_seed = master_seed,
                data_config = data_config,
                method = method
            ))
        end
        
        # 3. Pauli Baseline
        if test_pauli
            method = PauliMethod(
                name = "pauli_baseline",
                n_qubits = data_config.n_features # Usually matches features
            )
            push!(experiments, ExperimentConfig(
                name = "$(savename(data_config))_pauli",
                master_seed = master_seed,
                data_config = data_config,
                method = method
            ))
        end
    end
    
    @info "Generated $(length(experiments)) experiments."
    return experiments
end

"""
Execute a batch of experiments sequentially.
"""
function run_experiments!(experiments::Vector{ExperimentConfig})
    for (i, config) in enumerate(experiments)
        @info "Running Experiment $(i)/$(length(experiments))..."
        try
            run_experiment(config)
        catch e
            @error "Experiment $(config.name) failed!" exception=e
            # Continue to next experiment despite failure
        end
    end
end

export run_experiment, run_experiments!, generate_experiment_grid
