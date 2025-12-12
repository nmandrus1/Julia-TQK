using DrWatson
@quickactivate "TQK"

using TQK
using LinearAlgebra
using Plots
using Random
using StableRNGs
using Yao
using Dates
using DataFrames
using Printf

# ============================================================================
# 1. CORE UTILS: Ray Tracing Engine
# ============================================================================

include("../experiment_runner.jl")
include("periodicity.jl")
# for kernel ray profile

# ============================================================================
# 2. STEP 1: Run the Experiment (Training)
# ============================================================================

"""
    run_periodicity_experiment_batch()

Runs a batch of training experiments on a single dataset using the current timestamp.
Returns the unique `experiment_tag` used to filter results later.
"""
function run_periodicity_experiment_batch()
    # 1. Setup Unique Run ID
    timestamp = Dates.format(now(), "yyyy-mm-dd--HH:MM:SS")
    experiment_tag = "periodicity_8q_run_$(timestamp)"
    println("=== STARTING EXPERIMENT BATCH: $experiment_tag ===")

    master_seed::UInt = 2021
    tuning_batch_size = 100

    # 2. Teacher: RBF Data (Smooth target to force frequency adaptation)
    data_conf = DataConfig(
        dataset_name = "teacher_rbf_$(experiment_tag)", 
        n_samples = 1000, 
        n_features = 4, # Embedding dimension
        params = RBFDataParams(gamma=2.0),
        master_seed = master_seed
    )   

    # 3. Students: Sweep over Depths (Layers)
    experiments = ExperimentConfig[]
    
    # We fix qubits=4 and sweep layers to see how depth affects the frequency shift
    n_qubits = 8
    depths = [2, 4, 8, 16] 

    for l in depths
        # Setup Method
        method = ReuploadingMethod(
            name = "student_reup_L$(l)",
            circuit_config = ReuploadingConfig(n_qubits, 4, l; entanglement=FullEntanglement),
            optimizer = SPSAConfig(
                max_iter = 100, 
                n_resamples = 3, 
            )
        )
        
        # Setup Experiment
        push!(experiments, ExperimentConfig(
            name = "exp_$(experiment_tag)_L$(l)",
            master_seed = master_seed,
            data_config = data_conf,
            method = method,
            tuning_batch_size = tuning_batch_size
        ))
    end

    
    # # We fix layers=4 and sweep qubits to see how depth affects the frequency shift
    # n_layers = 4
    # qubits = [1, 2, 4, 8] 

    # for nq in qubits
    #     # Setup Method
    #     method = ReuploadingMethod(
    #         name = "student_reup_NQ$(nq)",
    #         circuit_config = ReuploadingConfig(nq, 4, n_layers; entanglement=FullEntanglement),
    #         optimizer = SPSAConfig(
    #             max_iter = 100, 
    #             n_resamples = 3, 
    #         )
    #     )
        
    #     # Setup Experiment
    #     push!(experiments, ExperimentConfig(
    #         name = "exp_$(experiment_tag)_NQ$(nq)",
    #         master_seed = master_seed,
    #         data_config = data_conf,
    #         method = method,
    #         tuning_batch_size = tuning_batch_size
    #     ))
    # end

    # 4. Execute
    println("-> Scheduled $(length(experiments)) training runs...")
    for (i, config) in enumerate(experiments)
        println("   [$(i)/$(length(experiments))] Running $(config.name)...")
        try
            run_experiment(config) # Results saved to data/experiments/results/...
        catch e
            @error "Failed $(config.name)" exception=e
        end
    end
    
    println("-> Batch Complete.")
    return experiment_tag
end

# ============================================================================
# 3. STEP 2: Analysis (Comparative Ray Tracing)
# ============================================================================

"""
    analyze_periodicity_changes(experiment_tag::String)

1. Loads all results matching the `experiment_tag`.
2. Loads the Training Data (X_train) to compute matrix ranks.
3. Generates comparative plots with Rank info.
"""
function analyze_periodicity_changes(experiment_tag::String)
    println("\n=== ANALYZING BATCH: $experiment_tag ===")
    
    # 1. Collect Results
    df_all = collect_results(datadir("experiments", "results"); subfolders=true)
    results = filter(row -> occursin(experiment_tag, row.artifacts.config.data_config.dataset_name), df_all)
    
    if nrow(results) == 0
        error("No results found for tag '$experiment_tag'.")
    end
    println("-> Found $(nrow(results)) experiments.")

    # 2. Load Training Data (Once)
    #    We assume all experiments in this batch used the same dataset configuration
    first_config = results[1, :artifacts].config.data_config
    println("-> Loading Training Data: $(first_config.dataset_name)...")
    
    # produce_data is cached, so this is fast
    data_container = produce_data(first_config) 
    X_train = data_container["X_train"] # Needed for Rank Calculation

    # 3. Pre-process DataFrame
    transform!(results, :artifacts => ByRow(a -> a.config.method.circuit_config.n_layers) => :layers)
    transform!(results, :artifacts => ByRow(a -> a.config.method.circuit_config.n_qubits) => :qubits)
    transform!(results, :artifacts => ByRow(a -> a.config.method.name) => :method_name)

    # 4. Define Ray Geometry (Fixed)
    rng_ray = StableRNG(999) 
    n_features = 4
    x_origin = rand(rng_ray, n_features) .* 2π
    d_raw = randn(rng_ray, n_features)
    d_vec = d_raw ./ norm(d_raw)
    
    range_val = 6π
    steps = 1000
    ks = range(0, range_val, length=steps)
    ray_points = hcat([x_origin .+ k .* d_vec for k in ks]...)

    # 5. Generate Plots
    
    # --- Group A: Varying Layers ---
    df_layers = filter(:method_name => n -> occursin("_L", n), results)
    if !isempty(df_layers)
        println("\n-> Generating plot for Varying Layers...")
        sort!(df_layers, :layers)
        generate_periodicity_plot(
            df_layers, 
            ks, x_origin, ray_points, X_train,
            :layers, "Layers", 
            "periodicity_evolution_layers_$(experiment_tag).png"
        )
    end

    # --- Group B: Varying Qubits ---
    df_qubits = filter(:method_name => n -> occursin("_NQ", n), results)
    if !isempty(df_qubits)
        println("\n-> Generating plot for Varying Qubits...")
        sort!(df_qubits, :qubits)
        generate_periodicity_plot(
            df_qubits, 
            ks, x_origin, ray_points, X_train,
            :qubits, "Qubits", 
            "periodicity_evolution_qubits_$(experiment_tag).png"
        )
    end
end

"""
    generate_periodicity_plot(...)

Generates the multi-panel plot including Numerical Rank information.
"""
function generate_periodicity_plot(results_df, ks, x_origin, ray_points, X_train,
                                   sort_col, param_label, filename)
    
    n_plots = nrow(results_df)
    # Increase plot height to accommodate titles
    p = plot(layout=(n_plots, 1), size=(800, 350*n_plots), legend=:topright, left_margin=15Plots.mm)
    range_val = maximum(ks)

    for (i, row) in enumerate(eachrow(results_df))
        art = row.artifacts
        config = art.config
        method = config.method
        
        param_val = row[sort_col]
        println("   -> Processing $param_label=$(param_val)...")

        # --- A. Parameters ---
        trained_params = art.tuning_result.best_params.params
        rng_optim = derive_rng(config.master_seed, SALT_OPTIMIZER)
        init_params = rand(rng_optim, n_trainable_params(method.circuit_config))
        
        # --- B. Rank Calculation ---
        # 1. Trained Rank (Already computed in artifact)
        #    Note: art.K_train is the matrix K(X_train, X_train)
        rank_final = rank(art.K_train)
        
        # 2. Untrained Rank (Compute on the fly)
        #    We need to build the full kernel matrix for the random params
        #    Use TQK's compute_kernel_matrix_hardware (or pure if available/aliased)
        K_init = compute_kernel_matrix_hardware(method.circuit_config, init_params, X_train)
        rank_init = rank(K_init)
        
        # --- C. Compute Profiles (Ray Tracing) ---
        profile_init = compute_kernel_ray_profile(method.circuit_config, init_params, x_origin, ray_points)
        profile_final = compute_kernel_ray_profile(method.circuit_config, trained_params, x_origin, ray_points)
        
        # --- D. Plotting ---
        subplot = p[i]
        D = 2^method.circuit_config.n_qubits
        
        # Baseline
        plot!(subplot, [0, range_val], [1/D, 1/D], label="Random (1/D)", 
              linestyle=:dash, color=:black, alpha=0.5)
        
        # Initial
        plot!(subplot, ks, profile_init, label="Untrained", 
              color=:blue, alpha=0.6, lw=1)
              
        # Final
        plot!(subplot, ks, profile_final, label="Trained (Acc: $(round(art.test_acc, digits=2)))", 
              color=:red, lw=2)
              
        # Title with Rank Info
        title_str = "L=$(method.circuit_config.n_layers), N=$(method.circuit_config.n_qubits) | Rank: $(rank_init) → $(rank_final)"
        title!(subplot, title_str)
        ylabel!(subplot, "Fidelity")
        if i == n_plots xlabel!(subplot, "Distance k") end
    end

    # Save
    fpath = plotsdir("untrained_vs_trained_periodicity", filename)
    mkpath(dirname(fpath)) # Ensure directory exists
    savefig(fpath)
    println("-> Plot saved to $fpath")
end

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

function run_full_pipeline()
    # 1. Run Experiments
    tag = run_periodicity_experiment_batch()
    
    # 2. Run Analysis on the generated data
    analyze_periodicity_changes(tag)
end

# Run immediately
run_full_pipeline()
