using Base: Experimental

using DrWatson
@quickactivate "TQK"

using TQK
using DataFrames
using JLD2
using JSON3

# ============================================================================
# Helper for dataframe saving
# ============================================================================

"""
Recursively flatten any struct or dictionary into a flat dictionary.
Uses dot notation for nested field names (e.g., "data_config.n_samples").
"""
function flatten_struct(obj; prefix::String="", sep::String=".")
    result = Dict{String, Any}()
    
    # Handle different types
    if obj isa Dict
        for (k, v) in obj
            key_str = string(k)
            new_prefix = isempty(prefix) ? key_str : prefix * sep * key_str
            merge!(result, flatten_struct(v, prefix=new_prefix, sep=sep))
        end
    elseif typeof(obj) <: Number || obj isa String || obj isa Symbol || obj isa Bool
        # Base types - just store
        result[prefix] = obj
    elseif obj isa AbstractArray
        # Store arrays as-is (you could serialize if needed)
        result[prefix] = obj
    elseif obj isa Nothing
        result[prefix] = missing
    else
        # It's a struct - get all fieldnames
        for field in fieldnames(typeof(obj))
            value = getfield(obj, field)
            new_prefix = isempty(prefix) ? string(field) : prefix * sep * string(field)
            merge!(result, flatten_struct(value, prefix=new_prefix, sep=sep))
        end
    end
    
    return result
end

# ============================================================================
# Grid Definition
# ============================================================================

"""
Define the parameter grid for experiments.
This function now separates parameters into logical groups.
"""
function define_parameter_grid()
    n_samples = 10000

    # --- Core Parameters (Common to all experiments) ---
    core_params = Dict(
        :n_samples => n_samples,
        :n_features => 2,
        :learning_curve_sizes => [collect(100:100:n_samples)],
        :seed => [42, 123, 456]
    )

    # --- RBF-Specific Data Parameters ---
    rbf_params = Dict(
        :dataset_type => "rbf",
        :rbf_gamma => [0.1, 1.0, 10.0],
        :n_support_vectors => [10, 50, 100, 1000]
    )

    # --- Quantum-Specific Data Parameters ---
    quantum_params = Dict(
        :dataset_type => "quantum_pauli",
        :quantum_paulis => [["Z", "ZZ"], ["X", "XY"], ["XYZ"]],
        :quantum_entanglement => ["full", "linear"],
        :quantum_reps => [1, 2]
    )
    
    # --- Reuploading-Specific Data Parameters ---
    reuploading_params = Dict(
        :kernel_type => :reuploading,
        :reuploading_n_qubits => [2, 3, 4, 5, 6, 7, 8],
        :reuploading_n_layers => [1, 2, 3, 4],
        :reuploading_optimizer => ["LBFGS", "Adam"]
    )

    
    # Generate all data configurations 
    classical_data_configs = merge(core_params, rbf_params)
    quantum_data_configs = merge(core_params, quantum_params)

    # generate the different 
    classical_vs_reupload = dict_list(merge(classical_data_configs, reuploading_params))
    quantum_vs_reupload = dict_list(merge(quantum_data_configs, reuploading_params))

    classical_vs_rbf = dict_list(merge(classical_data_configs, @dict(kernel_type=:rbf)))
    quantum_vs_rbf = dict_list(merge(quantum_data_configs, @dict(kernel_type=:rbf)))
    
    classical_vs_pauli = dict_list(merge(classical_data_configs, @dict(kernel_type=:pauli)))
    quantum_vs_pauli = dict_list(merge(quantum_data_configs, @dict(kernel_type=:pauli)))

    # Combine the configurations into a single DataFrame
    experiment_list = vcat(
        classical_vs_reupload,
        quantum_vs_reupload,
        classical_vs_rbf,
        quantum_vs_rbf,
        classical_vs_pauli,
        quantum_vs_pauli
    )

    return experiment_list
end

"""
Generate all valid ExperimentConfig combinations from the grid.
This version uses DrWatson.dict_list for a more concise and readable implementation.
"""
function generate_experiment_grid()
    exp_config_list = define_parameter_grid()

    configs_df = DataFrame()

    for params in exp_config_list

        # Create the full ExperimentConfig object for each parameter combination
        config = create_config_from_params(params)
        # Convert the config to a DataFrame row and append it
        append!(configs_df, config_to_dataframe_row(config), cols= :union)
    end

    return configs_df
end

"""
Create ExperimentConfig from parameter values
"""
function create_config_from_params(params::Dict{Symbol, Any})
    seed = params[:seed]
    lc_sizes = params[:learning_curve_sizes]
    kernel_type = params[:kernel_type]
    n_features = params[:n_features]


    # Build data config
    if params[:dataset_type] == "rbf"
        data_params = Dict(
                           :gamma => params[:rbf_gamma],
                           :n_support_vectors => params[:n_support_vectors]
                       )
        data_name = savename(data_params)
    else
        data_params = Dict(
                           :n_qubits => n_features,
                           :paulis => params[:quantum_paulis],
                           :reps => params[:quantum_reps],
                           :entanglement => params[:quantum_entanglement]
                       )
        params[:pauli_gates] = join(params[:quantum_paulis], '-')
        data_name = savename(data_params)
    end
    
    data_config = DataConfig(
        dataset_type = params[:dataset_type],
        dataset_name = data_name,
        n_samples = params[:n_samples],
        n_features = n_features,
        data_params = data_params,
        seed = seed
    )
    
    # Build kernel configs
    kernel_config = if kernel_type == :rbf
        RBFKernelConfig(gamma="auto")
    elseif kernel_type == :reuploading
        ReuploadingKernelConfig(
            n_qubits = params[:reuploading_n_qubits],
            n_features = n_features,
            n_layers = params[:reuploading_n_layers],
            optimizer= params[:reuploading_optimizer],
            seed = seed
        )
    elseif kernel_type == :pauli
        PauliKernelConfig(n_qubits = n_features, seed = seed)
    end
    
    #  Create experiment name
    ignored_keys = ["n_samples", "n_features", "learning_curve_sizes"] 
    exp_name = savename(params, ignores=ignored_keys, connector="__")
    
    return ExperimentConfig(
        experiment_name = exp_name,
        description = "Auto-generated from grid",
        data_config = data_config,
        kernel_config = kernel_config,
        learning_curve_sizes = lc_sizes,
        seed = seed
    )
end

"""
Convert ExperimentConfig to DataFrame row for easy filtering
"""
function config_to_dataframe_row(config::ExperimentConfig)
    flat_exp = flatten_struct(config)
    flat_exp["json"] = JSON3.write(config)
    return DataFrame([flat_exp])
end

# ============================================================================
# Save/Load Configurations
# ============================================================================

"""
Generate and save all experiment configurations
"""
function generate_and_save_configs(; force_regenerate=false)
    config_file = datadir("experiments", "all_configs.jld2")
    
    if !force_regenerate && isfile(config_file)
        @info "Loading existing configurations" config_file
        return load(config_file, "configs_df"), load(config_file, "configs")
    end
    
    @info "Generating experiment grid..."
    configs_df = generate_experiment_grid()
    
    # Save each config individually
    configs = Dict{String, ExperimentConfig}()
    for row in eachrow(configs_df)
        # Reconstruct config (we need to store the actual config objects)
        exp_name = row.experiment_name
        config = reconstruct_config_from_row(row)
        configs[exp_name] = config
        
        # Save individual config
        config_path = datadir("experiments", "configs", "$(exp_name).jld2")
        wsave(config_path, @dict config)
    end
    
    # Save summary
    wsave(config_file, @dict configs_df configs)
    
    @info "Generated $(nrow(configs_df)) experiment configurations"
    return configs_df, configs
end

"""
Reconstruct full ExperimentConfig from DataFrame row (simplified version)
In practice, you'd need more logic here based on what you stored
"""
function reconstruct_config_from_row(row)
    return JSON3.read(row[:json], ExperimentConfig)
    error("Use load_config(experiment_name) instead")
end

"""
Load a specific experiment configuration
"""
function load_config(experiment_name::String)
    config_path = datadir("experiments", "configs", "$(experiment_name).jld2")
    if !isfile(config_path)
        error("Config not found: $experiment_name")
    end
    return load(config_path, "config")
end

"""
Filter configurations by criteria
"""
function filter_configs(configs_df::DataFrame; kwargs...)
    filtered = configs_df
    
    for (key, value) in kwargs
        if key in names(filtered)
            if value isa Vector
                filtered = filter(row -> row[key] in value, filtered)
            else
                filtered = filter(row -> row[key] == value, filtered)
            end
        end
    end
    
    return filtered
end

# ============================================================================
# Experiment Execution
# ============================================================================

"""
Check if experiment has been completed
"""
function experiment_completed(experiment_name::String)
    result_file = datadir("results", experiment_name, "results.jld2")
    return isfile(result_file)
end

"""
Check if data has been generated
"""
function data_exists(data_config::DataConfig)
    if isempty(data_config.data_path)
        # Compute expected path
        data_params = @dict(
            dataset_type = data_config.dataset_type,
            n_samples = data_config.n_samples,
            n_features = data_config.n_features,
            seed = data_config.seed
        )
        filename = savename(data_params, "jld2")
        filepath = datadir("sims", data_config.dataset_type, filename)
        return isfile(filepath)
    else
        return isfile(data_config.data_path)
    end
end

"""
Run a single experiment by name, with automatic data reuse
"""
function run_experiment_by_name(experiment_name::String; force_rerun=false)
    # Check if already completed
    if !force_rerun && experiment_completed(experiment_name)
        @info "Experiment already completed, skipping" experiment_name
        return :skipped
    end
    
    # Load config
    config = load_config(experiment_name)
    
    # Check/prepare data (auto-loads if exists)
    if data_exists(config.data_config)
        @info "Data already exists, will reuse" config.data_config.dataset_type
    else
        @info "Generating new data" config.data_config.dataset_type
    end
    
    prepare_data!(config)  # This function already checks for existing data
    
    # Run experiment
    @info "Running experiment" experiment_name
    results = produce_all_kernels(config)
    
    # Save results with the experiment
    result_dir = datadir("results", experiment_name)
    mkpath(result_dir)
    
    result_data = Dict(
        :config => config,
        :results => results,
        :timestamp => now(),
        :status => "completed"
    )
    
    wsave(datadir("results", experiment_name, "results.jld2"), result_data)
    
    @info "Experiment completed" experiment_name
    return :completed
end

"""
Run multiple experiments from filtered list
"""
function run_experiments(experiment_names::Vector{String}; force_rerun=false)
    results = Dict{String, Symbol}()
    
    for (i, name) in enumerate(experiment_names)
        @info "=" ^ 80
        @info "Experiment $i/$(length(experiment_names))"
        status = run_experiment_by_name(name, force_rerun=force_rerun)
        results[name] = status
    end
    
    # Summary
    completed = count(==(:completed), values(results))
    skipped = count(==(:skipped), values(results))
    
    @info "=" ^ 80
    @info "Batch complete: $completed completed, $skipped skipped"
    
    return results
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
Quick start: Generate configs and run a small test subset
"""
function quick_start(; n_test=3)
    # Generate all configs
    configs_df, configs = generate_and_save_configs()
    
    @info "Total configurations: $(nrow(configs_df))"
    
    # Select a diverse test subset
    test_configs = first(configs_df, n_test)
    
    @info "Running $n_test test experiments:"
    show(test_configs[!, [:experiment_name, :dataset_type, :n_samples, :kernel_types]])
    println()
    
    # Run them
    test_names = test_configs.experiment_name
    results = run_experiments(test_names)
    
    return configs_df, results
end

"""
List all available experiment configs with filtering
"""
function list_experiments(; show_completed=false, kwargs...)
    configs_df, _ = generate_and_save_configs()
    
    # Apply filters
    filtered = filter_configs(configs_df; kwargs...)
    
    # Add completion status
    filtered.completed = [experiment_completed(name) for name in filtered.experiment_name]
    
    if !show_completed
        filtered = filter(row -> !row.completed, filtered)
    end
    
    return filtered
end

"""
Run all RBF experiments
"""
function run_rbf_experiments(; force_rerun=false)
    configs = list_experiments(dataset_type="rbf", show_completed=false)
    @info "Found $(nrow(configs)) RBF experiments to run"
    return run_experiments(configs.experiment_name, force_rerun=force_rerun)
end

"""
Run all quantum experiments
"""
function run_quantum_experiments(; force_rerun=false)
    configs = list_experiments(dataset_type="quantum_pauli", show_completed=false)
    @info "Found $(nrow(configs)) quantum experiments to run"
    return run_experiments(configs.experiment_name, force_rerun=force_rerun)
end

"""
Generate summary report of all results
"""
function summarize_results()
    configs_df, _ = generate_and_save_configs()
    
    # Add completion info
    configs_df.completed = [experiment_completed(name) for name in configs_df.experiment_name]
    
    @info "Experiment Summary"
    @info "Total configs: $(nrow(configs_df))"
    @info "Completed: $(sum(configs_df.completed))"
    @info "Remaining: $(sum(.!configs_df.completed))"
    
    # Group by dataset type
    gdf = groupby(configs_df, :dataset_type)
    for group in gdf
        dataset_type = first(group.dataset_type)
        completed = sum(group.completed)
        total = nrow(group)
        @info "  $dataset_type: $completed/$total completed"
    end
    
    return configs_df
end

# ============================================================================
# Main
# ============================================================================

"""
Interactive menu
"""
function main()
    println("\n" * "=" ^ 80)
    println("Quantum Kernel Learning - Experiment Grid Manager")
    println("=" ^ 80 * "\n")
    
    println("Commands:")
    println("  1. quick_start(n_test=3)           - Generate configs and test 3 experiments")
    println("  2. list_experiments()              - View all pending experiments")
    println("  3. run_rbf_experiments()           - Run all RBF experiments")
    println("  4. run_quantum_experiments()       - Run all quantum experiments")
    println("  5. summarize_results()             - View progress summary")
    println("  6. generate_and_save_configs()     - Regenerate config grid")
    println()
    println("Filtering examples:")
    println("  list_experiments(n_samples=1000, seed=42)")
    println("  list_experiments(dataset_type=\"rbf\", n_features=6)")
    println()
    
    # Check if configs exist
    config_file = datadir("experiments", "all_configs.jld2")
    if isfile(config_file)
        configs_df = load(config_file, "configs_df")
        @info "Loaded $(nrow(configs_df)) existing configurations"
    else
        @info "No configurations found. Run generate_and_save_configs() to create them."
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


"""
Quick test of an experiment config to run the entire process
"""
function quick_test()
    n_samples=1000
    exp = ExperimentConfig(
        experiment_name = "QUICK_TEST",
        description= "testing the pipeline",
        data_config=DataConfig(
            dataset_type="rbf",
            dataset_name="QUICK_TEST_rbf",
            n_samples=n_samples,
            data_params = Dict(
               :dataset_type => "quantum_pauli",
               :quantum_paulis =>["XYZ"],
               :quantum_entanglement => "linear",
               :quantum_reps => 1
            ),
        ),       
        kernel_config=PauliKernelConfig(),
        learning_curve_sizes=collect(100:100:n_samples),
    )

    # produce or load data

    return exp
end
