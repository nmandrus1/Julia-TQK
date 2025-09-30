
using DrWatson
@quickactivate "TQK"

using TQK
using DataFrames
using JLD2

# ============================================================================
# Grid Definition
# ============================================================================

"""
Define the parameter grid for experiments.
Returns vectors of parameter values to be combined.
"""
function define_parameter_grid()
    return Dict(
        # Data parameters
        :dataset_type => ["rbf", "quantum_pauli"],
        :n_samples => [500, 1000],
        :n_features => [4, 6],
        
        # RBF data specific
        :rbf_gamma => [0.1, 1.0, 10.0],
        :n_support_vectors => [10, 20],
        
        # Quantum data specific
        :quantum_paulis => [["Z", "ZZ"], ["X", "XY", "XYZ"]],
        :quantum_reps => [1, 2],
        
        # Kernel types to test
        :kernel_types => [
            [:rbf],                    # RBF only
            [:reuploading],           # Reuploading only
            [:pauli],                 # Pauli only
            [:rbf, :reuploading, :pauli]  # All three
        ],
        
        # Learning curve sizes
        :learning_curve_sizes => [
            [50, 100, 200],           # Small curves
            [100, 200, 400, 800]      # Full curves
        ],
        
        # Seeds for replicates
        :seed => [42, 123, 456]
    )
end

"""
Generate all valid ExperimentConfig combinations from grid.
Returns a DataFrame of configurations.
"""
function generate_experiment_grid()
    grid = define_parameter_grid()
    
    configs = DataFrame[]
    
    # Generate combinations
    for dataset_type in grid[:dataset_type]
        for n_samples in grid[:n_samples]
            for n_features in grid[:n_features]
                for kernel_types in grid[:kernel_types]
                    for lc_sizes in grid[:learning_curve_sizes]
                        for seed in grid[:seed]
                            
                            # Dataset-specific parameters
                            if dataset_type == "rbf"
                                for gamma in grid[:rbf_gamma]
                                    for n_sv in grid[:n_support_vectors]
                                        config = create_config_from_params(
                                            dataset_type, n_samples, n_features,
                                            kernel_types, lc_sizes, seed,
                                            rbf_gamma=gamma, n_support_vectors=n_sv
                                        )
                                        push!(configs, config_to_dataframe_row(config))
                                    end
                                end
                            else  # quantum
                                for paulis in grid[:quantum_paulis]
                                    for reps in grid[:quantum_reps]
                                        config = create_config_from_params(
                                            dataset_type, n_samples, n_features,
                                            kernel_types, lc_sizes, seed,
                                            quantum_paulis=paulis, quantum_reps=reps
                                        )
                                        push!(configs, config_to_dataframe_row(config))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return vcat(configs...)
end

"""
Create ExperimentConfig from parameter values
"""
function create_config_from_params(
    dataset_type, n_samples, n_features, kernel_types, lc_sizes, seed;
    rbf_gamma=nothing, n_support_vectors=nothing,
    quantum_paulis=nothing, quantum_reps=nothing
)
    # Build data config
    if dataset_type == "rbf"
        data_params = Dict(:gamma => rbf_gamma, :n_support_vectors => n_support_vectors)
        data_name = "rbf_g$(rbf_gamma)_sv$(n_support_vectors)"
    else
        data_params = Dict(:n_qubits => n_features, :paulis => quantum_paulis, :reps => quantum_reps)
        paulis_str = join(quantum_paulis, "-")
        data_name = "quantum_p$(paulis_str)_r$(quantum_reps)"
    end
    
    data_config = DataConfig(
        dataset_type = dataset_type,
        dataset_name = data_name,
        n_samples = n_samples,
        n_features = n_features,
        data_params = data_params,
        seed = seed
    )
    
    # Build kernel configs
    kernel_configs = KernelConfig[]
    for kt in kernel_types
        if kt == :rbf
            push!(kernel_configs, RBFKernelConfig(gamma="auto"))
        elseif kt == :reuploading
            push!(kernel_configs, ReuploadingKernelConfig(
                n_qubits = n_features,
                n_layers = 2,
                entanglement = "linear",
                max_iterations = 50,
                seed = seed
            ))
        elseif kt == :pauli
            push!(kernel_configs, PauliKernelConfig(
                n_qubits = n_features,
                reps = 2,
                n_search_iterations = 20,
                seed = seed
            ))
        end
    end
    
    # Create experiment name
    kernel_str = join(string.(kernel_types), "-")
    lc_str = "lc$(length(lc_sizes))"
    exp_name = savename(
        @dict(dataset_type, data_name, n_samples, n_features, kernel_str, lc_str, seed),
        connector="_"
    )
    
    return ExperimentConfig(
        experiment_name = exp_name,
        description = "Auto-generated from grid",
        data_config = data_config,
        kernel_configs = kernel_configs,
        learning_curve_sizes = lc_sizes,
        seed = seed
    )
end

"""
Convert ExperimentConfig to DataFrame row for easy filtering
"""
function config_to_dataframe_row(config::ExperimentConfig)
    dc = config.data_config
    
    row = Dict(
        :experiment_name => config.experiment_name,
        :dataset_type => dc.dataset_type,
        :n_samples => dc.n_samples,
        :n_features => dc.n_features,
        :n_kernels => length(config.kernel_configs),
        :kernel_types => join([kc.kernel_type for kc in config.kernel_configs], ","),
        :n_lc_points => length(config.learning_curve_sizes),
        :seed => config.seed
    )
    
    # Add dataset-specific params
    if dc.dataset_type == "rbf"
        row[:rbf_gamma] = dc.data_params[:gamma]
        row[:n_support_vectors] = dc.data_params[:n_support_vectors]
    else
        row[:quantum_paulis] = join(dc.data_params[:paulis], "-")
        row[:quantum_reps] = dc.data_params[:reps]
    end
    
    return DataFrame([row])
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
    # This is a simplified version - you'd need full parameter reconstruction
    # For now, we rely on loading from individual saved configs
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
