
using DrWatson
@quickactivate "TQK"

using TQK
using DataFrames
using JLD2
using Dates
using Statistics

# ============================================================================
# REGISTRY STRUCTURE (LONG FORMAT)
# ============================================================================

"""
Load or create experiment registry (metadata only).
"""
function load_experiment_registry()
    path = datadir("experiments", "experiment_registry.jld2")
    
    if isfile(path)
        return load(path, "registry")
    else
        return DataFrame(
            experiment_id = String[],
            created_at = DateTime[],
            dataset_type = String[],
            dataset_params = Dict[],
            n_samples = Int[],
            n_features = Int[],
            data_seed = Int[],
            learning_curve_sizes = Vector{Int}[],
            data_path = String[],
            result_path = String[]
        )
    end
end

"""
Load or create results registry (one row per kernel per experiment).
"""
function load_results_registry()
    path = datadir("experiments", "results_registry.jld2")
    
    if isfile(path)
        return load(path, "registry")
    else
        return DataFrame(
            experiment_id = String[],
            kernel_id = String[],
            status = Symbol[],
            completed_at = Union{DateTime, Missing}[],
            best_train_acc = Union{Float64, Missing}[],
            best_test_acc = Union{Float64, Missing}[],
            final_train_size = Union{Int, Missing}[],
            hyperparams_path = String[],
            curves_path = String[]
        )
    end
end

function save_experiment_registry!(registry::DataFrame)
    path = datadir("experiments", "experiment_registry.jld2")
    mkpath(dirname(path))
    wsave(path, @dict registry)
end

function save_results_registry!(registry::DataFrame)
    path = datadir("experiments", "results_registry.jld2")
    mkpath(dirname(path))
    wsave(path, @dict registry)
end

# ============================================================================
# REGISTRY UPDATES
# ============================================================================

function register_experiment!(exp_config::ExperimentConfig)
    registry = load_experiment_registry()
    exp_id = exp_config.experiment_name
    
    if exp_id in registry.experiment_id
        @warn "Experiment $exp_id already registered"
        return exp_id
    end
    
    result_path = datadir("experiments", "results", exp_id)
    mkpath(result_path)
    
    push!(registry, Dict(
        :experiment_id => exp_id,
        :created_at => now(),
        :dataset_type => string(exp_config.data_config.data_params.dataset_type),
        :dataset_params => DrWatson.struct2dict(exp_config.data_config.data_params),
        :n_samples => exp_config.data_config.n_samples,
        :n_features => exp_config.data_config.n_features,
        :data_seed => exp_config.data_config.seed,
        :learning_curve_sizes => exp_config.learning_curve_sizes,
        :data_path => "",
        :result_path => result_path
    ), cols=:union)
    
    save_experiment_registry!(registry)
    return exp_id
end

function register_kernel_result!(exp_id::String, kernel_id::String)
    registry = load_results_registry()
    
    # Check if already exists
    mask = (registry.experiment_id .== exp_id) .& (registry.kernel_id .== kernel_id)
    if any(mask)
        return
    end
    
    result_dir = datadir("experiments", "results", exp_id, kernel_id)
    mkpath(result_dir)
    
    push!(registry, Dict(
        :experiment_id => exp_id,
        :kernel_id => kernel_id,
        :status => :pending,
        :completed_at => missing,
        :best_train_acc => missing,
        :best_test_acc => missing,
        :final_train_size => missing,
        :hyperparams_path => joinpath(result_dir, "hyperparameters.jld2"),
        :curves_path => joinpath(result_dir, "learning_curves.jld2")
    ), cols=:union)
    
    save_results_registry!(registry)
end

function update_kernel_result!(exp_id::String, kernel_id::String;
                               status::Union{Symbol, Nothing}=nothing,
                               metrics::Union{Dict, Nothing}=nothing)
    registry = load_results_registry()
    
    mask = (registry.experiment_id .== exp_id) .& (registry.kernel_id .== kernel_id)
    idx = findfirst(mask)
    
    if isnothing(idx)
        error("Kernel result not found: $exp_id / $kernel_id")
    end
    
    if !isnothing(status)
        registry[idx, :status] = status
        if status == :completed
            registry[idx, :completed_at] = now()
        end
    end
    
    if !isnothing(metrics)
        registry[idx, :best_train_acc] = get(metrics, :best_train_acc, missing)
        registry[idx, :best_test_acc] = get(metrics, :best_test_acc, missing)
        registry[idx, :final_train_size] = get(metrics, :final_train_size, missing)
    end
    
    save_results_registry!(registry)
end

function update_experiment_data_path!(exp_id::String, data_path::String)
    registry = load_experiment_registry()
    idx = findfirst(==(exp_id), registry.experiment_id)
    registry[idx, :data_path] = data_path
    save_experiment_registry!(registry)
end

# ============================================================================
# QUERY UTILITIES
# ============================================================================

"""
Get full joined view of experiments and results.
"""
function get_full_results()
    exp_reg = load_experiment_registry()
    res_reg = load_results_registry()
    
    return leftjoin(res_reg, exp_reg, on=:experiment_id)
end

"""
Get summary: best result per experiment across all kernels.
"""
function summarize_experiments()
    df = get_full_results()
    
    # Only completed results
    completed = filter(row -> row.status == :completed, df)
    
    if nrow(completed) == 0
        return DataFrame()
    end
    
    # Group by experiment, find best test accuracy
    gdf = groupby(completed, :experiment_id)
    
    return combine(gdf) do sdf
        best_idx = argmax(sdf.best_test_acc)
        (
            best_kernel = sdf.kernel_id[best_idx],
            best_test_acc = sdf.best_test_acc[best_idx],
            best_train_acc = sdf.best_train_acc[best_idx],
            n_kernels_tested = nrow(sdf),
            dataset_type = sdf.dataset_type[1],
            n_features = sdf.n_features[1]
        )
    end
end

"""
Compare kernels across datasets.
"""
function compare_kernels()
    df = filter(row -> row.status == :completed, get_full_results())
    
    if nrow(df) == 0
        return DataFrame()
    end
    
    gdf = groupby(df, :kernel_id)
    
    return combine(gdf, 
        :best_test_acc => mean => :mean_test_acc,
        :best_test_acc => std => :std_test_acc,
        :best_train_acc => mean => :mean_train_acc,
        nrow => :n_experiments
    )
end

"""
Get results for specific dataset type.
"""
function get_dataset_results(dataset_type::String)
    df = filter(row -> row.dataset_type == dataset_type, get_full_results())
    return filter(row -> row.status == :completed, df)
end

"""
Find experiments by kernel and dataset type.
"""
function find_experiments(; experiment_id=nothing, kernel_id=nothing, dataset_type=nothing, status=:completed)
    df = get_full_results()

    if !isnothing(experiment_id)
        df = filter(row -> row.experiment_id == experiment_id, df)
    end
    
    if !isnothing(kernel_id)
        df = filter(row -> row.kernel_id == kernel_id, df)
    end
    
    if !isnothing(dataset_type)
        df = filter(row -> row.dataset_type == dataset_type, df)
    end
    
    if !isnothing(status)
        df = filter(row -> row.status == status, df)
    end
    
    return df
end

"""
Get top N experiments by test accuracy.
"""
function top_experiments(n::Int=10)
    df = filter(row -> row.status == :completed, get_full_results())
    
    if nrow(df) == 0
        return DataFrame()
    end
    
    sort!(df, :best_test_acc, rev=true)
    return first(df, min(n, nrow(df)))
end

"""
Get experiment statistics.
"""
function experiment_stats()
    exp_reg = load_experiment_registry()
    res_reg = load_results_registry()
    
    status_counts = combine(groupby(res_reg, :status), nrow => :count)
    
    return Dict(
        :total_experiments => nrow(exp_reg),
        :total_kernels_tested => nrow(res_reg),
        :status_breakdown => status_counts,
        :unique_datasets => length(unique(exp_reg.dataset_type)),
        :completed_results => sum(res_reg.status .== :completed)
    )
end

"""
Load learning curves for specific result.
"""
function load_curves(exp_id::String, kernel_id::String)
    res_reg = load_results_registry()
    mask = (res_reg.experiment_id .== exp_id) .& (res_reg.kernel_id .== kernel_id)
    idx = findfirst(mask)
    
    if isnothing(idx)
        error("Result not found: $exp_id / $kernel_id")
    end
    
    curves_path = res_reg.curves_path[idx]
    
    if !isfile(curves_path)
        error("Curves file not found: $curves_path")
    end
    
    return load(curves_path, "learning_curves")
end

# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

export load_experiment_registry, load_results_registry
export save_experiment_registry!, save_results_registry!
export register_experiment!, register_kernel_result!
export update_kernel_result!, update_experiment_data_path!
export get_full_results, summarize_experiments, compare_kernels
export get_dataset_results, find_experiments, top_experiments
export experiment_stats, load_curves
