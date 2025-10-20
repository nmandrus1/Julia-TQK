
using DrWatson
@quickactivate "TQK"

using DataFrames
using Statistics
using Plots # For plotting
using JLD2  # For loading saved result files

# --- REGISTRY & QUERY FUNCTIONS ---
# This file should contain all your registry loading and query functions
# (load_experiment_registry, load_results_registry, get_full_results, etc.)
include(scriptsdir("experiment_registry_refactored.jl"))


# --- HELPER FUNCTIONS ---

"""
Load hyperparameters for a specific result.
"""
function load_hyperparams(exp_id::String, kernel_id::String)
    res_reg = load_results_registry()
    mask = (res_reg.experiment_id .== exp_id) .& (res_reg.kernel_id .== kernel_id)
    idx = findfirst(mask)
    
    if isnothing(idx)
        error("Result not found: $exp_id / $kernel_id")
    end
    
    path = res_reg.hyperparams_path[idx]
    
    if !isfile(path)
        error("Hyperparameters file not found: $path")
    end
    
    # The saved file is a dict with keys "hyperparams" and "score"
    return load(path, "hyperparams")
end


# --- PLOTTING FUNCTIONS ---

# --- PLOTTING FUNCTIONS ---

"""
    plot_experiment_summary(experiment_id::String; top_n_reup::Int = 1)

Generates a comprehensive 3-panel plot for a specific experiment run.

It automatically finds the best-performing Reuploading kernels for the given 
`experiment_id` and uses them for the detailed plots.

# Arguments
- `experiment_id::String`: The ID of the experiment to plot.
- `top_n_reup::Int`: The number of top-performing reuploading kernels to display
  on the learning curve and loss plots. Defaults to 1.

# Plots
1.  **Learning Curves**: Compares the learning curves of the top `n` reuploading 
    kernels against the RBF and Pauli baselines for the same dataset.
2.  **Optimization Loss**: Shows the training loss curves for each of the top `n` 
    quantum kernels.
3.  **Final Accuracy Comparison**: A bar chart of the best test accuracies for 
    **all** kernels tested on this dataset.
"""
function plot_experiment_summary(experiment_id::String; top_n_reup::Int = 1)
    
    # --- Find the top N reuploading kernels for this experiment ---
    all_results_for_exp = find_experiments(experiment_id=experiment_id)
    
    reup_results = filter(row -> contains(row.kernel_id, "reup"), all_results_for_exp)

    if isempty(reup_results)
        @error "No reuploading kernels found for experiment '$experiment_id'. Cannot generate plot."
        return
    end

    # Sort by test accuracy and take the top N
    sort!(reup_results, :best_test_acc, rev=true)
    num_to_plot = min(top_n_reup, nrow(reup_results))
    top_kernels_df = first(reup_results, num_to_plot)
    kernels_to_plot = top_kernels_df.kernel_id
    
    @info "Plotting top $num_to_plot reuploading kernels: $kernels_to_plot"

    # --- Plot 1: Learning Curves ---
    p1 = plot(xlabel="Training Samples", ylabel="Accuracy", title="Learning Curves for $experiment_id", legend=:bottomright)
    
    # Combine top kernels with baselines for plotting
    all_curve_ids = vcat(kernels_to_plot, ["rbf", "pauli"])

    for kid in all_curve_ids
        try
            curves = load_curves(experiment_id, kid)
            sizes = sort(collect(keys(curves)))
            test_accs = [curves[s][:test_acc] for s in sizes]
            plot!(p1, sizes, test_accs, label=kid, marker=:circle, markersize=3)
        catch e
            @warn "Could not load learning curves for '$kid'. Skipping from plot."
        end
    end

    # --- Plot 2: Optimization Loss Curves ---
    p2 = plot(xlabel="Iteration", ylabel="Loss", title="Optimization Loss for Top Kernels", legend=:topright)
    
    if isempty(kernels_to_plot)
         annotate!(p2, 0.5, 0.5, text("No trainable kernels to plot", 10, :center))
    else
        for kid in kernels_to_plot
            try
                hyperparams = load_hyperparams(experiment_id, kid)
                if hasfield(typeof(hyperparams), :loss) && !isempty(hyperparams.loss)
                    plot!(p2, hyperparams.loss, lw=2, label=kid)
                else
                    # Add a placeholder line to show in the legend that data is missing
                    plot!(p2, [NaN], label="$kid (no loss data)", linestyle=:dash)
                end
            catch e
                @warn "Could not load hyperparameters to plot loss for $kid."
                plot!(p2, [NaN], label="$kid (file not found)", linestyle=:dash)
            end
        end
    end
    
    # --- Plot 3: Final Accuracy Comparison ---
    all_results = find_experiments(experiment_id=experiment_id)
    sort!(all_results, :best_test_acc, rev=true)
    
    p3 = bar(all_results.kernel_id, all_results.best_test_acc,
             title="Final Test Accuracy Comparison (All Kernels)",
             ylabel="Best Test Accuracy",
             xrotation=45,
             legend=false,
             ylims=(0, 1.05 * maximum(skipmissing(all_results.best_test_acc))))
             
    # --- Combine and Display ---
    return plot(p1, p2, p3, layout=(3, 1), size=(900, 1200), margin=10Plots.mm)
end

"""
    plot_reuploading_kernel_comparison(; kwargs...)

Compares the top 5 Reuploading kernels for a specific dataset against baselines.

Finds the dataset matching the keyword arguments, then generates a bar chart
of the best test accuracies.

# Example
# Compare top kernels for the dataset with 4 features and seed 42
plot_reuploading_kernel_comparison(n_features=4, data_seed=42)
"""
function plot_reuploading_kernel_comparison(; kwargs...)
    # Find the target dataset's experiment_id
    exp_reg = load_experiment_registry()
    for (key, value) in kwargs
        if hasproperty(exp_reg, key)
            exp_reg = filter(row -> row[key] == value, exp_reg)
        end
    end
    
    if nrow(exp_reg) == 0
        return "No dataset found matching criteria: $(kwargs)"
    end
    target_exp_id = first(exp_reg.experiment_id)

    # Get all results for that dataset
    all_results = find_experiments(experiment_id=target_exp_id)
    if isempty(all_results) return "No completed results for this dataset." end
    
    # Separate reuploading kernels from baselines
    reup_results = filter(row -> startswith(row.kernel_id, "reup"), all_results)
    baselines = filter(row -> row.kernel_id in ["rbf", "pauli"], all_results)
    
    # Get top 5 reuploading kernels
    sort!(reup_results, :best_test_acc, rev=true)
    top_5_reup = first(reup_results, min(5, nrow(reup_results)))
    
    # Combine baselines and top 5 for plotting
    plot_data = vcat(baselines, top_5_reup)
    sort!(plot_data, :best_test_acc, rev=true)

    # Create the bar chart
    p = bar(plot_data.kernel_id, plot_data.best_test_acc,
            title="Top Reuploading Kernels for $target_exp_id",
            ylabel="Best Test Accuracy",
            xrotation=45,
            legend=false,
            size=(800, 600),
            margin=10Plots.mm,
            ylims=(0, 1.05 * maximum(skipmissing(plot_data.best_test_acc))))
            
    return p
end

# Export the new plotting functions
export plot_experiment_summary, plot_reuploading_kernel_comparison
