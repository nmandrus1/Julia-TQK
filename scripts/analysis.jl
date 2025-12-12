using DrWatson
@quickactivate "TQK"
using DataFrames, Statistics, Plots

"""
    compare_methods_on_dataset(dataset_name::String; master_seed::Union{Int, UInt, Nothing}=nothing)

Finds all experiments that used the specified dataset (and optionally a specific seed)
and returns a DataFrame comparing their performance.
"""
function compare_methods_on_dataset(dataset_name::String; master_seed=nothing)
    # 1. Collect all results
    #    This loads the 'ExperimentArtifacts' struct from every saved file
    df = collect_results(datadir("experiments", "results"); subfolders=true)
    
    # 2. Filter by Dataset Name
    #    We access the nested config: artifact -> config -> data_config -> dataset_name
    filter!(row -> row.artifacts.config.data_config.dataset_name == dataset_name, df)
    
    # 3. Filter by Seed (Optional but recommended for strict "same data" checks)
    if !isnothing(master_seed)
        filter!(row -> row.artifacts.config.data_config.master_seed == master_seed, df)
    end
    
    if nrow(df) == 0
        @warn "No experiments found for dataset: $dataset_name (seed=$master_seed)"
        return DataFrame()
    end

    # 4. Extract Comparison Metrics
    #    We pull out the relevant fields to create a clean summary table
    comparison_df = DataFrame(
        Method = String[],
        Method_Type = String[],
        Hyperparams = String[],
        Test_Acc = Float64[],
        Train_Acc = Float64[],
        Best_KTA = Float64[]
    )

    for row in eachrow(df)
        art = row.artifacts
        method = art.config.method
        
        # Create a readable "Hyperparams" string (e.g., "Q=4, L=2")
        hparams = "N/A"
        if method isa ReuploadingMethod
            c = method.circuit_config
            hparams = "Q=$(c.n_qubits), L=$(c.n_layers), $(c.entanglement)"
        elseif method isa RBFMethod
            hparams = "Gamma Grid"
        end

        push!(comparison_df, (
            art.config.name,           # Method Name (Experiment Name)
            string(typeof(method)),    # Type (RBF, Reuploading, etc.)
            hparams,                   # Key differentiating params
            art.test_acc,
            art.train_acc,
            art.tuning_result.best_score
        ))
    end

    # 5. Sort by Performance
    sort!(comparison_df, :Test_Acc, rev=true)
    
    return comparison_df
end

"""
    plot_kernel_comparison(dataset_name::String)

Visualizes the performance of different kernels on the same dataset.
"""
function plot_kernel_comparison(dataset_name::String; seed=nothing)
    df = compare_methods_on_dataset(dataset_name; master_seed=seed)
    
    if nrow(df) == 0 return end

    # Create a Bar Chart of Test Accuracy
    p = bar(
        df.Method, 
        df.Test_Acc, 
        title="Model Comparison on: $dataset_name",
        ylabel="Test Accuracy",
        xlabel="Kernel Method",
        rotation=45,
        legend=false,
        ylims=(0, 1.05)
    )
    
    return p
end
