using DrWatson
@quickactivate "TQK"

using DataFrames
using TQK 
using StatsBase
using Printf

# 1. Collect Results
# dataframe of all ReuploadingMethod results
reup_df = collect_results(datadir("experiments", "results", "ReuploadingMethod"), subfolders=true)

# 2. Filter for Pilot Experiments
# dataframe of all reuploading experiments that used pilot test dataset
pilot_results = filter(row -> occursin("pilot_teacher", row[:artifacts].config.data_config.dataset_name), reup_df)

println("Found $(nrow(pilot_results)) pilot experiments.")
println("=====================================================================================================")
@printf("%-40s | %-10s %-10s | %-15s %-15s | %-10s\n", 
        "Experiment Name", "Qubits", "Layers", "Avg Overlap", "Expected (1/D)", "Error")
println("-----------------------------------------------------------------------------------------------------")

for exp in eachrow(pilot_results)
    # Experiment Artifact object
    artifacts = exp[:artifacts]
    exp_config = artifacts.config
    data_config = exp_config.data_config
    method = exp_config.method
    
    # Extract Kernel Matrix
    K_train = artifacts.K_train
    n_samples = size(K_train, 1)

    # Calculate Average Off-Diagonal Kernel Value (The "Overlap")
    # We exclude the diagonal because K(x,x) is always 1.0
    cross_terms = [K_train[i, j] for i in 1:n_samples for j in 1:n_samples if i != j]
    avg_K_train_cross_term = mean(cross_terms)

    # Extract Circuit Parameters
    # We assume the method is ReuploadingMethod based on the folder we loaded
    if method isa ReuploadingMethod
        n_qubits = method.circuit_config.n_qubits
        n_layers = method.circuit_config.n_layers
        
        # Calculate Hypersphere Dimension D
        D = 2^n_qubits
        expected_val = 1/D
        
        # Print Comparison
        name = exp_config.name
        error = abs(avg_K_train_cross_term - expected_val)
        
        @printf("%-40s | %-10d %-10d | %-15.6f %-15.6f | %-10.6f\n", 
                name, n_qubits, n_layers, avg_K_train_cross_term, expected_val, error)
    end
end
println("=====================================================================================================")
