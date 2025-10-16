

using DrWatson
@quickactivate "TQK"

using TQK
using Random
using Printf
using LIBSVM
using Statistics
using Plots
using Printf


# Assume your existing DataConfig and generate_pseudo_svm_dataset_fixed are defined here

function test_reupload_data_gen_stats(trials::Int = 1000)
    # 1. Initialize an array to store the count of positive labels for each trial
    positive_counts = Int[]

    for seed in rand(Int64, trials)
        data_config = DataConfig(
            n_samples=100,
            n_features=2,
            data_params=ReuploadingDataParams(
                                    n_qubits=6,
                                    n_layers=4,
                                    entanglement=linear,
                                    n_support_vectors=10
                                ),
            seed=seed
        )
        data = generate_reupload_data(data_config)
        @info data[:bias]
        
        # 2. Count positive labels and push the result to our array
        cnt = count(y -> y == 1, data[:y])
        push!(positive_counts, cnt)
    end

    # 3. Calculate descriptive statistics on the collected counts
    avg_pos = mean(positive_counts)
    std_dev = std(positive_counts)
    med_pos = median(positive_counts)
    min_pos = minimum(positive_counts)
    max_pos = maximum(positive_counts)

    # 4. Print the summary
    println("📊 Statistics for Positive Label Counts over $trials trials:")
    @printf "  - Average:  %.2f\n" avg_pos
    @printf "  - Std Dev:  %.2f\n" std_dev
    @printf "  - Median:   %d\n" med_pos
    @printf "  - Min | Max: %d | %d\n\n" min_pos max_pos    
end


function plot_classification_data(data::Dict)
   X = data[:X]'
   y = data[:y]
   
   # Create the plot
   p = plot(legend=:best, xlabel="Feature 1", ylabel="Feature 2")
   
   # Train data: circles for class -1, squares for class 1
   train_mask_neg = y .== -1
   train_mask_pos = y .== 1
   scatter!(p, X[train_mask_neg, 1], X[train_mask_neg, 2], 
            color=:blue, marker=:circle, label="Train (−1)", markersize=6)
   scatter!(p, X[train_mask_pos, 1], X[train_mask_pos, 2], 
            color=:red, marker=:circle, label="Train (+1)", markersize=6)
   
   return p
end
