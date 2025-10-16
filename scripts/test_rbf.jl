
using DrWatson
@quickactivate "TQK"

using TQK
using Random
using Printf
using LIBSVM
using Statistics
using Plots
using Printf

"""
Test RBF kernel hyperparameter search on RBF-generated data.
"""
function test_rbf_search(;seed::Union{Int, Nothing}=nothing)
    println("="^70)
    println("RBF KERNEL HYPERPARAMETER SEARCH TEST")
    println("="^70)
    
    if isnothing(seed)
        seed = 42
    end
    Random.seed!(seed)
    
    # Generate RBF-separable data
    println("\n1. Generating RBF-separable test data...")
    data_config = DataConfig(
        n_samples=400,
        n_features=2,
        test_size=0.3,
        data_params=RBFDataParams(
            gamma=1.0,  # True gamma
            n_support_vectors=50,
            alpha_range=(0.1, 2.0),
            bias_range=(-1.0, 1.0),
            feature_range=(0.0, 2π)
        ),
        seed=seed
    )
    
    data = produce_data(data_config)
    X_train = data[:X_train]
    y_train = data[:y_train]
    X_test = data[:X_test]
    y_test = data[:y_test]
    
    println("  Train samples: $(length(y_train))")
    println("  Test samples: $(length(y_test))")
    println("  Features: $(size(X_train, 1))")
    println("  True gamma: $(data_config.data_params.gamma)")
    
    # Create experiment configuration for search
    println("\n2. Setting up RBF kernel search...")
    exp_config = ExperimentConfig(
        experiment_name="rbf_search_test",
        data_config=data_config,
        kernel_config=RBFKernelHyperparameterSearchConfig(
            gamma_range=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        ),
        learning_curve_sizes=Int[],
        c_ranges=[0.1, 1.0, 10.0, 100.0],
        cv_folds=3,
        seed=seed
    )
    
    println("  Gamma values to test: $(exp_config.kernel_config.gamma_range)")
    println("  C values to test: $(exp_config.c_ranges)")
    
    # Run the search
    println("\n3. Running hyperparameter search...")
    best_hyperparams, best_score = search_rbf_hyperparameters(
        exp_config, X_train, y_train
    )
    
    println("\n4. Search Results:")
    println("  Best KTA score: $(round(best_score, digits=4))")
    println("  Best gamma: $(best_hyperparams.gamma)")
    println("  Best C: $(best_hyperparams.C)")
    println("  True gamma: $(data_config.data_params.gamma)")
    
    # Check if we found the correct gamma (or close to it)
    gamma_ratio = best_hyperparams.gamma / data_config.data_params.gamma
    if 0.5 < gamma_ratio < 2.0
        println("  ✓ Found gamma within 2x of true value")
    end
    
    # Evaluate on test set
    println("\n5. Evaluating on test set...")
    
    # Compute kernel matrices
    D_train = pairwise_distances_colmajor(X_train)
    K_train = exp.(-best_hyperparams.gamma .* D_train)
    
    D_test = zeros(size(X_test, 2), size(X_train, 2))
    for i in 1:size(X_test, 2)
        for j in 1:size(X_train, 2)
            D_test[i, j] = sum(abs2, X_test[:, i] - X_train[:, j])
        end
    end
    K_test = exp.(-best_hyperparams.gamma .* D_test)
    
    # Train SVM
    model = svmtrain(K_train, y_train, kernel=Kernel.Precomputed, 
                     cost=best_hyperparams.C, verbose=false)
    
    # Predictions
    train_pred = svmpredict(model, K_train)[1]
    test_pred = svmpredict(model, K_test')[1]
    
    train_acc = mean(train_pred .== y_train)
    test_acc = mean(test_pred .== y_test)
    
    println("  Train accuracy: $(round(train_acc * 100, digits=1))%")
    println("  Test accuracy: $(round(test_acc * 100, digits=1))%")
    
    # Compare with true gamma
    println("\n6. Comparison with true gamma...")
    D_train_true = pairwise_distances_colmajor(X_train)
    K_train_true = exp.(-data_config.data_params.gamma .* D_train_true)
    
    D_test_true = zeros(size(X_test, 2), size(X_train, 2))
    for i in 1:size(X_test, 2)
        for j in 1:size(X_train, 2)
            D_test_true[i, j] = sum(abs2, X_test[:, i] - X_train[:, j])
        end
    end
    K_test_true = exp.(-data_config.data_params.gamma .* D_test_true)
    
    model_true = svmtrain(K_train_true, y_train, kernel=Kernel.Precomputed, 
                         cost=best_hyperparams.C, verbose=false)
    test_pred_true = svmpredict(model_true, K_test_true')[1]
    test_acc_true = mean(test_pred_true .== y_test)
    
    println("  Test accuracy with TRUE gamma: $(round(test_acc_true * 100, digits=1))%")
    println("  Accuracy difference: $(round((test_acc - test_acc_true) * 100, digits=1))%")
    
    # Create visualization
    println("\n7. Creating visualization...")
    
    # Plot data
    p_data = scatter(X_train[1, y_train.==1], X_train[2, y_train.==1],
                     color=:red, marker=:circle, label="Train (+1)", alpha=0.6)
    scatter!(p_data, X_train[1, y_train.==-1], X_train[2, y_train.==-1],
             color=:blue, marker=:circle, label="Train (-1)", alpha=0.6)
    scatter!(p_data, X_test[1, y_test.==1], X_test[2, y_test.==1],
             color=:red, marker=:diamond, label="Test (+1)", markersize=5)
    scatter!(p_data, X_test[1, y_test.==-1], X_test[2, y_test.==-1],
             color=:blue, marker=:diamond, label="Test (-1)", markersize=5)
    plot!(p_data, xlabel="Feature 1", ylabel="Feature 2",
          title="RBF-Separable Test Data", legend=:best)
    
    # Plot KTA scores vs gamma
    gamma_vals = exp_config.kernel_config.gamma_range
    kta_scores = [evaluate_rbf_kta(g, X_train, y_train) for g in gamma_vals]
    
    p_scores = plot(gamma_vals, kta_scores, 
                    xlabel="Gamma", ylabel="KTA Score",
                    title="RBF Kernel: KTA vs Gamma",
                    xscale=:log10, marker=:circle, label="KTA Score",
                    lw=2, markersize=6)
    vline!(p_scores, [data_config.data_params.gamma], 
           label="True γ", linestyle=:dash, color=:red, lw=2)
    vline!(p_scores, [best_hyperparams.gamma],
           label="Best γ", linestyle=:dash, color=:green, lw=2)
    
    println("\n" * "="^70)
    println("TEST COMPLETE")
    println("="^70)
    println("\nSummary:")
    println("  ✓ Tested $(length(gamma_vals)) gamma values")
    println("  ✓ Found best gamma = $(best_hyperparams.gamma)")
    println("  ✓ Test accuracy: $(round(test_acc * 100, digits=1))%")
    
    if test_acc > 0.8
        println("\nPASS: Test accuracy > 80%")
    else
        println("\nWARNING: Test accuracy < 80%")
    end
    
    return Dict(
        :best_hyperparams => best_hyperparams,
        :best_score => best_score,
        :train_accuracy => train_acc,
        :test_accuracy => test_acc,
        :test_accuracy_true_gamma => test_acc_true,
        :plot_data => p_data,
        :plot_scores => p_scores,
        :data => data
    )
end

# Run the test if executed as main script
if abspath(PROGRAM_FILE) == @__FILE__
    results = test_rbf_search()
end




# Assume your existing DataConfig and generate_pseudo_svm_dataset_fixed are defined here

function test_rbf_data_gen_stats(trials::Int = 1000)
    # 1. Initialize an array to store the count of positive labels for each trial
    positive_counts = Int[]

    for seed in rand(Int64, trials)
        data_config = DataConfig(
            n_samples=100,
            n_features=2,
            data_params=RBFDataParams(gamma=2.0, n_support_vectors=10),
            seed=seed
        )
        data = generate_pseudo_svm_dataset_fixed(data_config)
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
