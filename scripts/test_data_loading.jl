using DrWatson
@quickactivate "TQK"

using TQK
using Statistics


using Statistics

"""
Test the balance of labels across multiple random seeds
"""
function test_label_balance(; n_trials=50, n_samples=1000, n_support_vectors=20, 
                            gamma=1.0, alpha_range=(0.1, 2.0), bias_range=(-1, 1),
                            feature_range=(0, 2π))
    
    positive_fractions = Float64[]
    
    for seed in 1:n_trials
        # Create config (adjust based on your actual DataConfig structure)
        data_params = RBFDataParams(
            n_support_vectors=n_support_vectors,
            feature_range=feature_range,
            gamma=gamma,
            alpha_range=alpha_range,
            bias_range=bias_range
        )
        config = DataConfig(n_samples=n_samples, data_params=data_params, seed=seed)
        
        # Generate data
        data = generate_pseudo_svm_dataset(config)
        
        # Calculate fraction of positive labels
        n_positive = sum(data[:y] .== 1)
        fraction_positive = n_positive / n_samples
        push!(positive_fractions, fraction_positive)
        
        # Print progress every 10 trials
        if seed % 10 == 0
            println("Seed $seed: $(round(fraction_positive * 100, digits=1))% positive")
        end
    end
    
    # Compute statistics
    println("\n" * "="^50)
    println("Label Balance Statistics (over $n_trials trials)")
    println("="^50)
    println("Mean fraction positive: $(round(mean(positive_fractions), digits=3))")
    println("Std deviation:          $(round(std(positive_fractions), digits=3))")
    println("Min fraction positive:  $(round(minimum(positive_fractions), digits=3))")
    println("Max fraction positive:  $(round(maximum(positive_fractions), digits=3))")
    println("\nExpected: ~0.5 for balanced data")
    println("Std should be ~$(round(sqrt(0.25/n_samples), digits=3)) for pure randomness")
    
    return positive_fractions
end


function test_label_balance(config::DataConfig; n_trials=50)
    
    positive_fractions = Float64[]
    
    for seed in 1:n_trials        
        # Generate data
        data = produce_data(config)
        
        # Calculate fraction of positive labels

        train_positive = sum(data[:y_train] .== 1)
        test_positive = sum(data[:y_test] .== 1)
        fraction_positive = (train_positive + test_positive)/ config.n_samples
        push!(positive_fractions, fraction_positive)
        
        # Print progress every 10 trials
        if seed % 10 == 0
            println("Seed $seed: $(round(fraction_positive * 100, digits=1))% positive")
        end
    end
    
    # Compute statistics
    println("\n" * "="^50)
    println("Label Balance Statistics (over $n_trials trials)")
    println("="^50)
    println("Mean fraction positive: $(round(mean(positive_fractions), digits=3))")
    println("Std deviation:          $(round(std(positive_fractions), digits=3))")
    println("Min fraction positive:  $(round(minimum(positive_fractions), digits=3))")
    println("Max fraction positive:  $(round(maximum(positive_fractions), digits=3))")
    println("\nExpected: ~0.5 for balanced data")
    println("Std should be ~$(round(sqrt(0.25/config.n_samples), digits=3)) for pure randomness")
    
    return positive_fractions
end


