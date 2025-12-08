using LIBSVM
using MLUtils
using Statistics
using StableRNGs

"""
    tune_svm_c(K_train::AbstractMatrix, y_train::AbstractVector, c_grid::Vector{Float64}; cv_folds=5, seed=42)

Finds the best C parameter using Cross-Validation on a precomputed kernel matrix.
"""
function tune_svm_c(K_train::AbstractMatrix, y_train::AbstractVector, c_grid::Vector{Float64}; cv_folds=5, seed=42)
    best_c = c_grid[1]
    best_acc = -1.0
    results = Dict{Float64, Float64}()
    rng = StableRNG(seed=seed)

    # Pre-compute CV folds to ensure consistency across C values
    # We use MLUtils.kfolds or similar manual indexing for precomputed kernels
    n_samples = length(y_train)
    fold_indices = collect(MLUtils.kfolds(n_samples, k=cv_folds, shuffle=true, rng=rng))

    for c in c_grid
        fold_accuracies = Float64[]
        
        for (train_idx, val_idx) in fold_indices
            # Slice Kernel Matrix: K[train, train] for training, K[val, train] for prediction
            k_sub_train = K_train[train_idx, train_idx]
            k_sub_val   = K_train[val_idx, train_idx]
            
            y_sub_train = y_train[train_idx]
            y_sub_val   = y_train[val_idx]
            
            # Train SVM
            model = svmtrain(k_sub_train, y_sub_train, kernel=Kernel.Precomputed, cost=c, verbose=false)
            
            # Predict
            # LIBSVM expects (n_features, n_samples). For precomputed, features = support vectors (train samples)
            # So input is K[val, train]' -> (n_train, n_val)
            y_pred, _ = svmpredict(model, k_sub_val')
            
            acc = mean(y_pred .== y_sub_val)
            push!(fold_accuracies, acc)
        end
        
        avg_acc = mean(fold_accuracies)
        results[c] = avg_acc
        
        if avg_acc > best_acc
            best_acc = avg_acc
            best_c = c
        end
    end
    
    return best_c, best_acc, results
end
