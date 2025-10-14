using Statistics
using PythonCall
using LinearAlgebra

"""
    pairwise_distances_colmajor(X)

Compute pairwise squared Euclidean distances for column-major data.
X is (n_features × n_samples).
"""
function pairwise_distances_colmajor(X::AbstractMatrix)
    n = size(X, 2)
    D = zeros(n, n)
    for i in 1:n
        for j in i:n
            d = sum(abs2, X[:, i] - X[:, j])
            D[i,j] = D[j,i] = d
        end
    end
    return D
end



"""
    evaluate_rbf_cv(gamma, X, y, C, cv_folds)

Evaluate RBF kernel with given gamma using cross-validation.
"""
function evaluate_rbf_cv(gamma::Float64, X::AbstractMatrix, y::AbstractVector,
                         C::Float64, cv_folds::Int)
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    np = pyimport("numpy")
    
    # X is expected to be (n_samples, n_features), which is what scikit-learn wants.
    # Let scikit-learn compute the RBF kernel directly instead of pre-computing in Julia.
    
    # 1. Create an SVC classifier that knows how to compute the RBF kernel itself.
    clf = svm_module.SVC(kernel="rbf", gamma=gamma, C=C)
    
    # 2. Pass the feature matrix X directly, not the precomputed kernel K.
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    
    scores = model_selection.cross_val_score(clf, X_np, y_np, cv=cv_folds)
    
    return mean(pyconvert(Vector{Float64}, scores))
end

"""
    search_rbf_hyperparameters(config, X_train, y_train)

Search for best RBF kernel hyperparameters (gamma and C) using CV.
Returns (best_hyperparameters, best_score).
"""
function search_rbf_hyperparameters(
    config::ExperimentConfig{D, K},
    X_train::AbstractMatrix,
    y_train::AbstractVector
) where {D, K<:RBFKernelHyperparameterSearchConfig}
    
    kernel_config = config.kernel_config
    gamma_range = kernel_config.gamma_range
    
    @info "Starting RBF kernel search with CV" gamma_range=gamma_range
    
    best_gamma = gamma_range[1]
    best_C = config.c_ranges[1]
    best_score = -Inf
    
    # Grid search over gamma and C
    for gamma in gamma_range
        for C in config.c_ranges
            score = evaluate_rbf_cv(gamma, X_train, y_train, C, config.cv_folds)
            
            @info "Evaluated" gamma=gamma C=C cv_score=score
            
            if score > best_score
                best_score = score
                best_gamma = gamma
                best_C = C
            end
        end
    end
    
    @info "Search complete" best_gamma=best_gamma best_C=best_C best_score=best_score
    
    best_hyperparams = RBFHyperparameters(
        gamma=best_gamma,
        C=best_C,
        seed=config.seed
    )
    
    return best_hyperparams, best_score
end

"""
    tune_rbf_C(gamma, X, y, C_range, cv_folds)

Tune the SVM C parameter for given RBF gamma.
"""
function tune_rbf_C(gamma::Float64, X::AbstractMatrix, y::AbstractVector,
                    C_range::Vector{Float64}, cv_folds::Int)
    
    best_C = C_range[1]
    best_score = -Inf
    
    @info "Tuning C parameter for RBF" gamma=gamma C_range=C_range
    
    for C in C_range
        score = evaluate_rbf_cv(gamma, X, y, C, cv_folds)
        
        if score > best_score
            best_score = score
            best_C = C
        end
        
        @info "Evaluated C" C=C cv_score=score
    end
    
    @info "C tuning complete" best_C=best_C best_score=best_score
    
    return best_C
end
