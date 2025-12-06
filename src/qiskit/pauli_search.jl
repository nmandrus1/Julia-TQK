using PythonCall
using Random
using Statistics



"""
    evaluate_kernel_cv(K, y, C, cv_folds)

Performs cross-validation on an already computed kernel matrix.
"""
function evaluate_kernel_cv(K::AbstractMatrix, y::AbstractVector, C::Float64, cv_folds::Int)
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    numpy = pyimport("numpy")
    
    try
        clf = svm_module.SVC(kernel="precomputed", C=C)
        scores = model_selection.cross_val_score(clf, numpy.asarray(K), numpy.asarray(y), cv=cv_folds)
        return mean(pyconvert(Vector{Float64}, scores))
    catch e
        @warn "CV failed for a configuration" exception=e
        return -Inf
    end
end

# --- MULTI-STAGE SEARCH FUNCTIONS ---

"""
    find_best_proxy_C(config, X_sub, y_sub) -> Float64

STAGE 1: Search for a single, good proxy C value on a subset of data.
"""
function find_best_proxy_C(config::ExperimentConfig{<:DataParams, <:PauliKernelHyperparameterSearchConfig}, X_sub::AbstractMatrix, y_sub::AbstractVector)
    kernel_config = config.kernel_config
    scores_by_C = Dict(C => [] for C in config.c_ranges)
    
    @info "Starting Stage 1: Finding proxy C on a subset of data" n_samples=size(X_sub, 1)
    
    # Run a small number of random Pauli searches
    for i in 1:10 # A small, fixed number of iterations is sufficient
        hyperparams = PauliKernelHyperparameters(
            paulis = generate_constrained_pauli_set(kernel_config.search_constraints, seed=config.seed + i),
            reps = rand(kernel_config.reps),
            entanglement = rand(kernel_config.entanglement),
            # NOTE: Needed for complete definition of this config,
            # but this value of C is never used
            C=1.0
        )
        
        K_sub = compute_pauli_kernel_matrix(hyperparams, X_sub)
        
        for C in config.c_ranges
            score = evaluate_kernel_cv(K_sub, y_sub, C, config.cv_folds)
            push!(scores_by_C[C], score)
        end
    end
    
    # Average the scores for each C and pick the best
    avg_scores = Dict(C => mean(filter(!isinf, scores)) for (C, scores) in scores_by_C)
    proxy_C = findmax(avg_scores)[2]
    
    @info "Stage 1 complete. Best proxy C found." proxy_C=proxy_C avg_scores=avg_scores
    return proxy_C
end


"""
    tune_final_C(best_pauli_config, K_full, y_train, config) -> (Float64, Float64)

STAGE 3: Fine-tune C for the best kernel using a cached kernel matrix.
"""
function tune_final_C(best_pauli_config, K_full::AbstractMatrix, y_train::AbstractVector, config::ExperimentConfig)
    @info "Starting Stage 3: Fine-tuning C for the best kernel"
    
    best_C = config.c_ranges[1]
    best_score = -Inf
    
    for C in config.c_ranges
        score = evaluate_kernel_cv(K_full, y_train, C, config.cv_folds)
        if score > best_score
            best_score = score
            best_C = C
        end
    end
    
    @info "Stage 3 complete. Final C chosen." best_C=best_C final_score=best_score
    return best_C, best_score
end


"""
    search_pauli_hyperparameters(config, X_train, y_train)

The main three-stage search orchestrator.
"""
function search_pauli_hyperparameters(
    config::ExperimentConfig{D, K},
    X_train::AbstractMatrix,
    y_train::AbstractVector
) where {D, K<:PauliKernelHyperparameterSearchConfig}
    
    kernel_config = config.kernel_config
    Random.seed!(config.seed)
    
    # --- STAGE 1: Find Proxy C ---
    subset_size = min(400, size(X_train, 1)) # Use up to 400 samples
    X_sub = X_train[1:subset_size, :]
    y_sub = y_train[1:subset_size]
    proxy_C = find_best_proxy_C(config, X_sub, y_sub)
    
    # --- STAGE 2: Main Pauli Search with Proxy C ---
    @info "Starting Stage 2: Main Pauli search with fixed proxy C" proxy_C=proxy_C
    best_score_proxy = -Inf
    best_pauli_config = nothing

    for iter in 1:kernel_config.n_search_iterations
        hyperparams = PauliKernelHyperparameters(
            paulis = generate_constrained_pauli_set(kernel_config.search_constraints, seed=config.seed + iter),
            reps = rand(kernel_config.reps),
            entanglement = rand(kernel_config.entanglement),
            C=proxy_C
        )
        
        @info hyperparams.paulis

        K_full = compute_pauli_kernel_matrix(hyperparams, X_train)
        score = evaluate_kernel_cv(K_full, y_train, proxy_C, config.cv_folds)
        
        if score > best_score_proxy
            best_score_proxy = score
            best_pauli_config = (paulis=hyperparams.paulis, reps=hyperparams.reps, entanglement=hyperparams.entanglement, K_cache=K_full)
        end

        if iter % 5 == 0
            @info "Search progress" iteration=iter best_score_with_proxy_C=best_score_proxy
        end
    end
    
    @info "Stage 2 complete. Best Pauli configuration found." best_config=best_pauli_config
    
    # --- STAGE 3: Fine-Tune C for the Best Kernel ---
    final_C, final_score = tune_final_C(best_pauli_config, best_pauli_config.K_cache, y_train, config)
    
    # --- Finalize and Return ---
    final_hyperparams = PauliKernelHyperparameters(
        n_qubits=kernel_config.n_qubits,
        paulis=best_pauli_config.paulis,
        reps=best_pauli_config.reps,
        entanglement=best_pauli_config.entanglement,
        C=final_C,
        seed=config.seed
    )

    @info "Pauli search fully complete!"
    return final_hyperparams, final_score
end
