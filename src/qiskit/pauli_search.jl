using PythonCall
using Random
using Statistics

"""
    evaluate_pauli_config_kta(paulis, reps, entanglement, n_qubits, X, y)

Evaluate a Pauli configuration using Kernel Target Alignment (KTA).
Returns the KTA score (higher is better).
"""
function evaluate_pauli_config_kta(paulis::Vector{String}, reps::Int, 
                                    entanglement::String, n_qubits::Int,
                                    X::AbstractMatrix, y::AbstractVector)
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    # X is column-major (n_features × n_samples), Qiskit needs row-major
    X_row = X'
    
    try
        feature_map = qiskit_lib.PauliFeatureMap(
            feature_dimension=size(X, 1),
            reps=reps,
            entanglement=entanglement,
            paulis=pylist(paulis)
        )
        
        kernel = qiskit_kernels.FidelityStatevectorKernel(feature_map=feature_map)
        K = pyconvert(Matrix{Float64}, kernel.evaluate(x_vec=X_row))
        
        # Compute KTA
        y_outer = y * y'
        kta = sum(K .* y_outer) / (sqrt(sum(K.^2)) * sqrt(sum(y_outer.^2)))
        
        return kta
    catch e
        @warn "Failed to evaluate Pauli config" paulis exception=e
        return -Inf
    end
end

"""
    evaluate_pauli_config_cv(paulis, reps, entanglement, n_qubits, X, y, C, cv_folds)

Evaluate a Pauli configuration using cross-validation with precomputed kernel SVM.
Returns mean CV accuracy.
"""
function evaluate_pauli_config_cv(paulis::Vector{String}, reps::Int,
                                   entanglement::String, n_qubits::Int,
                                   X::AbstractMatrix, y::AbstractVector,
                                   C::Float64, cv_folds::Int)
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    numpy = pyimport("numpy")
    
    X_row = X'
    
    try
        feature_map = qiskit_lib.PauliFeatureMap(
            feature_dimension=size(X, 1),
            reps=reps,
            entanglement=entanglement,
            paulis=pylist(paulis)
        )
       
        kernel = qiskit_kernels.FidelityStatevectorKernel(feature_map=feature_map)
        K_py = kernel.evaluate(x_vec=X_row)
        K_np = numpy.asarray(K_py)
        y_np = numpy.asarray(y)
       
        # Cross-validation with precomputed kernel
        clf = svm_module.SVC(kernel="precomputed", C=C)
        scores = model_selection.cross_val_score(clf, K_np, y_np, cv=cv_folds)
        
        return mean(pyconvert(Vector{Float64}, scores))
    catch e
        @warn "Failed CV for Pauli config" paulis exception=e
        return -Inf
    end
end

"""
    search_pauli_hyperparameters(config, X_train, y_train)

Search for best Pauli kernel hyperparameters using random search with CV.
Returns (best_hyperparameters, best_score).
"""
function search_pauli_hyperparameters(
    config::ExperimentConfig{D, K},
    X_train::AbstractMatrix,
    y_train::AbstractVector
) where {D, K<:PauliKernelHyperparameterSearchConfig}
    
    kernel_config = config.kernel_config
    Random.seed!(config.seed)
    
    best_score = -Inf
    best_hyperparams = nothing
    all_results = []
    
    @info "Starting Pauli kernel search with CV" n_iterations=kernel_config.n_search_iterations
    
    for iter in 1:kernel_config.n_search_iterations
        # Generate random configuration
        paulis = generate_constrained_pauli_set(
            kernel_config.search_constraints,
            seed=config.seed + iter
        )
        reps = rand(kernel_config.reps)
        entanglement = rand(kernel_config.entanglement)
        
        # Try each C value and pick best
        best_C_score = -Inf
        best_C = config.c_ranges[1]
        
        for C in config.c_ranges
            score = evaluate_pauli_config_cv(
                paulis, reps, entanglement,
                kernel_config.n_qubits,
                X_train, y_train, C, config.cv_folds
            )
            
            if score > best_C_score
                best_C_score = score
                best_C = C
            end
        end
        
        push!(all_results, (paulis=paulis, reps=reps, 
                           entanglement=entanglement, C=best_C, score=best_C_score))
        
        if best_C_score > best_score
            best_score = best_C_score
            best_hyperparams = PauliKernelHyperparameters(
                n_qubits=kernel_config.n_qubits,
                paulis=paulis,
                reps=reps,
                entanglement=entanglement,
                C=best_C,
                seed=config.seed
            )
        end
        
        if iter % 5 == 0
            @info "Search progress" iteration=iter best_score=best_score
        end
    end
    
    @info "Search complete" best_score=best_score best_paulis=best_hyperparams.paulis best_C=best_hyperparams.C
    
    return best_hyperparams, best_score
end

"""
    tune_svm_C(hyperparams, X, y, C_range, cv_folds)

Tune the SVM C parameter for given kernel hyperparameters.
"""
function tune_svm_C(hyperparams::PauliKernelHyperparameters,
                    X::AbstractMatrix, y::AbstractVector,
                    C_range::Vector{Float64}, cv_folds::Int)
    
    best_C = C_range[1]
    best_score = -Inf
    
    @info "Tuning C parameter" C_range=C_range
    
    for C in C_range
        score = evaluate_pauli_config_cv(
            hyperparams.paulis,
            hyperparams.reps,
            hyperparams.entanglement,
            hyperparams.n_qubits,
            X, y, C, cv_folds
        )
        
        if score > best_score
            best_score = score
            best_C = C
        end
    end
    
    @info "C tuning complete" best_C=best_C best_score=best_score
    
    return best_C
end
