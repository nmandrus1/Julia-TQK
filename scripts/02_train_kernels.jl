using DrWatson
@quickactivate "TQK"

using TQK
using Iterators
using PythonCall

# INPUT is a train/test split 

# Phase 1: Hyperparameter Selection
    # Pauli Kernel: Combinatorial Search - Pick best one  
        # - Save best PauliConfig

    # RBF: find best gamma
        # - Save best RBFConfig 

    # Reuploading: Train kernel using CV and pick best parameters
        # - Save best parameters
        # - Save optimization parameters
        # - Save loss/optimization stuff

# Phase 2: Produce a Kernel for each training set size
        # RBF - Standard 
        # Pauli - Use qiskit QSVM evaluate
        # Reupload - Use in-house kernel evaluation 

# Phase 3: Save kernel
        # Save results to disk using the kernel-config filename



# ============================================================================
# Phase 1: Hyperparameter Selection (Cached)
# ============================================================================

"""
Select best hyperparameters using full training data.
Results are cached per (data_config, kernel_type) pair.
"""
function select_hyperparameters(cfg::ExperimentConfig{D, K}, X_train, y_train) where {D, K <:PauliKernelHyperparameterSearchConfig}
    @info "Searching Pauli configurations"
    pauli_hyperparams, best_score = search_pauli_kernel(cfg, X_train, y_train)
    @info "Selected" pauli_hyperparams best_score
    return pauli_hyperparams, best_score
end

"""
Select best hyperparameters using full training data.
Results are cached per (data_config, kernel_type) pair.
"""
function select_hyperparameters(cfg::ExperimentConfig{D, K}, X_train, y_train) where {D, K <:RBFKernelHyperparameterConfig}
    @info "Cross-validating RBF gamma"
    best_gamma, best_score = select_rbf_gamma(X_train, y_train, cfg)
    @info "Selected" best_gamma best_score
    return best_gamma, best_score
end

"""
Select best hyperparameters using full training data.
Results are cached per (data_config, kernel_type) pair.
"""
function select_hyperparameters(cfg::ExperimentConfig{D, K}, X_train, y_train) where {D, K <:ReuploadingKernelHyperparameterSearchConfig}
    @info "Cross-validating RBF gamma"
    best_weights, best_score = select_reuploading_params(X_train, y_train, cfg)
    @info "Selected" best_weights best_score
    return best_weights, best_score
end


function pairwise_distances_colmajor(X)
    # X is (n_features, n_samples)
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

function evaluate_pauli_config(paulis, reps, entanglement, n_qubits, X, y)
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    # X is column-major, need row-major for Qiskit
    X_row = X'
    
    feature_map = qiskit_lib.PauliFeatureMap(
        feature_dimension=size(X, 1),
        reps=reps,
        entanglement=entanglement,
        paulis=pylist(paulis)
    )
    
    kernel = qiskit_kernels.FidelityStatevectorKernel(feature_map=feature_map)
    K = pyconvert(Matrix{Float64}, kernel.evaluate(x_vec=X_row))
    
    # KTA score
    y_outer = y * y'
    kta = sum(K .* y_outer) / (sqrt(sum(K.^2)) * sqrt(sum(y_outer.^2)))
    
    return kta
end



function select_rbf_hyperparameters(config::ExperimentConfig{D, K}, X_train, y_train) where {D, K<:RBFHyperparameterSearchConfig}
    gamma_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    
    best_score = -Inf
    best_params::RBFHyperparameters = nothing
    
    for (gamma, C) in product(gamma_range, config)
        D = pairwise_distances_colmajor(X_train)
        K = exp.(-gamma .* D)
        
        clf = svm_module.SVC(kernel="precomputed", C=C)
        scores = model_selection.cross_val_score(clf, K, y_train, cv=cv_folds)
        score = mean(pyconvert(Vector{Float64}, scores))
        
        if score > best_score
            best_score = score
            best_params = RBFHyperparameters(gamma=gamma, C=C, config.seed)
        end
    end
    
    return best_params, best_score
end

function search_pauli_hyperparameters(config::ExperimentConfig{D, K}, X_train, y_train) where {D, K<:PauliKernelHyperparameterSearchConfig}
   
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    
    kernel_config = config.kernel_config
    best_score = -Inf
    best_config::PauliKernelHyperparameters = nothing
    
    # random combinations of the allowed parameters for the Pauli Kernel
    for _ in 1:config.n_search_iterations
        paulis = generate_constrained_pauli_set(kernel_config.search_constraints)
        reps = rand(kernel_config.reps)
        entanglement = rand(kernel_config.entanglement)
        
        K = compute_pauli_kernel(paulis, reps, entanglement, kernel_config.n_qubits, X_train)
        
        # Tune C for this kernel
        for C in config.c_ranges
            clf = svm_module.SVC(kernel="precomputed", C=C)
            scores = model_selection.cross_val_score(clf, K, y_train, cv=config.cv_folds)
            score = mean(pyconvert(Vector{Float64}, scores))
            
            if score > best_score
                best_score = score
                best_config = PauliKernelHyperparameters(
                                                         n_qubits=kernel_config.n_qubits,
                                                         paulis=paulis,
                                                         reps=reps,
                                                         entanglement=entanglement,
                                                         C=C
                                                         seed=config.seed
                                                     )
            end
        end
    end
    
    return best_config, best_score
end

function search_reuploading_hyperparameters(config::ExperimentConfig{D, K}, X_train, y_train) where {D, K<:ReuploadingKernelHyperparameterSearchConfig}

    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    
    kernel_config = config.kernel_config
    best_score = -Inf
    best_config::ReuploadingKernelHyperparameters = nothing

    for (gamma, C) in product(kernel_config.learning_rates, config.c_ranges)

        # define feature map
        feature_map = ReuploadingCircuit(
                        kernel_config.n_qubits,
                        kernel_config.n_features,
                        kernel_config.n_layers,
                        kernel_config.entanglement)

        # Initialize with small random parameters
        assign_random_params!(feature_map, seed=config.seed)
        
        kernel = FidelityKernel(feature_map)
        
        clf = svm_module.SVC(kernel="precomputed", C=C)
        scores = model_selection.cross_val_score(clf, K, y_train, cv=cv_folds)
        score = mean(pyconvert(Vector{Float64}, scores))
        
        if score > best_score
            best_score = score
            best_params = RBFHyperparameters(gamma=gamma, C=C, config.seed)
        end
    end
    
    return best_params, best_score
    
end

# ============================================================================
# Phase 2: Produce Kernels for Training Sizes
# ============================================================================

"""
Produce kernel for specific training size using best hyperparameters.
"""
function produce_kernel(kernel_config::KernelConfig, data::Dict, n_samples::Int)
    X_train = data[:X_train][:, 1:n_samples]  # (n_features, n_samples)
    y_train = data[:y_train][1:n_samples]
    
    if kernel_config isa RBFKernelConfig
        return produce_rbf_kernel(kernel_config, X_train, y_train)
    elseif kernel_config isa PauliKernelConfig
        return produce_pauli_kernel(kernel_config, X_train, y_train)
    elseif kernel_config isa ReuploadingKernelConfig
        return produce_reuploading_kernel(kernel_config, X_train, y_train)
    end
end

function produce_rbf_kernel(config::RBFKernelConfig, X_train, y_train)
    D = pairwise_distances_colmajor(X_train)
    K = exp.(-config.gamma .* D)
    
    return Dict(
        :type => :rbf,
        :gamma => config.gamma,
        :K_train => K,
        :n_samples => length(y_train)
    )
end

function produce_pauli_kernel(config::PauliKernelConfig, X_train, y_train)
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    feature_map = qiskit_lib.PauliFeatureMap(
        feature_dimension=size(X_train, 1),
        reps=config.reps[1],
        entanglement=config.entanglement[1],
        paulis=pylist(config.fixed_paulis)
    )
    
    kernel = qiskit_kernels.FidelityStatevectorKernel(feature_map=feature_map)
    K = pyconvert(Matrix{Float64}, kernel.evaluate(x_vec=X_train'))
    
    return Dict(
        :type => :pauli,
        :kernel => kernel,
        :K_train => K,
        :n_samples => length(y_train)
    )
end

function produce_reuploading_kernel(config::ReuploadingKernelConfig, X_train, y_train)
    # Create feature map
    entanglement_map = Dict("linear"=>linear, "alternating"=>alternating, "all_to_all"=>all_to_all)
    
    feature_map = ReuploadingCircuit(
        config.n_qubits,
        size(X_train, 1),
        config.n_layers,
        entanglement_map[config.entanglement]
    )
    
    assign_random_params!(feature_map, seed=config.seed)
    kernel = FidelityKernel(feature_map)
    
    # Train kernel
    loss_fn = K -> kernel_alignment_loss(K, y_train)
    trainer = QuantumKernelTrainer(kernel, loss_fn, X_train', y_train)
    
    optimizer = config.optimizer == "LBFGS" ? LBFGS() : error("Unknown optimizer")
    sol = train!(trainer, optimizer=optimizer, iterations=config.max_iterations)
    
    # Evaluate trained kernel
    K = evaluate(kernel, X_train')
    
    return Dict(
        :type => :reuploading,
        :kernel => kernel,
        :feature_map => feature_map,
        :K_train => K,
        :training_loss => sol.minimum,
        :n_samples => length(y_train)
    )
end

# ============================================================================
# Main Pipeline
# ============================================================================

"""
Run complete kernel production pipeline.
"""
function produce_kernels_pipeline(config::ExperimentConfig)
    # Load data
    data = prepare_data!(config)
    
    # Phase 1: Hyperparameter selection (cached)
    hp_params = struct_to_dict(config.data_config)
    hp_params[:kernel_type] = config.kernel_config.kernel_type
    
    best_config, _ = produce_or_load(
        datadir("hyperparameters"),
        hp_params,
        prefix="best"
    ) do
        best = select_hyperparameters(config.kernel_config, data)
        return @dict(best)
    end
    
    best_kernel_config = best_config[:best]
    
    # Phase 2: Produce kernels for each training size
    results = Dict{Int, Any}()
    
    for n_samples in config.learning_curve_sizes
        @info "Producing kernel" n_samples
        
        kernel_result = produce_kernel(best_kernel_config, data, n_samples)
        results[n_samples] = kernel_result
        
        # Save individual result
        save_params = merge(hp_params, @dict(n_samples))
        filepath = datadir("kernels", savename(save_params, "jld2"))
        wsave(filepath, kernel_result)
    end
    
    return Dict(
        :best_config => best_kernel_config,
        :results => results
    )
end
