
using LIBSVM
using Statistics
using Printf

"""
Smooth hinge loss for SVM (from svc_loss_demo.jl)
"""
function smooth_hinge_loss(K::Matrix, y::AbstractVector)
    Y = y * y'
    margins = Y .* K   
    loss = mean(max.(0, 1 .- margins).^2)
    return loss
end

"""
    evaluate_kernel_svm(kernel, X_train, y_train, X_val, y_val, C)

Train SVM on kernel and return validation accuracy.
"""
function evaluate_kernel_svm(
    kernel::FidelityKernel,
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    X_val::AbstractMatrix,
    y_val::AbstractVector,
    C::Float64
)
    # Compute kernels
    K_train = TQK.evaluate(kernel, X_train)
    K_val = TQK.evaluate(kernel, X_val, X_train)
    
    # Train SVM
    model = svmtrain(K_train, y_train, 
                    kernel=Kernel.Precomputed, 
                    cost=C, 
                    verbose=false)
    
    # Evaluate
    val_pred = svmpredict(model, K_val')[1]
    val_acc = mean(val_pred .== y_val)
    
    return val_acc
end


"""
    multi_fidelity_c_search(config, X_train, y_train, X_val, y_val)

Search over a set of C candidates for best performing parameter
using Cross validation. 

Returns: (best_C, best_score)
"""
function cv_C_search(
    config::ExperimentConfig{D, ReuploadingKernelHyperparameterSearchConfig},
    X_train::AbstractMatrix,
    y_train::AbstractVector,
) where D
    
    kernel_config = config.kernel_config
    
    # Phase 1: Coarse search (20% data, 20 iterations)
    @info "Starting Reuploading C search with CV" 

    best_C = config.c_ranges[1]
    best_score = -Inf

    
    svm_module = pyimport("sklearn.svm")
    model_selection = pyimport("sklearn.model_selection")
    np = pyimport("numpy")
    
    # Compute kernels
    K = TQK.evaluate(kernel, X_train)

    y_np = np.asarray(y)
    K_np = np.asarray(K)
        
    for C in config.c_ranges
        # Create fresh feature map
        feature_map = create_reuploading_feature_map(kernel_config, config.seed)
        kernel = FidelityKernel(feature_map)

        clf = svm_module.SVC(kernel="precomputed", C=C)
        scores = model_selection.cross_val_score(clf, K_np, y_np, cv=cv_folds)

        score = mean(pyconvert(Vector{Float64}, scores))
        @info "Evaluated" C=C cv_score=score

        if score > best_score
            best_score = score
            best_C = C
        end
    end

    @info "Search Complete" best_C=best_C best_score=best_score
    
    return best_C 
end


function random_starts_search(
    config::ExperimentConfig{D, ReuploadingKernelHyperparameterSearchConfig},
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    X_val::AbstractMatrix,
    y_val::AbstractVector;
    number_of_starts=6,
    learning_rate=0.01
) where D
    
    kernel_config = config.kernel_config
    n_full = size(X_train, 1)
    
    # Phase 1: Coarse search (20% data, 20 iterations)
    n_phase1 = div(n_full, 5)
    phase1_results = []
    
    @info "Phase 1: Coarse search on $n_phase1 samples, 20 iterations"
    
    for n in number_of_starts
        # Create fresh feature map
        feature_map = create_reuploading_feature_map(kernel_config, config.seed)
        kernel = FidelityKernel(feature_map)
        
        # Train
        loss_fn = K -> smooth_hinge_loss(K, y_train[1:n_phase1])
        trainer = QuantumKernelTrainer(
            kernel, loss_fn, 
            X_train[1:n_phase1, :], 
            y_train[1:n_phase1]
        )
        
        train!(trainer, 
               optimizer=OptimizationOptimisers.AMSGrad(eta=learning_rate),
               iterations=20,
               callback=(s, l) -> nothing)
        
        # Evaluate
        val_acc = evaluate_kernel_svm(kernel, 
            X_train[1:n_phase1, :], y_train[1:n_phase1],
            X_val, y_val)
        
        push!(phase1_results, (fm = feature_map, val_acc=val_acc, kernel=kernel))
        @info "  Param Set $n: val_acc=$(round(val_acc*100, digits=1))%"
    end
    
    # Phase 2: Refine top 3 (50% data, 20 iterations)
    n_phase2 = div(n_full, 2)
    top3 = sort(phase1_results, by=x->x.val_acc, rev=true)[1:min(3, length(phase1_results))]
    phase2_results = []
    
    @info "Phase 2: Refining top 3 on $n_phase2 samples, 50 iterations"
    
    for result in top3        
        # Continue feature map
        feature_map = result.fm
        kernel = FidelityKernel(feature_map)
        
        # Train with more data
        loss_fn = K -> smooth_hinge_loss(K, y_train[1:n_phase2])
        trainer = QuantumKernelTrainer(
            kernel, loss_fn,
            X_train[1:n_phase2, :],
            y_train[1:n_phase2]
        )
        
        losses = Float64[]
        train!(trainer,
               optimizer=OptimizationOptimisers.AMSGrad(eta=learning_rate),
               iterations=20,
               callback=(s, l) -> push!(losses, l))
        
        # Evaluate
        val_acc = evaluate_kernel_svm(kernel,
            X_train[1:n_phase2, :], y_train[1:n_phase2],
            X_val, y_val, C)
        
        push!(phase2_results, (fm = feature_map, val_acc=val_acc, kernel=kernel, losses=losses))
        @info "  Param Set $n: val_acc=$(round(val_acc*100, digits=1))%"
    end
    
    # Phase 3: Full training on winner
    best = argmax(r -> r.val_acc, phase2_results)
    best_weights = best.fm.weights
    best_biases = best.fm.biases
    
    @info "Phase 3: Final training with param set on $n_full samples, 30 more iterations"
    
    # Fresh feature map for final training
    feature_map = best.fm
    kernel = FidelityKernel(feature_map)
    
    # Train with full data
    loss_fn = K -> smooth_hinge_loss(K, y_train)
    trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_train)
    
    final_losses = Float64[]
    train!(trainer,
           optimizer=OptimizationOptimisers.AMSGrad(eta=learning_rate),
           iterations=kernel_config.max_iterations,
           callback=(s, l) -> push!(final_losses, l))
    
    # Final evaluation
    final_val_acc = evaluate_kernel_svm(kernel, X_train, y_train, X_val, y_val)
    
    @info "Final: val_acc=$(round(final_val_acc*100, digits=1))%"
        
    return best_weights, best_biases, final_losses, Dict(
        :phase1 => phase1_results,
        :phase2 => phase2_results,
        :final_val_acc => final_val_acc
    )

end

"""
Helper to create reuploading feature map from config.
"""
function create_reuploading_feature_map(
    kernel_config::ReuploadingKernelHyperparameterSearchConfig,
    seed::Int
)
    ent_map = Dict(
        "linear" => linear,
        "alternating" => alternating,
        "all_to_all" => all_to_all
    )
    
    feature_map = ReuploadingCircuit(
        kernel_config.n_qubits,
        kernel_config.n_features,
        kernel_config.n_layers,
        ent_map[kernel_config.entanglement]
    )
    
    assign_random_params!(feature_map, seed=seed)
    
    return feature_map
end

"""
    train_reuploading_kernel_with_cv(X_train, y_train, config)

Main entry point for training reuploading kernel with cross-validation.
Splits data into train/val, runs multi-fidelity search, returns best hyperparameters.
"""
function train_reuploading_kernel_with_cv(
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    config::ExperimentConfig{D, ReuploadingKernelHyperparameterSearchConfig}
) where D
    
    # Split into train/val (80/20)
    n = length(y_train)
    n_val = div(n, 5)
    
    X_val = X_train[end-n_val+1:end, :]
    X_train_sub = X_train[1:end-n_val, :]
    y_val = y_train[end-n_val+1:end]
    y_train_sub = y_train[1:end-n_val]
    
    @info "Training set: $(size(X_train_sub, 1)), Validation set: $(size(X_val, 1))"
    
    # Run multi-fidelity search
    best_weights, best_biases, losses, history = random_starts_search(
        config, X_train_sub, y_train_sub, X_val, y_val
    )

    best_C = cv_C_search(config, X_train, y_train)
    
    return ReuploadingKernelHyperparameters(
        thetas=weights,
        biases=biases,
        loss=losses,
        C=best_C
    ), history[:final_val_acc]
end

"""
    train_reuploading_kernel_simple(X_train, y_train, config; C=1.0)

Simple training without CV (for when C is already known).
"""
function train_reuploading_kernel_simple(
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    config::ExperimentConfig{D, ReuploadingKernelHyperparameterSearchConfig};
    C::Float64=1.0,
    learning_rate::Float64=0.01
) where D
    
    kernel_config = config.kernel_config
    
    feature_map = create_reuploading_feature_map(kernel_config, config.seed)
    kernel = FidelityKernel(feature_map)
    
    loss_fn = K -> smooth_hinge_loss(K, y_train, C=C)
    trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_train)
    
    losses = Float64[]
    train!(trainer,
           optimizer=OptimizationOptimisers.AMSGrad(eta=learning_rate),
           iterations=kernel_config.max_iterations,
           callback=(s, l) -> push!(losses, l))
    
    weights, biases = get_params(feature_map)
    
    return ReuploadingKernelHyperparameters(
        thetas=weights,
        biases=biases,
        loss=losses,
        C=C
    ), losses[end]
end
