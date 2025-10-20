using DrWatson
@quickactivate "TQK"

using TQK 

# Load required packages
using Printf
using CSV
using Tables
using LinearAlgebra
using Random
using Plots
using Optimization 
using OptimizationOptimJL
using OptimizationOptimisers
using LIBSVM
using MultivariateStats
using MLUtils
using StatsBase
# using Yao

include("experiment_runner2.jl")

# for some functions I don't want to reimplement
include("../src/reupload_hyperparam_search.jl")

# Set random seed for reproducibility
#Random.seed!(42)

@kwdef struct BreastCancerDataParams <: DataParams
    dataset_type::String="UCIBreastCancer"
end

"""
Load and prepare UCI Breast Cancer Dataset for binary classification
"""
function load_breast_cancer(;n_features=10, normalize=false, seed=nothing)
    Random.seed!(seed)
    rng = Random.default_rng()

    # 568x32 matrix column 1 is id, column 2 is label so we want to f
    bc = CSV.File(datadir("datasets", "UCIBreastCancer", "wdbc.data")) |> Tables.matrix
   
    # 568 element vector 
    y_raw = bc[:, 2]
    # Binary: one class vs rest (Malignant ("M") vs Benign ("B"))
    y = [diagnosis == "M" ? 1.0 : -1.0 for diagnosis in y_raw]

    # remove id and label columns and make column order
    X_features = bc[:, 3:end]
    X_data = permutedims(X_features)
   
    (X_train_raw, y_train), (X_test_raw, y_test) = splitobs(rng, (X_data, y); at=(1 - 0.2), shuffle=true)    

    # Normalize features (important for quantum circuits!)
    # NO DATA SNOOPING
    X_train_mean = mean(X_train_raw, dims=2)
    X_train_std = std(X_train_raw, dims=2)

    X_train_normalized = (X_train_raw .- X_train_mean) ./ X_train_std
    X_test_normalized = (X_test_raw.- X_train_mean) ./ X_train_std
    
    #Scale features to (0, 2π) range based on training data statistics.
    x_min = minimum(X_train_normalized, dims=2)
    x_max = maximum(X_test_normalized, dims=2)

    X_train_scaled = (X_train_normalized .- x_min) ./ (x_max .- x_min) .* 2π
    X_test_scaled = (X_test_normalized .- x_min) ./ (x_max .- x_min) .* 2π


    #Apply PCA dimensionality reduction.
    pca_model = fit(PCA , X_train_scaled; maxoutdim=n_features)

    # Transform both sets
    X_train = MultivariateStats.predict(pca_model, X_train_scaled)
    X_test = MultivariateStats.predict(pca_model, X_test_scaled)

    X_train = permutedims(X_train)
    X_test =  permutedims(X_test)
    return @dict X_train y_train X_test y_test   
end

"""
Random starts search for reuploading kernel parameters.
"""
function _random_starts_search(
    config::ExperimentConfig,
    X_train::AbstractMatrix,
    y_train::AbstractVector,
    X_val::AbstractMatrix,
    y_val::AbstractVector;
    n_starts=5
)
    kconfig = config.kernel_config
    best_score = -Inf
    best_weights = nothing
    best_biases = nothing
    all_losses = []
    history = Dict()
    
    @info "Running $n_starts random starts..."
    
    kernel_config = config.kernel_config
    iterations = kernel_config.iterations
    n_full = size(X_train, 1)

    proxy_C = find_best_proxy_C_reup(config, X_train, y_train)

    for start in 1:n_starts
        @info "  Start $start/$n_starts"
        
        # Create fresh feature map
        feature_map = ReuploadingCircuit(kconfig.n_qubits, kconfig.n_features, kconfig.n_layers, linear)
        assign_random_params!(feature_map, range=(-π, π) .* (1/kconfig.n_qubits), seed=config.seed + start)
        
        kernel = FidelityKernel(feature_map)
        
        # Progressive training through iteration stages
        for (stage, iters) in enumerate(kconfig.iterations)
            Y_outer = y_train * y_train'
            loss_fn = K -> smooth_hinge_loss(K, Y_outer)
            
            trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_train)
            
            train!(trainer,
                  optimizer=OptimizationOptimisers.AMSGrad(eta=kconfig.learning_rate),
                  iterations=iters,
                  callback=(s, l) -> nothing
              )
        end
        
        # Evaluate on validation set
        K_val = TQK.evaluate(kernel, X_val, X_train)
        K_train_full = TQK.evaluate(kernel, X_train)
        
        # Quick SVM evaluation with default C=1.0
        model = svmtrain(K_train_full, y_train, kernel=Kernel.Precomputed, cost=proxy_C, verbose=false)
        val_pred = svmpredict(model, K_val')[1]
        val_acc = mean(val_pred .== y_val)
        
        push!(all_losses, val_acc)
        
        @info "    Validation accuracy: $(round(val_acc*100, digits=1))%"
        
        if val_acc > best_score
            best_score = val_acc
            best_weights, best_biases = get_params(feature_map)
            history[:final_val_acc] = val_acc
        end
    end
    
    @info "Best validation accuracy: $(round(best_score*100, digits=1))%"
    return best_weights, best_biases, all_losses, history
end

"""
Modified reuploading kernel training with CV.
"""
function _train_reuploading_kernel_with_cv(
    config::ExperimentConfig,
    X_train::AbstractMatrix,
    y_train::AbstractVector
)     
    # Split into train/val (80/20)
    n = length(y_train)
    n_val = div(n, 5)
    
    X_val = X_train[end-n_val+1:end, :]
    X_train_sub = X_train[1:end-n_val, :]
    y_val = y_train[end-n_val+1:end]
    y_train_sub = y_train[1:end-n_val]
    
    @info "Training set: $(size(X_train_sub, 1)), Validation set: $(size(X_val, 1))"
    
    # Run multi-fidelity search
    best_weights, best_biases, losses, history = _random_starts_search(
        config, X_train_sub, y_train_sub, X_val, y_val
    )
    
    best_C = cv_C_search(config, X_train, y_train, best_weights, best_biases)
    
    return ReuploadingKernelHyperparameters(
        nqubits=config.kernel_config.n_qubits,
        nlayers=config.kernel_config.n_layers,
        thetas=best_weights,
        biases=best_biases,
        loss=losses,
        C=best_C,
    ), history[:final_val_acc]
end


function smooth_hinge_loss(K::Matrix, Y_outer::AbstractMatrix)
    # Approximate decision function from kernel
    # For kernel methods: f(x) = Σⱼ αⱼyⱼK(xⱼ,x)
    # Simplified version: use kernel-based scores
    # K_normalized = K ./ (norm(K) + 1e-8)  # Normalize to prevent explosion
    # Y = y* y'
    margins = Y_outer .* K   
    
    # Smooth hinge loss using squared max
    smooth_hinge = mean(max.(0, 1 .- margins).^2)
    
    return smooth_hinge
end


# Main experimen

"""
Simple comparison between tuned RBF and Reuploading kernels on breast cancer data.
"""
function compare_kernels_breast_cancer(;n_features=8, seed=42)
    println("="^60)
    println("RBF vs Reuploading Kernel Comparison - Breast Cancer")
    println("="^60)
    
    Random.seed!(seed)
    
    # Load data
    println("\n1. Loading breast cancer data...")
    data = load_breast_cancer(;n_features=n_features, seed=seed)
    X_train = data[:X_train]  # Convert to row-major for consistency
    X_test = data[:X_test]
    y_train = data[:y_train]
    y_test = data[:y_test]
    
    println("  Train samples: $(length(y_train))")
    println("  Test samples: $(length(y_test))")
    println("  Features: $n_features")
    
    # ==========================================================================
    # RBF Kernel Hyperparameter Search
    # ==========================================================================
    println("\n2. Tuning RBF Kernel...")
    
    # Set up RBF search configuration
    rbf_config = ExperimentConfig(
        experiment_name="rbf_breast_cancer",
        data_config=DataConfig(
            n_samples=length(y_train),
            n_features=n_features,
            seed=seed,
            data_params=BreastCancerDataParams(),
        ),
        kernel_config=RBFKernelHyperparameterSearchConfig(
            gamma_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        ),
        c_ranges=[0.1, 1.0, 10.0, 100.0],
        cv_folds=5,
        seed=seed
    )
    
    rbf_hyperparams, rbf_score = search_rbf_hyperparameters(rbf_config, X_train, y_train)
    
    println("  Best RBF parameters:")
    println("    Gamma: $(rbf_hyperparams.gamma)")
    println("    C: $(rbf_hyperparams.C)")
    println("    CV Score: $(round(rbf_score, digits=4))")
    
    # ==========================================================================
    # Reuploading Kernel Hyperparameter Search  
    # ==========================================================================
    println("\n3. Tuning Reuploading Kernel...")
    
    # Set up reuploading search configuration
    reup_config = ExperimentConfig(
        experiment_name="reup_breast_cancer",
        data_config=DataConfig(
            n_samples=length(y_train),
            n_features=n_features,
            seed=seed,
            data_params=BreastCancerDataParams(),
        ),
        kernel_config=ReuploadingKernelHyperparameterSearchConfig(
            n_qubits=8,
            n_features=n_features,
            n_layers=4,
            entanglement="all_to_all",
            iterations=[20, 30, 50],  # Progressive training
            learning_rate=0.01,
            seed=seed
        ),
        c_ranges=[0.1, 1.0, 10.0, 100.0],
        cv_folds=5,
        seed=seed
    )
    
    reup_hyperparams, reup_score = _train_reuploading_kernel_with_cv(reup_config, X_train, y_train)
    
    println("  Best Reuploading parameters:")
    println("    Qubits: $(reup_hyperparams.nqubits)")
    println("    Layers: $(reup_hyperparams.nlayers)")
    println("    C: $(reup_hyperparams.C)")
    println("    CV Score: $(round(reup_score, digits=4))")
    
    # ==========================================================================
    # Final Evaluation on Test Set
    # ==========================================================================
    println("\n4. Final Test Set Evaluation...")
    
    # RBF evaluation
    println("\n  RBF Kernel:")
    D_train_sq = pairwise(SqEuclidean(), X_train', dims=2)
    K_train_rbf = exp.(-rbf_hyperparams.gamma .* D_train_sq)
    
    D_test_sq = pairwise(SqEuclidean(), X_test', X_train', dims=2)
    K_test_rbf = exp.(-rbf_hyperparams.gamma .* D_test_sq)
    
    model_rbf = svmtrain(K_train_rbf, y_train, kernel=Kernel.Precomputed, cost=rbf_hyperparams.C, verbose=false)
    train_pred_rbf = svmpredict(model_rbf, K_train_rbf)[1]
    test_pred_rbf = svmpredict(model_rbf, K_test_rbf')[1]
    
    train_acc_rbf = mean(train_pred_rbf .== y_train)
    test_acc_rbf = mean(test_pred_rbf .== y_test)
    
    println("    Train Accuracy: $(round(train_acc_rbf*100, digits=1))%")
    println("    Test Accuracy:  $(round(test_acc_rbf*100, digits=1))%")
    
    # Reuploading evaluation
    println("\n  Reuploading Kernel:")
    
    # Reconstruct trained feature map
    feature_map = ReuploadingCircuit(reup_hyperparams.nqubits, n_features, reup_hyperparams.nlayers, linear)
    assign_params!(feature_map, reup_hyperparams.thetas, reup_hyperparams.biases)
    
    kernel_reup = FidelityKernel(feature_map)
    K_train_reup = TQK.evaluate(kernel_reup, X_train)  # Back to column-major
    K_test_reup = TQK.evaluate(kernel_reup, X_test, X_train)
    
    model_reup = svmtrain(K_train_reup, y_train, kernel=Kernel.Precomputed, cost=reup_hyperparams.C, verbose=false)
    train_pred_reup = svmpredict(model_reup, K_train_reup)[1]
    test_pred_reup = svmpredict(model_reup, K_test_reup')[1]
    
    train_acc_reup = mean(train_pred_reup .== y_train)
    test_acc_reup = mean(test_pred_reup .== y_test)
    
    println("    Train Accuracy: $(round(train_acc_reup*100, digits=1))%")
    println("    Test Accuracy:  $(round(test_acc_reup*100, digits=1))%")
    
    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    println("\n" * "="^60)
    println("SUMMARY COMPARISON")
    println("="^60)
    
    @sprintf "%-20s %12s %12s %12s" "Kernel" "CV Score" "Train Acc" "Test Acc"
    println("-"^60)
    @sprintf "%-20s %12.4f %11.1f%% %11.1f%%" "RBF" rbf_score train_acc_rbf test_acc_rbf
    @sprintf "%-20s %12.4f %11.1f%% %11.1f%%" "Reuploading" reup_score train_acc_reup test_acc_reup
    
    # Determine winner
    println("\nWinner by test accuracy: ", test_acc_rbf > test_acc_reup ? "RBF" : "Reuploading")
    println("Test accuracy difference: $(round(abs(test_acc_rbf - test_acc_reup)*100, digits=1))%")
    
    # Return results for further analysis if needed
    return Dict(
        :rbf_results => Dict(
            :hyperparams => rbf_hyperparams,
            :cv_score => rbf_score,
            :train_acc => train_acc_rbf,
            :test_acc => test_acc_rbf
        ),
        :reup_results => Dict(
            :hyperparams => reup_hyperparams,
            :cv_score => reup_score,
            :train_acc => train_acc_reup,
            :test_acc => test_acc_reup
        ),
        :data => data
    )
end


