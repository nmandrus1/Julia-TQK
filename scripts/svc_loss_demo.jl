using DrWatson
@quickactivate "TQK"

using TQK 

# Load required packages
using LinearAlgebra
using Random
using Plots
using Optimization 
using OptimizationOptimJL
using OptimizationOptimisers
using BlackBoxOptim
using OptimizationBBO
using LIBSVM
using MultivariateStats
using MLUtils
using RDatasets
using StatsBase

# Set random seed for reproducibility
#Random.seed!(42)

"""
Load and prepare Iris dataset for binary classification
"""
function load_iris_binary(; positive_class="setosa", n_features=4, seed=nothing)
    Random.seed!(seed)
    iris = dataset("datasets", "iris")
    
    X = permutedims(Matrix(Float64.(iris[:, 1:4])))  # 4 features
    y_raw = iris.Species
    
    # Binary: one class vs rest
    y = [species == positive_class ? 1.0 : -1.0 for species in y_raw]
   
    # Normalize features (important for quantum circuits!)
    X_normalized = (X .- mean(X, dims=1)) ./ std(X, dims=1)

    rng = Random.default_rng()

    # Split train/test
    (X_train_raw, y_train), (X_test_raw, y_test) = splitobs(rng, (X_normalized, y); at=(1 - 0.2), shuffle=true)    
    y_train = Vector{Float64}(y_train)
    y_test = Vector{Float64}(y_test)
    
    #Scale features to (0, 2π) range based on training data statistics.
    x_min = minimum(X_train_raw, dims=2)
    x_max = maximum(X_train_raw, dims=2)

    X_train_scaled = (X_train_raw .- x_min) ./ (x_max .- x_min) .* 2π
    X_test_scaled = (X_test_raw .- x_min) ./ (x_max .- x_min) .* 2π

    # scaler = (min=x_min, max=x_max)

    #Apply PCA dimensionality reduction.
    # pca_model = fit(PCA , X_train_scaled; maxoutdim=n_features)

    # Transform both sets
    # X_train = MultivariateStats.predict(pca_model, X_train_scaled)
    # X_test = MultivariateStats.predict(pca_model, X_test_scaled)

    X_train = permutedims(X_train_scaled)
    X_test =  permutedims(X_test_scaled)
    return @dict X_train y_train X_test y_test   
end

function centered_kernel_target_alignment(K::Matrix, y::AbstractVector)
    n = length(y)
    H = I - ones(n, n) / n  # Centering matrix
    K_centered = H * K * H
    y_outer = y * y'
    y_centered = H * y_outer * H
    
    numerator = tr(K_centered * y_centered)
    denominator = sqrt(tr(K_centered^2) * tr(y_centered^2))
    
    return numerator / denominator 
end

"""
Smooth SVM loss using tanh approximation.

The standard SVM primal is:
    L = (1/2)||w||² + C·Σᵢ max(0, 1 - yᵢf(xᵢ))

We approximate with tanh:
    L_smooth = (1/2)||w||² + C·Σᵢ [(1 - tanh(β(mᵢ - 1)))/2]

where mᵢ = yᵢf(xᵢ) is the margin.
"""
function smooth_svm_loss_tanh(K::Matrix, y::AbstractVector; C=1.0, β=5.0, λ=0.01)
    n = length(y)
    
    # Approximate decision function from kernel
    # For kernel methods: f(x) = Σⱼ αⱼyⱼK(xⱼ,x)
    # Simplified version: use kernel-based scores
    # K_normalized = K ./ (norm(K) + 1e-8)  # Normalize to prevent explosion
    decision_values = K * y
    
    # Compute margins: m = y · f(x)
    margins = y .* decision_values
    # println(margins)
    
    # Smooth hinge loss using tanh
    # ℓ(m) = (1 - tanh(β(m - 1)))/2
    smooth_hinge = sum((1 .- tanh.(β .* (margins .- 1))) ./ 2)
    
    # Regularization (approximate ||w||² from kernel)
    # regularization = 0.5 * λ * sum(K .^ 2) / n^2
    
    # total_loss = regularization + (C / n) * smooth_hinge
    total_loss = C * smooth_hinge
    
    return total_loss
end


"""
Smooth SVM loss using decision function approximation and squared hinge loss .
"""
function smooth_svm_loss(K::Matrix, y::AbstractVector)
    Y = y * y' # Ideal kernel matrix Y_ij = y_i * y_j

    # We want K_ij to be close to Y_ij.
    # The margin for each pair (i,j) is Y_ij * K_ij.
    margins = Y .* K

    # Apply the smooth squared hinge loss
    # Penalize any pair where the margin is less than 1.
    loss = sum(max.(0, 1 .- margins).^2)

    # return loss / (length(y)^2) # Normalize    
    return loss 
end

# Main experiment
function run_gradient_descent_demo(;seed::Union{Int, Nothing}=nothing)
    println("=== Quantum Kernel Gradient Descent Demo ===\n")
   
    # Create dataset
    # data = load_iris_binary()
    # data = produce_data(
    #            DataConfig(
    #                 n_samples=500,
    #                 data_params=RBFDataParams(
    #                     gamma=5.0,
    #                     n_support_vectors=50,
    #                 ),
    #                 seed=seed
    #             )
    #         )

    data = produce_data(
               DataConfig(
                    n_samples=500,
                    data_params=QuantumPauliDataParams(
                        n_qubits=2,
                        paulis=["Y", "ZY"],
                        entanglement="linear",
                        reps=2,
                        gap=0.1,
                        grid_points_per_dim=100,
                    ),
                    seed=seed
                )
            )

    X_train = permutedims(data[:X_train])
    X_test = permutedims(data[:X_test])
    y_train = data[:y_train]
    y_test = data[:y_test]
   
    # Create simple quantum feature map
    n_qubits = 4
    n_features = 2
    n_layers = 8
    entanglement = all_to_all

    if isnothing(seed)
        seed = rand(Int)
    end

    Random.seed!(seed)
    
    feature_map = ReuploadingCircuit(n_qubits, n_features, n_layers, entanglement)
    
    # Initialize with small random parameters
    assign_random_params!(feature_map, range=(-π, π) .* (1/n_qubits), seed=seed)
    # assign_random_params!(feature_map, range=(-(1/sqrt(n_features), 1/sqrt(n_features))), seed=seed)
    
    # Create kernel
    kernel = FidelityKernel(feature_map)
    K_initial = TQK.evaluate(kernel, X_train)  # With random params
    
    println("\nQuantum circuit configuration:")
    println("  Qubits: $n_qubits")
    println("  Layers: $n_layers")
    println("  Total parameters: $(2 * n_params(feature_map)) ($(n_params(feature_map)) weights + $(n_params(feature_map)) biases)")
    println("  Seed: $seed")

    # support vectors and corresponding ground truth values
    # to be updated on callback
    # 
    model = svmtrain(K_initial, y_train, kernel=Kernel.Precomputed)
    indices = model.SVs.indices
    ai = vec(model.coefs) .* y_train[indices]

    # loss_fn = K -> svm_loss(K, ai, y, indices)
    # loss_fn = K -> centered_kernel_target_alignment(K, y_train)
    # loss_fn = K -> smooth_svm_loss_tanh(K, y_train, β=1.0)
    loss_fn = K -> smooth_svm_loss(K, y_train)
    trainer = QuantumKernelTrainer(
        kernel,
        loss_fn,
        X_train,
        y_train,
    )

    # Train
    # losses = Vector{Float64}(undef, ITERS+1)

    # Track metrics during training
    metrics = Dict(
        "iteration" => Int[],
        "loss" => Float64[],
        "train_accuracy" => Float64[],
        "test_accuracy" => Float64[],
    )
    
    function evaluation_callback(state, loss)
        iter = state.iter
        
        # Compute kernel matrices
        K_train = trainer.K_cache
        K_test = TQK.evaluate(kernel, X_test, X_train)
        
        # Train SVM on current kernel
        model = svmtrain(K_train, y_train, kernel=Kernel.Precomputed, verbose=false)
        
        # Evaluate
        train_pred = svmpredict(model, K_train)[1]
        test_pred = svmpredict(model, K_test')[1]
        
        train_acc = mean(train_pred .== y_train)
        test_acc = mean(test_pred .== y_test)
               
        # Store metrics
        push!(metrics["iteration"], iter)
        push!(metrics["loss"], loss)
        push!(metrics["train_accuracy"], train_acc)
        push!(metrics["test_accuracy"], test_acc)
        
        if iter % 5 == 0 || iter == 1
            println("Iter $iter: loss=$(round(loss, digits=4)), " *
                    "train_acc=$(round(train_acc*100, digits=1))%, " *
                    "test_acc=$(round(test_acc*100, digits=1))%")
        end
    end

    println("\nStarting training ...\n")
    ITERS = 500
    sol = train!(trainer,
                 optimizer= OptimizationOptimisers.Adam(eta=0.01),
                 # optimizer= LBFGS(),
                 iterations=ITERS,
                 #callback=(state, loss) -> losses[state.iter + 1] = loss
                 callback=evaluation_callback
             )
    #return sol

    losses = metrics["loss"]
   
    # Plot results
    println("\nCreating loss plot...")
    p = Plots.plot(1:length(losses)-1, losses[1:length(losses)-1], 
             xlabel="Iterations", 
             ylabel="Loss", 
             title="Quantum Kernel Training Loss",
             label="Training Loss",
             lw=2,
             marker=:circle,
             markersize=2,
             markerstrokewidth=0)
    
   p_acc = plot(metrics["iteration"], 
             [metrics["train_accuracy"].*100 metrics["test_accuracy"].*100],
             xlabel="Iterations", 
             ylabel="Accuracy (%)", 
             title="Classification Accuracy",
             label=["Train" "Test"],
             lw=2,
             marker=[:circle :square],
             markersize=3,
             markerstrokewidth=0,
             color=[:blue :red])   

    
    p_data = plot_classification_data(data)

    K_train = trainer.K_cache
    K_test = TQK.evaluate(kernel, X_test, X_train)

    model = svmtrain(K_train, y_train, kernel=Kernel.Precomputed, verbose=false)
    # Evaluate
    train_pred = svmpredict(model, K_train)[1]
    test_pred = svmpredict(model, K_test')[1]
    
    train_acc = mean(train_pred .== y_train)
    test_acc = mean(test_pred .== y_test)

    println("Final: train_acc=$(round(train_acc*100, digits=1))%, " * "test_acc=$(round(test_acc*100, digits=1))%")



    # p_boundary = plot_svm_decision_boundary(model, data)
    # return losses, kernel, p, p_kernels
    return kernel, p, p_acc, p_data, model, metrics, data
end


function plot_classification_data(data::Dict)
    X_train = data[:X_train]'
    X_test = data[:X_test]'
    y_train = data[:y_train]
    y_test = data[:y_test]
    
    # Create the plot
    p = plot(legend=:best, xlabel="Feature 1", ylabel="Feature 2")
    
    # Train data: circles for class -1, squares for class 1
    train_mask_neg = y_train .== -1
    train_mask_pos = y_train .== 1
    scatter!(p, X_train[train_mask_neg, 1], X_train[train_mask_neg, 2], 
             color=:blue, marker=:circle, label="Train (−1)", markersize=6)
    scatter!(p, X_train[train_mask_pos, 1], X_train[train_mask_pos, 2], 
             color=:red, marker=:circle, label="Train (+1)", markersize=6)
    
    # Test data: triangles for class -1, diamonds for class 1
    test_mask_neg = y_test .== -1
    test_mask_pos = y_test .== 1
    scatter!(p, X_test[test_mask_neg, 1], X_test[test_mask_neg, 2], 
             color=:blue, marker=:utriangle, label="Test (−1)", markersize=7)
    scatter!(p, X_test[test_mask_pos, 1], X_test[test_mask_pos, 2], 
             color=:red, marker=:diamond, label="Test (+1)", markersize=7)
    
    return p
end



function plot_svm_decision_boundary(model, kernel, data; resolution=100, margin=0.5)
    X_train = data[:X_train]
    X_test = data[:X_test]
    y_train = data[:y_train]
    y_test = data[:y_test]
    
    # Get feature ranges
    x_min, x_max = minimum(X_train[1, :]) - margin, maximum(X_train[1, :]) + margin
    y_min, y_max = minimum(X_train[2, :]) - margin, maximum(X_train[2, :]) + margin
    
    # Create mesh grid
    x_range = range(x_min, x_max, length=resolution)
    y_range = range(y_min, y_max, length=resolution)
    
    # Create grid points matrix (2 x resolution^2)
    X_grid = vcat([[x y] for x in x_range for y in y_range]...)
    
    # Compute kernel matrix between grid and training data
    K_grid = TQK.evaluate(kernel, X_grid, X_train')
    
    # Predict on all grid points at once
    Z_flat = svmpredict(model, K_grid')[1]
    
    # Reshape to grid
    Z = reshape(Z_flat, resolution, resolution)
    
    # Plot decision boundary
    p = contourf(x_range, y_range, Z, levels=[-1.5, 0, 1.5], 
                 color=:RdBu, alpha=0.7, colorbar=false)
    
    # Plot training data
    train_neg = y_train .== -1
    train_pos = y_train .== 1
    scatter!(p, X_train[1, train_neg], X_train[2, train_neg], 
             color=:blue, marker=:circle, label="Train (−1)", markersize=6, alpha=0.2)
    scatter!(p, X_train[1, train_pos], X_train[2, train_pos], 
             color=:red, marker=:circle, label="Train (+1)", markersize=6, alpha=0.2)
    
    # Plot test data
    test_neg = y_test .== -1
    test_pos = y_test .== 1
    scatter!(p, X_test[1, test_neg], X_test[2, test_neg], 
             color=:blue, marker=:utriangle, label="Test (−1)", markersize=7, alpha=0.2)
    scatter!(p, X_test[1, test_pos], X_test[2, test_pos], 
             color=:red, marker=:diamond, label="Test (+1)", markersize=7, alpha=0.2)
    
    plot!(p, xlabel="Feature 1", ylabel="Feature 2", legend=:best)
    return p
end

function controlled_experiment(; n_trials=5)
    """Run multiple trials with FIXED seeds to remove randomness"""
    
    results = []
    
    for trial in 1:n_trials
        println("\n" * "="^60)
        println("TRIAL $trial")
        println("="^60)
        
        # FIXED seed for data
        data = load_iris_binary(seed=42)  # Same data every time
        X_train, y_train = data[:X_train], data[:y_train]
        X_test, y_test = data[:X_test], data[:y_test]
        
        # FIXED seed for initialization
        Random.seed!(100 + trial)  # Different init each trial, but reproducible
        feature_map = ReuploadingCircuit(8, 4, 2, linear)
        assign_random_params!(feature_map, seed=100+trial)
        kernel = FidelityKernel(feature_map)
        
        # Initial performance
        K_init = TQK.evaluate(kernel, X_train)
        model_init = svmtrain(K_init, y_train, kernel=Kernel.Precomputed, verbose=false)
        init_acc = mean(svmpredict(model_init, K_init)[1] .== y_train)
        init_kta = tr(K_init * (y_train*y_train')) / sqrt(tr(K_init^2) * tr((y_train*y_train')^2))
        
        # Train
        loss_fn = K -> -tr(K * (y_train*y_train')) / sqrt(tr(K^2) * tr((y_train*y_train')^2))
        trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_train)
        
        sol = train!(trainer, optimizer=OptimizationOptimisers.Adam(eta=0.01), iterations=100, 
                    callback=(s,l) -> nothing)  # Silent
        
        # Final performance
        K_final = TQK.evaluate(kernel, X_train)
        model_final = svmtrain(K_final, y_train, kernel=Kernel.Precomputed, verbose=false)
        final_acc = mean(svmpredict(model_final, K_final)[1] .== y_train)
        final_kta = tr(K_final * (y_train*y_train')) / sqrt(tr(K_final^2) * tr((y_train*y_train')^2))
        
        push!(results, (
            init_kta=init_kta,
            final_kta=final_kta,
            init_acc=init_acc,
            final_acc=final_acc,
            improvement=final_kta - init_kta
        ))
        
        println("  Init:  KTA=$(round(init_kta, digits=3)), Acc=$(round(init_acc*100, digits=1))%")
        println("  Final: KTA=$(round(final_kta, digits=3)), Acc=$(round(final_acc*100, digits=1))%")
        println("  Δ:     $(round((final_kta-init_kta), digits=3))")
    end
    
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    improvements = [r.improvement for r in results]
    println("Mean KTA improvement: $(round(mean(improvements), digits=4))")
    println("Std KTA improvement:  $(round(std(improvements), digits=4))")
    println("Always improves? $(all(improvements .> 0))")
    println("="^60)
    
    return results
end

# results = controlled_experiment(n_trials=5)
