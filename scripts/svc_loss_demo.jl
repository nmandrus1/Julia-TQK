using DrWatson
@quickactivate "TQK"

using TQK 

# Load required packages
using Printf
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
    # println("Norm K_Centered = ", norm(K_centered))
    y_outer = y * y'
    y_centered = H * y_outer * H
    
    # numerator = tr(K_centered * y_centered)
    # denominator = sqrt(tr(K_centered^2) * tr(y_centered^2)) + 1e-8
    numerator = tr(K_centered' * y_centered)
    denominator = norm(K_centered) * norm(y_centered) + 1e-8
    # println(numerator, "/", denominator)
    
    return numerator / denominator
end


function kernel_alignment_squared_error(K::Matrix, y::AbstractVector)
    n = length(y)
    y_outer = y * y'

    loss = mean((K .- y_outer).^2)
    
    return loss
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
    Y = y* y'
    margins = Y .* K   
    
    # Smooth hinge loss using tanh
    # ℓ(m) = (1 - tanh(β(m - 1)))/2
    smooth_hinge = mean((1 .- tanh.(β .* (margins .- 1))) ./ 2)
    
    # Regularization (approximate ||w||² from kernel)
    regularization = λ * norm(K)
    
    # total_loss = regularization + (C / n) * smooth_hinge
    # println("Smooth Hinge Loss: ", smooth_hinge)
    total_loss = C * smooth_hinge + regularization
    
    return total_loss
end



function smooth_hinge_loss(K::Matrix, y::AbstractVector; C=1.0, λ=0.01)
    # Approximate decision function from kernel
    # For kernel methods: f(x) = Σⱼ αⱼyⱼK(xⱼ,x)
    # Simplified version: use kernel-based scores
    # K_normalized = K ./ (norm(K) + 1e-8)  # Normalize to prevent explosion
    Y = y* y'
    margins = Y .* K   
    
    # Smooth hinge loss using squared max
    smooth_hinge = mean(max.(0, 1 .- margins).^2)
    
    # Regularization (approximate ||w||² from kernel)
    # regularization = 1/2 * λ * norm(K)
    
    # total_loss = regularization + (C / n) * smooth_hinge
    # println("Smooth Hinge Loss: ", smooth_hinge)
    # total_loss = C * smooth_hinge + regularization
    total_loss = C * smooth_hinge 
    
    return total_loss
end


# Main experiment
function run_gradient_descent_demo(;seed::Union{Int, Nothing}=nothing)
    println("=== Quantum Kernel Gradient Descent Demo ===\n")
   
    # Create dataset
    # data = load_iris_binary()
    data = produce_data(
               DataConfig(
                    n_samples=5,
                    data_params=RBFDataParams(
                        gamma=10.0,
                        n_support_vectors=2,
                    ),
                    seed=seed
                )
            )

    # data = produce_data(
    #            DataConfig(
    #                 n_samples=1000,
    #                 data_params=QuantumPauliDataParams(
    #                     n_qubits=2,
    #                     paulis=["Z", "YX", "ZY"],
    #                     entanglement="linear",
    #                     reps=2,
    #                     gap=0.1,
    #                     grid_points_per_dim=100,
    #                 ),
    #                 seed=seed
    #             )
    #         )
    
    X_train = permutedims(data[:X_train])
    X_test = permutedims(data[:X_test])
    y_train = data[:y_train]
    y_test = data[:y_test]
   
    # Create simple quantum feature map
    n_qubits = 4
    n_features = 2
    n_layers = 4
    entanglement = linear

    if isnothing(seed)
        seed = rand(Int)
    end

    Random.seed!(seed)
    
    feature_map = ReuploadingCircuit(n_qubits, n_features, n_layers, entanglement)
    
    # Initialize with small random parameters
    assign_random_params!(feature_map, range=(-π, π) .* (1/n_qubits), seed=seed)
    println("Initial weights and biases")
    
    # Create kernel
    kernel = FidelityKernel(feature_map)
    K_train_initial = TQK.evaluate(kernel, X_train)
    model = svmtrain(K_train_initial, y_train, kernel=Kernel.Precomputed, verbose=false)
    p_separator_initial = plot_svm_decision_boundary(model, kernel, data)
    
    println("\nQuantum circuit configuration:")
    println("  Qubits: $n_qubits")
    println("  Layers: $n_layers")
    println("  Total parameters: $(2 * n_params(feature_map)) ($(n_params(feature_map)) weights + $(n_params(feature_map)) biases)")
    println("  Seed: $seed")


    # loss_fn = K -> -centered_kernel_target_alignment(K, y_train)
    # loss_fn = K -> kernel_target_alignment(K, y_train)
    # loss_fn = K -> smooth_svm_loss_tanh(K, y_train; C=1.0, λ=0.01, β=5.0)
    loss_fn = K -> smooth_hinge_loss(K, y_train; C = 1.0, λ=0.1)
    trainer = QuantumKernelTrainer(
        kernel,
        loss_fn,
        X_train,
        y_train,
    )

    weights, biases = get_params(trainer.kernel.feature_map)
    initial_params = vcat(weights, biases)
    verify_gradients_detailed(trainer, initial_params; n_check=10)
    compare_gradient_methods(trainer, initial_params)

    # Train

    # Track metrics during training
    metrics = Dict(
        "iteration" => Int[],
        "loss" => Float64[],
        "train_accuracy" => Float64[],
        "test_accuracy" => Float64[],
        "weight_gradient_norms" => Float64[],
        "bias_gradient_norms" => Float64[],
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

        
        # get gradients
        # 
        # Pack gradients
        norm_weight_grads = trainer._grad_norms[1]
        norm_bias_grads = trainer._grad_norms[2]
               
        # Store metrics
        push!(metrics["iteration"], iter)
        push!(metrics["loss"], loss)
        push!(metrics["train_accuracy"], train_acc)
        push!(metrics["test_accuracy"], test_acc)
        push!(metrics["weight_gradient_norms"], norm_weight_grads)
        push!(metrics["bias_gradient_norms"], norm_bias_grads)
        
        if iter % 5 == 0 || iter == 1
            @printf "Iter %d: loss=%.4f, train_acc=%.1f%%, test_acc=%.1f%%, norm(weights_g)=%e, norm(bias_g)= %e\n" iter loss (train_acc * 100) (test_acc * 100) norm_weight_grads norm_bias_grads
        end
    end

    println("\nStarting training ...\n")
    # ITERS = 1
    # sol = train!(trainer,
    #              optimizer= OptimizationOptimisers.AMSGrad(eta=0.001),
    #              # optimizer= LBFGS(),
    #              iterations=ITERS,
    #              #callback=(state, loss) -> losses[state.iter + 1] = loss
    #              callback=evaluation_callback
    #          )
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
    return kernel, p, p_acc, p_data, model, metrics, data, p_separator_initial
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


"""
Comprehensive gradient verification with detailed diagnostics
"""
function verify_gradients_detailed(trainer, params; ε=1e-5, n_check=5)
    nparams = n_params(trainer.kernel.feature_map)
    X = trainer.X
    
    println("\n" * "="^70)
    println("GRADIENT VERIFICATION")
    println("="^70)
    println("Checking first $n_check of $(2*nparams) parameters")
    println("Step size ε = $ε")
    
    # Test 1: Parameter assignment works
    println("\n[1] Testing parameter assignment...")
    test_params = randn(2*nparams)
    assign_params!(trainer.kernel.feature_map, test_params[1:nparams], test_params[nparams+1:end])
    w_check, b_check = get_params(trainer.kernel.feature_map)
    @assert w_check ≈ test_params[1:nparams] "Weight assignment failed"
    @assert b_check ≈ test_params[nparams+1:end] "Bias assignment failed"
    println("✅ Parameters assign correctly")
    
    # Test 2: Parameters affect kernel values
    println("\n[2] Testing parameter→kernel sensitivity...")
    assign_params!(trainer.kernel.feature_map, params[1:nparams], params[nparams+1:end])
    k_base = TQK.evaluate(trainer.kernel, X[1,:], X[2,:])
    
    test_params_perturbed = copy(params)
    test_params_perturbed[1] += 0.1  # Large perturbation
    assign_params!(trainer.kernel.feature_map, test_params_perturbed[1:nparams], test_params_perturbed[nparams+1:end])
    k_pert = TQK.evaluate(trainer.kernel, X[1,:], X[2,:])
    
    println("K(x₁,x₂) change: $(abs(k_pert - k_base))")
    @info abs(k_pert - k_base) > 1e-6 "Kernel insensitive to parameters!"
    println("✅ Parameters affect kernel")
    
    # Test 3: Parameters affect loss
    println("\n[3] Testing parameter→loss sensitivity...")
    assign_params!(trainer.kernel.feature_map, params[1:nparams], params[nparams+1:end])
    K_base = TQK.evaluate(trainer.kernel, X; workspace=trainer.workspace)
    loss_base = trainer.loss_fn(K_base)
    
    assign_params!(trainer.kernel.feature_map, test_params_perturbed[1:nparams], test_params_perturbed[nparams+1:end])
    K_pert = TQK.evaluate(trainer.kernel, X; workspace=trainer.workspace)
    loss_pert = trainer.loss_fn(K_pert)
    
    println("Loss change: $(abs(loss_pert - loss_base))")
    @info abs(loss_pert - loss_base) > 1e-6 "Loss insensitive to parameters!"
    println("✅ Parameters affect loss")
    
    # Test 4: Gradient comparison
    println("\n[4] Computing gradients...")
    println("-"^70)
    println("Param | Finite Diff | Analytic | Rel Error | Status")
    println("-"^70)
    
    # Get analytic gradients once
    assign_params!(trainer.kernel.feature_map, params[1:nparams], params[nparams+1:end])
    K = TQK.evaluate(trainer.kernel, X; workspace=trainer.workspace)
    _, (grad_w_analytic, grad_b_analytic) = hybrid_loss_gradient(K, X, trainer.kernel, trainer.loss_fn)
    
    max_error = 0.0
    
    for i in 1:min(n_check, length(params))
        # Finite difference
        params_plus = copy(params)
        params_plus[i] += ε
        assign_params!(trainer.kernel.feature_map, params_plus[1:nparams], params_plus[nparams+1:end])
        K_plus = TQK.evaluate(trainer.kernel, X; workspace=trainer.workspace)
        loss_plus = trainer.loss_fn(K_plus)
        
        params_minus = copy(params)
        params_minus[i] -= ε
        assign_params!(trainer.kernel.feature_map, params_minus[1:nparams], params_minus[nparams+1:end])
        K_minus = TQK.evaluate(trainer.kernel, X; workspace=trainer.workspace)
        loss_minus = trainer.loss_fn(K_minus)
        
        fd_grad = (loss_plus - loss_minus) / (2ε)
        
        # Analytic
        analytic_grad = i <= nparams ? grad_w_analytic[i] : grad_b_analytic[i-nparams]
        
        # Error
        rel_error = abs(fd_grad - analytic_grad) / (abs(fd_grad) + abs(analytic_grad) + 1e-8)
        max_error = max(max_error, rel_error)
        
        status = rel_error < 0.01 ? "✅" : "❌"
        @printf "%5d | %+.4e | %+.4e | %.4e | %s\n" i fd_grad analytic_grad rel_error status
    end
    
    println("-"^70)
    println("Maximum relative error: $(max_error)")
    println("="^70)
    
    return max_error < 0.01
end

"""
Compare two gradient implementations
"""
function compare_gradient_methods(trainer, params)
    nparams = n_params(trainer.kernel.feature_map)
    assign_params!(trainer.kernel.feature_map, params[1:nparams], params[nparams+1:end])
    
    K = TQK.evaluate(trainer.kernel, trainer.X; workspace=trainer.workspace)
    
    # Method 1: Your adjoint implementation
    loss1, (grad_w1, grad_b1) = loss_gradient(
        trainer.kernel, K, trainer.loss_fn, trainer.X, trainer.workspace
    )
    
    # Method 2: Hybrid Zygote implementation
    loss2, (grad_w2, grad_b2) = hybrid_loss_gradient(
        K, trainer.X, trainer.kernel, trainer.loss_fn
    )
    
    println("\n" * "="^70)
    println("GRADIENT METHOD COMPARISON")
    println("="^70)
    println("Loss (adjoint): ", loss1)
    println("Loss (zygote):  ", loss2)
    println("Loss difference: ", abs(loss1 - loss2))
    println()
    println("Weight grad norm (adjoint): ", norm(grad_w1))
    println("Weight grad norm (zygote):  ", norm(grad_w2))
    println("Weight grad difference:     ", norm(grad_w1 - grad_w2))
    println()
    println("Bias grad norm (adjoint): ", norm(grad_b1))
    println("Bias grad norm (zygote):  ", norm(grad_b2))
    println("Bias grad difference:     ", norm(grad_b1 - grad_b2))
    println("="^70)
    
    return (w_match = norm(grad_w1 - grad_w2) < 1e-5,
            b_match = norm(grad_b1 - grad_b2) < 1e-5)
end
