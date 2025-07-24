using DrWatson
@quickactivate "TQK"

using TQK 

# Load required packages
using LinearAlgebra
using Random
using Printf
using Plots

# Set random seed for reproducibility
#Random.seed!(42)

# Create a simple 2D dataset (linearly separable)
function create_simple_dataset(n_samples=10)
    X = zeros(n_samples, 2)
    y = zeros(n_samples)
    
    # Class 0: points around (-1, -1)
    for i in 1:div(n_samples, 2)
        X[i, :] = [-1.0, -1.0] + 0.3 * randn(2)
        y[i] = -1.0
    end
    
    # Class 1: points around (1, 1)
    for i in (div(n_samples, 2) + 1):n_samples
        X[i, :] = [1.0, 1.0] + 0.3 * randn(2)
        y[i] = 1.0
    end
    
    return X, y
end

# Simple MSE loss function for kernel alignment
function kernel_alignment_loss(K::Matrix, y::Vector)
    y_outer = y * y'  # Target kernel based on labels
    
    # MSE loss between normalized kernels
    loss = sum((K- y_outer).^2)
    return loss
end

# Gradient descent function
function train_quantum_kernel!(kernel, X, y; 
                              learning_rate=0.1, 
                              n_epochs=50,
                              verbose=true)
    
    losses = Float64[]
    
    # Get initial parameters
    fm = kernel.feature_map
    weights = copy(fm.weights)
    biases = copy(fm.biases)
    
    if verbose
        println("Initial parameters:")
        println("  Weights: ", round.(weights[1:min(4, length(weights))], digits=3), "...")
        println("  Biases:  ", round.(biases[1:min(4, length(biases))], digits=3), "...")
    end
    
    for epoch in 1:n_epochs
        # Compute loss and gradients
        loss_fn = K -> kernel_alignment_loss(K, y)
        
        # Forward pass and gradient computation
        current_loss, (grad_weights, grad_biases) = loss_gradient(kernel, loss_fn, X, memory_budget_gb=0.001)
        
        # Current loss
        push!(losses, current_loss)
        
        # Gradient descent update
        weights .-= learning_rate * grad_weights
        biases .-= learning_rate * grad_biases
        
        # Update feature ma paramet5rs
        assign_params!(fm, weights, biases)
        
        if verbose && (epoch % 10 == 0 || epoch == 1)
            println("Epoch $epoch: Loss = $(round(current_loss, digits=6))")
            println("  ||∇w|| = $(round(norm(grad_weights), digits=4)), ||∇b|| = $(round(norm(grad_biases), digits=4))")
        end
    end
    
    if verbose
        println("\nFinal parameters:")
        println("  Weights: ", round.(weights[1:min(4, length(weights))], digits=3), "...")
        println("  Biases:  ", round.(biases[1:min(4, length(biases))], digits=3), "...")
    end
    
    return losses
end

# Main experiment
function run_gradient_descent_demo(;seed::Union{Int, Nothing}=nothing)
    println("=== Quantum Kernel Gradient Descent Demo ===\n")
    
    # Create dataset
    X, y = create_simple_dataset(50)
    println("Created dataset with $(size(X, 1)) samples and $(size(X, 2)) features")
    
    # Create simple quantum feature map
    n_qubits = 8
    n_features = 2
    n_layers = 1
    entanglement = linear

    if isnothing(seed)
        seed = rand(Int)
    end

    Random.seed!(seed)
    
    feature_map = ReuploadingCircuit(n_qubits, n_features, n_layers, entanglement)
    
    # Initialize with small random parameters
    assign_random_params!(feature_map, seed=seed)
    
    # Create kernel
    kernel = FidelityKernel(feature_map)
    K_initial = evaluate(kernel, X)  # With random params
    
    println("\nQuantum circuit configuration:")
    println("  Qubits: $n_qubits")
    println("  Layers: $n_layers")
    println("  Total parameters: $(2 * n_params(feature_map)) ($(n_params(feature_map)) weights + $(n_params(feature_map)) biases)")
    println("  Seed: $seed")


    # Train
    println("\nStarting gradient descent...\n")
    losses = train_quantum_kernel!(kernel, X, y, 
                                  learning_rate=0.001, 
                                  n_epochs=1000,
                                  verbose=true)
    
    # Plot results
    println("\nCreating loss plot...")
    p = plot(1:length(losses), losses, 
             xlabel="Epoch", 
             ylabel="Loss", 
             title="Quantum Kernel Training Loss",
             label="Training Loss",
             lw=2,
             marker=:circle,
             markersize=2,
             markerstrokewidth=0)
    
    display(p)
    
    # Show improvement
    println("\n=== Results Summary ===")
    println("Initial loss: $(round(losses[1], digits=6))")
    println("Final loss:   $(round(losses[end], digits=6))")
    println("Improvement:  $(round(100 * (1 - losses[end]/losses[1]), digits=2))%")
    
    # Visualize kernel matrices
    println("\nComputing kernel matrices...")
    K_final = evaluate(kernel, X)  # With trained params
    
    p1 = heatmap(K_initial, title="Initial Kernel", c=:viridis, clim=(0,1))
    p2 = heatmap(K_final, title="Trained Kernel", c=:viridis, clim=(0,1))
    p3 = heatmap(y*y', title="Target Pattern", c=:viridis, clim=(0,1))
    
    p_kernels = plot(p1, p2, p3, layout=(1,3), size=(900,300))
    display(p_kernels)
    
    return losses, kernel, p, p_kernels
end

# Run the demo
losses, trained_kernel = run_gradient_descent_demo()
