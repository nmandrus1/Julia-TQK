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

function norm_kernel_alignment_loss(K::Matrix, y::Vector)
    y_outer = y * y'  # Target kernel based on labels

    # Normalize K (e.g., Frobenius norm)
    K_norm = K ./ norm(K, 2) # Using Frobenius norm (norm(K, 2) is the spectral norm, 
                             # norm(K) or norm(K, :frob) for Frobenius)
                             # Let's assume Frobenius for now for simplicity of discussion
                             # K_norm = K ./ norm(K, :frob)

    # Normalize y_outer if desired (usually y_outer is the target, so fixed)
    # y_outer_norm = y_outer ./ norm(y_outer, :frob) # Only if y_outer needs dynamic normalization

    # MSE loss between normalized kernels
    # Using y_outer directly if it's your target, not its normalized version
    loss = sum((K_norm - y_outer).^2) # Or (K_norm - y_outer_norm).^2 if y_outer is also normalized
    return loss
end

# Main experiment
function run_gradient_descent_demo(;seed::Union{Int, Nothing}=nothing)
    println("=== Quantum Kernel Gradient Descent Demo ===\n")
    
    # Create dataset
    X, y = create_simple_dataset(500)
    println("Created dataset with $(size(X, 1)) samples and $(size(X, 2)) features")
    
    # Create simple quantum feature map
    n_qubits = 4
    n_features = 2
    n_layers = 8
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

    loss_fn = K -> kernel_alignment_loss(K, y)
    trainer = QuantumKernelTrainer(
        kernel,
        loss_fn,
        X,
        y,
    )

    # Train
    ITERS = 250
    losses = Vector{Float64}(undef, ITERS+1)
    function cb(state, loss)
        println("Iter $(state.iter): loss = $loss")
        losses[state.iter + 1] = loss
    end

    println("\nStarting training ...\n")
    sol = train!(trainer,
                 optimizer=BFGS(),
                 iterations=ITERS,
                 #callback=(state, loss) -> losses[state.iter + 1] = loss
                 callback=cb
             )
    #return sol

   
    # Plot results
    println("\nCreating loss plot...")
    p = Plots.plot(1:length(losses), losses, 
             xlabel="Iterations", 
             ylabel="Loss", 
             title="Quantum Kernel Training Loss",
             label="Training Loss",
             lw=2,
             marker=:circle,
             markersize=2,
             markerstrokewidth=0)
    
    # #display(p)
    
    # # Show improvement
    # println("\n=== Results Summary ===")
    # println("Initial loss: $(round(losses[1], digits=6))")
    # println("Final loss:   $(round(losses[end], digits=6))")
    # println("Improvement:  $(round(100 * (1 - losses[end]/losses[1]), digits=2))%")
    
    # Visualize kernel matrices
    println("\nComputing kernel matrices...")
    K_final = evaluate(kernel, X)  # With trained params
    
    p1 = heatmap(K_initial, title="Initial Kernel", c=:viridis, clim=(0,1))
    p2 = heatmap(K_final, title="Trained Kernel", c=:viridis, clim=(0,1))
    p3 = heatmap(y*y', title="Target Pattern", c=:viridis, clim=(0,1))
    
    p_kernels = Plots.plot(p1, p2, p3, layout=(1,3), size=(900,300))
    #display(p_kernels)
    
    # return losses, kernel, p, p_kernels
    return kernel, p, p_kernels
end

# Run the demo
#losses, trained_kernel, p, p_kernesl = run_gradient_descent_demo()
