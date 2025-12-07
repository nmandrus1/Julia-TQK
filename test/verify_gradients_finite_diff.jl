using TQK
using Test
using Yao
using Zygote
using LinearAlgebra

# Include your source files
# include("../src/feature_maps/types.jl")
# include("../src/feature_maps/reupload.jl")
# include("../src/kernels/pure_fidelity.jl")

"""
    finite_difference_gradient(f, x; ε=1e-5)

Computes the gradient of scalar function f at x using central finite differences.
"""
function finite_difference_gradient(f, x::Vector{Float64}; ε=1e-5)
    grad = zeros(length(x))
    for i in 1:length(x)
        # Perturb +ε
        x_plus = copy(x)
        x_plus[i] += ε
        y_plus = f(x_plus)
        
        # Perturb -ε
        x_minus = copy(x)
        x_minus[i] -= ε
        y_minus = f(x_minus)
        
        # Central difference
        grad[i] = (y_plus - y_minus) / (2ε)
    end
    return grad
end

@testset "Gradient Correctness Verification" begin
    # 1. Setup Data
    n_qubits = 2
    n_features = 4
    n_layers = 2 # slightly deeper to ensure chain rule works through layers
    n_samples = 3 # keep small for speed
    
    X = rand(n_samples, n_features)
    y = sign.(randn(n_samples))
    
    config = ReuploadingConfig(n_qubits, n_features, n_layers)
    params = randn(config.total_params)
    
    # Define the loss closure for easy calling
    loss_fn(p) = variational_kta_loss(config, p, X, y)

    println("Computing Zygote Gradient...")
    t_zygote = @elapsed begin
        grad_zygote = gradient(loss_fn, params)[1]
    end
    println("  -> Done in $t_zygote seconds.")

    println("Computing Finite Difference Gradient...")
    t_fd = @elapsed begin
        grad_fd = finite_difference_gradient(loss_fn, params)
    end
    println("  -> Done in $t_fd seconds.")

    # 3. Compare
    # We use a relative tolerance (rtol) because gradients can vary in magnitude.
    # 1e-4 is standard for Finite Diff comparison vs Float64 AD.
    
    @testset "Element-wise Comparison" begin
        # Check that they are close
        @test isapprox(grad_zygote, grad_fd, rtol=1e-4, atol=1e-5)
        
        # Calculate maximum error
        diff = abs.(grad_zygote .- grad_fd)
        max_err = maximum(diff)
        println("Max Absolute Error: ", max_err)
        
        # If this passes, Zygote is accurately tracking the chain rule through 
        # your feature map, entanglement, and kernel calculation.
    end
end
