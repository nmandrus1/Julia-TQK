using Test
using Yao
using Zygote
using LinearAlgebra

# Include the files we just wrote (adjust paths as necessary for your project structure)
include("../src/feature_maps/types.jl")       # Abstract types
include("../src/feature_maps/reupload.jl") # Factory
include("../src/kernels/pure_fidelity.jl")    # Pure Kernel

@testset "Pure Kernel AD Compatibility" begin
    # 1. Setup Data
    n_qubits = 2
    n_features = 4
    n_layers = 1
    n_samples = 5
    
    # Random data and labels (+1/-1)
    X = rand(n_samples, n_features)
    y = sign.(randn(n_samples))
    
    # 2. Setup Config
    config = ReuploadingConfig(n_qubits, n_features, n_layers)
    
    # 3. Initialize Parameters
    #    (Weights + Biases)
    params = randn(config.total_params)
    
    # --- Test 1: Forward Pass ---
    @testset "Forward Pass" begin
        K = compute_kernel_matrix_pure(config, params, X)
        @test size(K) == (n_samples, n_samples)
        @test all(K .>= 0) # Probabilities must be positive
        @test isapprox(diag(K), ones(n_samples)) # Self-fidelity is always 1.0
    end

    # --- Test 2: Gradient Existence ---
    @testset "Backward Pass (Zygote)" begin
        # Can we take the gradient of the loss w.r.t params?
        grads = gradient(params) do p
            kernel_target_alignment_loss(config, p, X, y)
        end
        
        # Did we get a result?
        @test !isnothing(grads[1])
        
        # Is the gradient the correct shape?
        @test length(grads[1]) == config.total_params
        
        # Is it non-zero? (It should be for random params)
        @test norm(grads[1]) > 0.0
        
        println("Gradient Norm: ", norm(grads[1]))
    end
end
