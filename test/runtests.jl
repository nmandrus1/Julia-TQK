using Test
using LinearAlgebra
using Statistics
using TQK
using Yao, YaoBlocks

# Utility functions for testing and validation
"""
    is_positive_semidefinite(K::Matrix; tol=1e-10)

Check if kernel matrix is positive semi-definite.
"""
function is_positive_semidefinite(K::Matrix; tol=1e-10)
    eigenvalues = eigvals(Symmetric(K))
    return all(λ -> λ >= -tol, eigenvalues)
end

"""
    kernel_alignment(K1::Matrix, K2::Matrix)

Compute kernel alignment between two kernel matrices.
"""
function kernel_alignment(K1::Matrix, K2::Matrix)
    return tr(K1 * K2) / (sqrt(tr(K1 * K1)) * sqrt(tr(K2 * K2)))
end

# Test suite
@testset "TQK Test Suite" begin
    @testset "FidelityKernel Tests" begin
        # Simple test feature map
        function test_feature_map(x::Vector, params)
            n_qubits = 2
            circuit = chain(n_qubits)
            circuit = chain(circuit, put(n_qubits, 1 => Ry(x[1])))
            if length(x) > 1
                circuit = chain(circuit, put(n_qubits, 2 => Ry(x[2])))
            end
            circuit = chain(circuit, control(1, 2 => X))
            return circuit
        end
        
        # Create kernel
        kernel = FidelityKernel(test_feature_map, 2, nothing)
        
        @testset "Single kernel evaluation" begin
            x = [0.5, 0.3]
            y = [0.7, 0.2]
            
            k_val = evaluate(kernel, x, y)
            @test 0 <= k_val <= 1
            
            # Self-similarity should be close to 1
            k_self = evaluate(kernel, x, x)
            @test k_self ≈ 1.0 atol=1e-10
        end
        
        @testset "Kernel matrix properties" begin
            X = rand(2, 5)  # 5 samples
            K = evaluate(kernel, X)
            
            # Check symmetry
            @test K ≈ K' atol=1e-10
            
            # Check positive semi-definite
            @test is_positive_semidefinite(K)
            
            # Check diagonal elements
            for i in 1:size(K, 1)
                @test K[i, i] ≈ 1.0 atol=1e-10
            end
        end
        
        @testset "Two dataset kernel" begin
            X_train = rand(2, 4)
            X_test = rand(2, 3)
            
            K = evaluate(kernel, X_train, X_test)
            @test size(K) == (4, 3)
            
            # Verify individual elements
            for i in 1:4, j in 1:3
                k_ij = evaluate(kernel, X_train[:, i], X_test[:, j])
                @test K[i, j] ≈ k_ij atol=1e-10
            end
        end
        
        @testset "Caching" begin
            kernel_cached = FidelityKernel(test_feature_map, 2, nothing, use_cache=true)
            
            x = [0.5, 0.3]
            y = [0.7, 0.2]
            
            # First evaluation
            k1 = evaluate(kernel_cached, x, y)
            
            # Second evaluation (should use cache)
            k2 = evaluate(kernel_cached, x, y)
            
            @test k1 == k2
            
            # Clear cache
            clear_cache!(kernel_cached)
        end
    end
    
    @testset "Reuploading Circuit Tests" begin
        @testset "Circuit Construction" begin
            opts = ReuploadBuilderOpts(3, 4, 2, linear)
            circuit, params = build_reuploading_circuit(opts)
            
            # Check parameter counts
            shapes = get_parameter_shapes(params)
            @test shapes.n_weights > 0
            @test shapes.n_biases > 0
            @test shapes.n_inputs == 4
            
            # Check circuit properties
            @test nqubits(circuit) == 3
        end
        
        @testset "Parameter Initialization" begin
            opts = ReuploadBuilderOpts(2, 3, 1, linear)
            circuit, params = build_reuploading_circuit(opts)
            
            w_init, θ_init = random_parameters(params, seed=42)
            
            shapes = get_parameter_shapes(params)
            @test length(w_init) == shapes.n_weights
            @test length(θ_init) == shapes.n_biases
            
            # Check values are in expected range
            @test all(w -> -pi <= w <= pi, w_init)
            @test all(θ -> -pi <= θ <= pi, θ_init)
        end
    end
end

# Performance benchmarking function (not part of test suite)
function benchmark_kernel(; n_samples=10, n_features=4, n_qubits=4)
    println("\n" * "="^60)
    println("Benchmarking FidelityKernel")
    println("Parameters: $n_samples samples, $n_features features, $n_qubits qubits")
    println("="^60)
    
    # Generate random data
    X = rand(n_features, n_samples)
    
    # Create a more realistic feature map using reuploading circuit
    function reupload_feature_map(x::Vector, params)
        # Simple reuploading circuit for benchmarking
        circuit = chain(n_qubits)
        
        # Data encoding layer
        for (i, xi) in enumerate(x)
            if i <= n_qubits
                circuit = chain(circuit, put(n_qubits, i => Ry(2π * xi)))
            end
        end
        
        # Entangling layer
        for i in 1:(n_qubits-1)
            circuit = chain(circuit, control(i, i+1 => X))
        end
        
        # Second encoding layer
        for (i, xi) in enumerate(x)
            if i <= n_qubits
                circuit = chain(circuit, put(n_qubits, i => Rz(2π * xi)))
            end
        end
        
        return circuit
    end
    
    # Create kernel
    kernel = FidelityKernel(reupload_feature_map, n_qubits, nothing, parallel=true)
    
    # Warm-up
    println("\nWarming up...")
    evaluate(kernel, X[:, 1:min(2, n_samples)])
    
    # Benchmark
    println("\nComputing full kernel matrix...")
    @time K = evaluate(kernel, X)
    
    println("\nKernel matrix statistics:")
    println("  Size: $(size(K))")
    println("  Min value: $(round(minimum(K), digits=6))")
    println("  Max value: $(round(maximum(K), digits=6))")
    println("  Mean diagonal: $(round(mean(diag(K)), digits=6))")
    println("  Is PSD: $(is_positive_semidefinite(K))")
    
    # Check condition number
    cond_num = cond(K)
    println("  Condition number: $(round(cond_num, sigdigits=3))")
    
    return K
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Run the test suite
    println("Running TQK test suite...")
    
    # Optionally run a small benchmark after tests
    if get(ENV, "RUN_BENCHMARK", "false") == "true"
        println("\nRunning benchmark...")
        K = benchmark_kernel(n_samples=5, n_features=3, n_qubits=3)
    end
end
