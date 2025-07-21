using Test
using LinearAlgebra
using Statistics
using Yao, YaoBlocks
using Random

using TQK

# Utility functions
function is_positive_semidefinite(K::Matrix; tol=1e-10)
    eigenvalues = eigvals(Symmetric(K))
    return all(λ -> λ >= -tol, eigenvalues)
end

function kernel_alignment(K1::Matrix, K2::Matrix)
    return tr(K1 * K2) / (sqrt(tr(K1 * K1)) * sqrt(tr(K2 * K2)))
end

@testset "Quantum Kernel Tests" begin
    
    @testset "ReuploadingCircuit Tests" begin
        @testset "Construction" begin
            n_qubits_val = 3
            n_features_val = 4
            n_layers = 2
            
            fm = ReuploadingCircuit(n_qubits_val, n_features_val, n_layers, linear)
            
            @test n_qubits(fm) == n_qubits_val
            @test n_features(fm) == n_features_val
            @test n_parameters(fm) == nparameters(fm.circuit)
            
            # Check initialization
            @test all(fm.weights .== 0.0)
            @test all(fm.biases .== 0.0)
            @test length(fm.gate_features) == n_parameters(fm)
        end
        
        @testset "Different entanglement strategies" begin
            n_qubits_val = 4
            n_features_val = 2
            n_layers = 2
            
            for strategy in [linear, all_to_all, alternating]
                fm = ReuploadingCircuit(n_qubits_val, n_features_val, n_layers, strategy)
                @test n_qubits(fm) == n_qubits_val
                @test n_features(fm) == n_features_val
            end
        end
        
        @testset "Parameter assignment" begin
            fm = ReuploadingCircuit(2, 3, 1, linear)
            n_params = n_parameters(fm)
            
            # Test assign_params! with weights and biases
            weights = rand(n_params)
            biases = rand(n_params)
            assign_params!(fm, weights, biases)
            
            @test fm.weights == weights
            @test fm.biases == biases
            
            # Test get_params
            w, b = get_params(fm)
            @test w == weights
            @test b == biases
        end
        
        @testset "Random parameter initialization" begin
            fm = ReuploadingCircuit(2, 3, 1, linear)
            
            # Test with default range
            assign_random_params!(fm, seed=42)
            @test all(-π .<= fm.weights .<= π)
            @test all(-π .<= fm.biases .<= π)
            
            # Test with custom range
            assign_random_params!(fm, (0.0, 1.0), seed=42)
            @test all(0.0 .<= fm.weights .<= 1.0)
            @test all(0.0 .<= fm.biases .<= 1.0)
            
            # Test reproducibility
            fm2 = ReuploadingCircuit(2, 3, 1, linear)
            assign_random_params!(fm2, seed=42)
            assign_random_params!(fm, seed=42)
            @test fm.weights == fm2.weights
            @test fm.biases == fm2.biases
        end
        
        @testset "Angle computation" begin
            fm = ReuploadingCircuit(2, 2, 1, linear)
            
            # Set known parameters
            weights = [1.0, 2.0, 3.0, 4.0]
            biases = [0.1, 0.2, 0.3, 0.4]
            assign_params!(fm, weights, biases)
            
            # Test angle computation
            x = [0.5, 0.3]
            compute_angles!(fm, x)
            angles = fm.angles
            
            # Verify angles are computed correctly
            for i in 1:length(angles)
                expected = weights[i] * x[fm.gate_features[i]] + biases[i]
                @test angles[i] ≈ expected
            end
        end
        
        @testset "Input mapping" begin
            fm = ReuploadingCircuit(2, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            
            x = rand(2)
            
            # Test that map_inputs! modifies the circuit
            original_params = parameters(fm.circuit)
            map_inputs!(fm, x)
            new_params = parameters(fm.circuit)
            
            @test length(original_params) == length(new_params)
            @test !all(original_params .== new_params)  # Parameters should change
        end
    end
    
    @testset "FidelityKernel Tests" begin
        @testset "Basic kernel evaluation" begin
            fm = ReuploadingCircuit(2, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=false)
            
            x = [0.5, 0.3]
            y = [0.7, 0.2]
            
            # Test kernel value is in valid range
            k_val = evaluate(kernel, x, y)
            @test 0 <= k_val <= 1
            
            # Test self-similarity
            k_self = evaluate(kernel, x, x)
            @test k_self ≈ 1.0 atol=1e-10
            
            # Test symmetry
            k_xy = evaluate(kernel, x, y)
            k_yx = evaluate(kernel, y, x)
            @test k_xy ≈ k_yx atol=1e-10
        end
        
        @testset "Kernel matrix properties" begin
            fm = ReuploadingCircuit(3, 2, 2, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=false)
            
            # Generate test data - ROWS are samples
            n_samples = 5
            X = rand(n_samples, 2)  # 5 samples × 2 features
            
            # Test allocating version
            K = evaluate(kernel, X)
            
            # Check dimensions
            @test size(K) == (n_samples, n_samples)
            
            # Check symmetry
            @test K ≈ K' atol=1e-10
            
            # Check positive semi-definite
            @test is_positive_semidefinite(K)
            
            # Check diagonal elements (self-similarity)
            for i in 1:n_samples
                @test K[i, i] ≈ 1.0 atol=1e-10
            end
            
            # Check off-diagonal elements
            for i in 1:n_samples, j in 1:n_samples
                if i != j
                    @test 0 <= K[i, j] <= 1
                end
            end
        end
        
        @testset "In-place kernel matrix evaluation" begin
            fm = ReuploadingCircuit(3, 2, 2, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=false)
            
            # Generate test data
            n_samples = 5
            X = rand(n_samples, 2)
            
            # Pre-allocate matrix
            K_inplace = zeros(n_samples, n_samples)
            
            # Test in-place version
            evaluate!(K_inplace, kernel, X)
            
            # Compare with allocating version
            K_alloc = evaluate(kernel, X)
            @test K_inplace ≈ K_alloc atol=1e-10
            
            # Test that it actually modifies in-place
            K_test = ones(n_samples, n_samples) * 999.0
            evaluate!(K_test, kernel, X)
            @test K_test ≈ K_alloc atol=1e-10
            @test !any(K_test .≈ 999.0)  # All values should be changed
        end
        
        @testset "Two dataset kernel matrix" begin
            fm = ReuploadingCircuit(2, 3, 1, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=false)
            
            # ROWS are samples
            X_train = rand(4, 3)  # 4 samples × 3 features
            X_test = rand(3, 3)   # 3 samples × 3 features
            
            # Test allocating version
            K = evaluate(kernel, X_train, X_test)
            
            # Check dimensions
            @test size(K) == (4, 3)
            
            # Verify individual elements
            for i in 1:4, j in 1:3
                k_ij = evaluate(kernel, X_train[i, :], X_test[j, :])
                @test K[i, j] ≈ k_ij atol=1e-10
            end
        end
        
        @testset "In-place two dataset evaluation" begin
            fm = ReuploadingCircuit(2, 3, 1, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=false)
            
            X_train = rand(4, 3)
            X_test = rand(3, 3)
            
            # Pre-allocate matrix
            K_inplace = zeros(4, 3)
            
            # Test in-place version
            evaluate!(K_inplace, kernel, X_train, X_test)
            
            # Compare with allocating version
            K_alloc = evaluate(kernel, X_train, X_test)
            @test K_inplace ≈ K_alloc atol=1e-10
            
            # Test dimension checking
            K_wrong = zeros(3, 3)  # Wrong dimensions
            @test_throws AssertionError evaluate!(K_wrong, kernel, X_train, X_test)
        end
        
        @testset "Caching functionality" begin
            fm = ReuploadingCircuit(2, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            
            kernel = FidelityKernel(fm, use_cache=true, cache_size=10)
            
            x = [0.5, 0.3]
            y = [0.7, 0.2]
            
            # First evaluation
            k1 = evaluate(kernel, x, y)
            
            # Cache lookup is commented out in the current implementation
            # so we skip cache-related tests for now
            
            # Clear cache should still work without error
            clear_cache!(kernel)
        end
        
        @testset "Edge cases" begin
            # Single qubit, single feature
            fm = ReuploadingCircuit(1, 1, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
            
            x = [0.5]
            k = evaluate(kernel, x, x)
            @test k ≈ 1.0 atol=1e-10
            
            # Zero input
            fm2 = ReuploadingCircuit(2, 2, 1, linear)
            assign_random_params!(fm2, seed=42)
            kernel2 = FidelityKernel(fm2, use_cache=false)
            
            x_zero = [0.0, 0.0]
            k_zero = evaluate(kernel2, x_zero, x_zero)
            @test k_zero ≈ 1.0 atol=1e-10
        end
    end
    
    @testset "Integration Tests" begin
        @testset "Different circuit configurations" begin
            configs = [
                (n_qubits=2, n_features=2, n_layers=1, ent=linear),
                (n_qubits=3, n_features=4, n_layers=2, ent=linear),
                (n_qubits=4, n_features=2, n_layers=3, ent=all_to_all),
            ]
            
            for config in configs
                fm = ReuploadingCircuit(config.n_qubits, config.n_features, 
                                      config.n_layers, config.ent)
                assign_random_params!(fm, seed=42)
                kernel = FidelityKernel(fm, use_cache=false)
                
                # Generate test data - ROWS are samples
                n_samples = 3
                X = rand(n_samples, config.n_features)
                
                # Test both allocating and in-place versions
                K_alloc = evaluate(kernel, X)
                K_inplace = zeros(n_samples, n_samples)
                evaluate!(K_inplace, kernel, X)
                
                @test K_alloc ≈ K_inplace atol=1e-10
                @test is_positive_semidefinite(K_alloc)
                @test all(diag(K_alloc) .≈ 1.0)
            end
        end       
    end
    
    @testset "Performance Benchmarks" begin
        @testset "Kernel matrix computation time" begin
            fm = ReuploadingCircuit(3, 3, 2, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=true)
            
            # Small dataset - ROWS are samples
            X_small = rand(10, 3)  # 10 samples × 3 features
            
            # Compare allocating vs in-place performance
            K_alloc = @time evaluate(kernel, X_small)
            
            K_inplace = zeros(10, 10)
            @time evaluate!(K_inplace, kernel, X_small)
            
            @test K_alloc ≈ K_inplace atol=1e-10
            @test is_positive_semidefinite(K_alloc)
        end
    end

    # Add these tests to the existing runtests.jl file

    @testset "Memory-Aware Tiling Tests" begin
        @testset "Tile size calculation" begin
            # Test tile size calculation for different qubit counts and memory budgets
            @test calculate_tile_size(3, 1.0) > 1000  # 3 qubits should fit many statevectors
            @test calculate_tile_size(10, 1.0) > 100  # 10 qubits still fits reasonable amount
        
            # Test with split factor
            full_tile = calculate_tile_size(5, 1.0, 1.0)
            half_tile = calculate_tile_size(5, 1.0, 0.5)
            @test half_tile ≈ full_tile ÷ 2
        
            # Test error for insufficient memory
            @test_throws ErrorException calculate_tile_size(20, 0.001)  # 20 qubits won't fit
        end
    
        @testset "Tiled evaluation correctness" begin
            fm = ReuploadingCircuit(4, 3, 2, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
        
            # Test data
            X = rand(50, 3)
        
            # Evaluate with different memory budgets (different tile sizes)
            K_large_tiles = zeros(50, 50)
            evaluate!(K_large_tiles, kernel, X, memory_budget_gb=4.0)
        
            K_small_tiles = zeros(50, 50)
            evaluate!(K_small_tiles, kernel, X, memory_budget_gb=0.1)
        
            # Results should be identical regardless of tile size
            @test K_large_tiles ≈ K_small_tiles atol=1e-10
            @test is_positive_semidefinite(K_large_tiles)
            @test all(diag(K_large_tiles) .≈ 1.0)
        end
    
        @testset "Asymmetric tiled evaluation" begin
            fm = ReuploadingCircuit(3, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
        
            X = rand(30, 2)
            Y = rand(25, 2)
        
            # Test with different memory budgets
            K_large = zeros(30, 25)
            evaluate!(K_large, kernel, X, Y, memory_budget_gb=2.0)
        
            K_small = zeros(30, 25)
            evaluate!(K_small, kernel, X, Y, memory_budget_gb=0.05)
        
            @test K_large ≈ K_small atol=1e-10
        
            # Verify individual elements
            for i in 1:5, j in 1:5  # Check subset
                k_ij = evaluate(kernel, X[i, :], Y[j, :])
                @test K_large[i, j] ≈ k_ij atol=1e-10
            end
        end

        @testset "Tile boundary correctness" begin
            fm = ReuploadingCircuit(5, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)

            # Use a small, fixed number of samples
            n_samples = 30
            X = rand(n_samples, 2)

            # Force a small tile size (e.g., 10) by providing a tight memory budget
            # statevec_bytes = (2^5 * sizeof(ComplexF64)) = 512 bytes
            # To get tile_size=10, we need budget for ~10 statevectors: 10 * 512 bytes ≈ 0.000005 GB
            forced_budget_gb = 11 * (2^5 * sizeof(ComplexF64)) / (1024^3)

            K = zeros(n_samples, n_samples)
            evaluate!(K, kernel, X, memory_budget_gb=forced_budget_gb)
    
            # Now your tile_size is small (~10) and n_samples is 30, so tiling is tested
            # without a huge matrix allocation.
    
            K_direct = evaluate(kernel, X) # Get reference matrix
            @test K ≈ K_direct atol=1e-10
            @test is_positive_semidefinite(K)
        end    
    
        @testset "Memory scaling with qubit count" begin
            # Test that the system handles varying qubit counts correctly
            for n_qubits in [2, 4, 6, 8]
                fm = ReuploadingCircuit(n_qubits, 2, 1, linear)
                assign_random_params!(fm, seed=42)
                kernel = FidelityKernel(fm, use_cache=false)
            
                # Use consistent small dataset
                n_samples = 20
                X = rand(n_samples, 2)
            
                # Calculate memory requirement
                bytes_per_statevec = (2^n_qubits) * sizeof(ComplexF64)
                required_gb = (n_samples * bytes_per_statevec) / (1024^3)
            
                println("Testing $n_qubits qubits, requires $(required_gb) GB for full caching")
            
                # Test with limited memory (forces tiling for larger qubit counts)
                K = zeros(n_samples, n_samples)
                evaluate!(K, kernel, X, memory_budget_gb=0.1)
            
                # Verify correctness
                @test is_positive_semidefinite(K)
                @test all(diag(K) .≈ 1.0)
            
                # Check a few elements directly
                for i in 1:min(3, n_samples), j in 1:min(3, n_samples)
                    k_direct = evaluate(kernel, X[i, :], X[j, :])
                    @test K[i, j] ≈ k_direct atol=1e-10
                end
            end
        end
    
        @testset "Edge cases with tiny memory budgets" begin
            fm = ReuploadingCircuit(3, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
        
            X = rand(100, 2)
        
            # Test with memory budget that forces tile_size = 1
            min_memory_gb = (2^3 * sizeof(ComplexF64) * 2) / (1024^3)  # Just enough for 2 statevectors
        
            K = zeros(10, 10)  # Only compute small subset
            X_subset = X[1:10, :]
        
            # Should still work correctly even with minimal tiling
            evaluate!(K, kernel, X_subset, memory_budget_gb=min_memory_gb)
        
            @test is_positive_semidefinite(K)
            @test all(diag(K) .≈ 1.0)
        end
    

        @testset "Performance with different tile sizes" begin
            fm = ReuploadingCircuit(4, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
        
            X = rand(50, 2)
        
            # Time with large tiles (more memory)
            K_large = zeros(50, 50)
            t_large = @elapsed evaluate!(K_large, kernel, X, memory_budget_gb=2.0)
        
            # Time with small tiles (less memory)
            K_small = zeros(50, 50)
            t_small = @elapsed evaluate!(K_small, kernel, X, memory_budget_gb=0.05)
        
            println("Large tiles time: $t_large s")
            println("Small tiles time: $t_small s")
            println("Overhead ratio: $(t_small/t_large)")
        
            # Verify results match
            @test K_large ≈ K_small atol=1e-10
        end
    
        
        @testset "Statevector reuse in tiling" begin
            fm = ReuploadingCircuit(3, 2, 1, linear)
            assign_random_params!(fm, seed=42)
            kernel = FidelityKernel(fm, use_cache=false)
    
            # Use a small number of samples that will easily fit in one tile
            n_samples = 20
            X = rand(n_samples, 2)
    
            # Provide a generous budget that ensures n_samples fits in one tile
            generous_budget_gb = (n_samples * 2) * (2^3 * sizeof(ComplexF64)) / (1024^3)

            K = zeros(n_samples, n_samples)
            evaluate!(K, kernel, X, memory_budget_gb=generous_budget_gb)
    
            K_direct = evaluate(kernel, X) # Get reference matrix
            @test K ≈ K_direct atol=1e-10
        end
    end
end

