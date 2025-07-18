# test/perf/kernel_benchmarks.jl
using Test
using BenchmarkTools
using ProfileView
using Profile
using TQK  # your package

function profile_kernel(num_qubits::Int = 4, num_features::Int = 6, num_layers::Int = 2, entanglement::EntanglementBlock = linear; num_datapoints::Int=500)
    reup = ReuploadingCircuit(num_qubits, num_features, num_layers, entanglement)
    assign_random_params!(reup)

    kernel = FidelityKernel(reup, use_cache=true, parallel=false)
    X = rand(500, num_features)
    for i in 1:100
        K = zeros(500, 500)
        @profile evaluate!(K, kernel, X)
    end
end


function benchmark(num_qubits::Int = 4, num_features::Int = 6, num_layers::Int = 2, entanglement::EntanglementBlock = linear; num_datapoints::Int=500)
    reup = ReuploadingCircuit(num_qubits, num_features, num_layers, entanglement)

    kernel = FidelityKernel(reup, use_cache=true, parallel=false)
    assign_random_params!(reup, seed=nothing)
    X = rand(500, num_features)

    @benchmark evaluate!(K, $kernel, $X) seconds=60 setup=(K = zeros(500, 500))
end

function benchmark_kernel_evaluate()
    # Setup
    n_qubits = 4
    n_features = 3
    circuit = ReuploadingCircuit(n_qubits, n_features, 2, linear)
    kernel = FidelityKernel(circuit, use_cache=true)
    
    println("\nKernel Evaluation Benchmarks")
    println("="^50)
    
    # Test different sizes
    sizes = [10, 50, 100, 200]
    results = []
    
    for n in sizes
        X = randn(n_features, n)
        
        # Sequential
        kernel.parallel = false
        t_seq = @belapsed evaluate($kernel, $X)
        
        # Parallel
        kernel.parallel = true
        t_par = @belapsed evaluate($kernel, $X)
        
        speedup = t_seq / t_par
        push!(results, (n, t_seq, t_par, speedup))
        
        println("n=$n samples:")
        println("  Sequential: $(round(t_seq*1000, digits=2)) ms")
        println("  Parallel:   $(round(t_par*1000, digits=2)) ms")
        println("  Speedup:    $(round(speedup, digits=2))x")
    end
    
    # Performance assertions
    @testset "Performance Regression" begin
        # Ensure parallel is faster for large matrices
        @test results[end][4] > 1.2  # At least 20% speedup
        
        # Check scaling (should be roughly O(n²))
        t_ratio = results[end][2] / results[1][2]
        n_ratio = (results[end][1] / results[1][1])^2
        @test 0.5 < t_ratio/n_ratio < 2.0  # Within reasonable bounds
    end
    
    return results
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_kernel_evaluate()
end
