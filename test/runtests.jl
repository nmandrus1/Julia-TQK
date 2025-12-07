using Test
using TQK 

# 1. Define the specific test files you want to run
#    (Order matters if some depend on others, but usually they should be independent)
tests = [
    "test_pure_gradients.jl",
    "test_kernel_equivalence.jl",
    "verify_gradients_finite_diff.jl",
    "test_spsa_training.jl"
]

println("Running tests for TQK...")

@testset "All TQK Tests" begin
    for t in tests
        t_path = joinpath(@__DIR__, t)
        println("  -> Running $t ...")
        @testset "$t" begin
            include(t_path)
        end
    end
end
