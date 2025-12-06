
using Test
using Yao
using LinearAlgebra

# Include necessary files (adjust paths if needed)
# Ensure TQK.jl or these specific files are loaded
include("../src/feature_maps/types.jl")
include("../src/feature_maps/reupload.jl")
include("../src/kernels/pure_fidelity.jl")      # Phase 1 Method
include("../src/kernels/compute_uncompute.jl")   # Phase 2 Method

@testset "Kernel Math Equivalence" begin
    # 1. Setup
    n_qubits = 2
    n_features = 4
    n_layers = 2
    
    config = ReuploadingConfig(n_qubits, n_features, n_layers)
    params = randn(config.total_params)
    
    # Create two random data points
    x = rand(n_features)
    y = rand(n_features)
    
    # 2. Method 1: Statevector Overlap (The "Simulator Shortcut")
    #    K = |<ψ(x)|ψ(y)>|²
    circ_x = build_circuit(config, params, x)
    circ_y = build_circuit(config, params, y)
    ψ_x = apply(zero_state(n_qubits), circ_x)
    ψ_y = apply(zero_state(n_qubits), circ_y)
    
    fidelity_phase1 = abs2(dot(state(ψ_x), state(ψ_y)))
    
    # 3. Method 2: Compute-Uncompute (The "Hardware Reality")
    #    K = |<0| U†(x)U(y) |0>|²
    fidelity_phase2 = compute_fidelity_hardware_compatible(config, params, x, y)
    
    println("Phase 1 Value: ", fidelity_phase1)
    println("Phase 2 Value: ", fidelity_phase2)
    
    # 4. Assert Equality
    #    Floating point arithmetic might differ slightly, so we use approx
    @test isapprox(fidelity_phase1, fidelity_phase2, atol=1e-9)
    
    # 5. Matrix Test
    #    Verify the full loop logic works
    X_batch = rand(3, n_features)
    K1 = compute_kernel_matrix_pure(config, params, X_batch)
    K2 = compute_kernel_matrix_hardware(config, params, X_batch)
    
    @test isapprox(K1, K2, atol=1e-9)
end
