
using Test
using Yao
using LinearAlgebra
using Random

# Include new files (User must update TQK.jl or include manually)
include("../src/feature_maps/types.jl")
include("../src/feature_maps/reupload.jl")
include("../src/utils/mock_hardware.jl")
include("../src/optimizers/spsa.jl")

@testset "Hardware Training (SPSA)" begin
    # 1. Setup Simple Problem
    #    We want to classify points in 2D.
    n_qubits = 2
    n_features = 2
    n_layers = 1
    
    # Easy dataset: 3 points. 
    # If SPSA can't overfit 3 points, it's broken.
    X = [0.0 0.0; 
         0.5 0.5; 
         0.9 0.9]
    y = [-1.0, 1.0, -1.0] 
    
    config = ReuploadingConfig(n_qubits, n_features, n_layers)
    params = randn(config.total_params)
    
    # 2. Define the Noisy Loss Function
    #    100 shots = highly noisy!
    noisy_loss(p) = hardware_compatible_loss(config, p, X, y, 100)
    
    println("Initial Loss (Noisy): ", noisy_loss(params))
    
    # 3. Run SPSA
    spsa_conf = SPSAConfig(max_iter=50, a=2.0, c=0.1)
    
    final_params, history = optimize_spsa(noisy_loss, params, spsa_conf)
    
    final_loss = noisy_loss(final_params)
    println("Final Loss (Noisy): ", final_loss)
    
    # 4. Verify Improvement
    #    KTA is negative alignment. Ideally -1.0.
    #    We check if we improved significantly from random.
    @test final_loss < history[1] 
    
    # 5. Sanity Check on "Silent" failure
    #    Ensure parameters actually moved
    @test final_params != params
end
