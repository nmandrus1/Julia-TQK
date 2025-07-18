using DrWatson
@quickactivate "TrainableQuantumKernel" # <- project name

# optimization lib
using Optim

# trainable circuit builder
include(scriptsdir("reupload.jl"))



struct OptimizationConfiguration
    dir_type::DirectionVectorType = random
    alpha::Float = 0.1
    iterations::Integer = 100
    loss_fn::Function
    circuit_opts::CircuitBuilderOpts
end

@enum DirectionVectorType random gradient

# Visualize the loss landscape in one direction
function one_d_visualization(config::OptimizationConfiguration)
    # compute initial random parameters
    # loop:
    #     compute gradient
    #     update parameters
    #     record loss

    # build circuit
    circuit_opts = config.circuit_opts
    circuit, params = build_reuploading_circuit(circuit_opts)

    w_rand, θ_rand = random_parameters!(params, seed=11)  

    set_weights!(params, w_rand)
    set_biases!(params, θ_rand)
    
end

 f
