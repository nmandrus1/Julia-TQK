
# src/optimizers/spsa.jl
using LinearAlgebra
using Random
using StableRNGs

"""
    SPSAState

Stores the hyperparameters for the SPSA optimizer.
"""
struct SPSAConfig
    rng::AbstractRNG
    max_iter::Int
    a::Float64  # Learning rate numerator
    c::Float64  # Perturbation numerator
    A::Float64  # Stability constant
    alpha::Float64 # Learning rate decay
    gamma::Float64 # Perturbation decay
end

# Standard defaults often used in Qiskit/PennyLane
function SPSAConfig(;
    seed::Integer,
    max_iter::Int=100,
    a::Float64=0.628,
    c::Float64=0.2,
    A::Float64=0.1 * max_iter, # Note: A can use max_iter
    alpha::Float64=0.602,
    gamma::Float64=0.101
)
    # The inner constructor call must use positional arguments
    # matching the struct's definition order.
    return SPSAConfig(StableRNG(seed), max_iter, a, c, A, alpha, gamma)
end

"""
    optimize_spsa(loss_function, initial_params, config::SPSAConfig)

Minimizes 'loss_function(params)' using SPSA.
"""
function optimize_spsa(loss_fn::Function, init_params::Vector{Float64}, config::SPSAConfig)
    params = copy(init_params)
    n_params = length(params)
    history = Float64[]
    
    for k in 1:config.max_iter
        # 1. Decay constants
        a_k = config.a / (k + config.A)^config.alpha
        c_k = config.c / k^config.gamma
        
        # 2. Generate Perturbation Vector (Bernoulli +/- 1)
        delta = rand(config.rng, [-1.0, 1.0], n_params)
        
        # 3. Evaluate two points
        theta_plus = params .+ c_k .* delta
        theta_minus = params .- c_k .* delta
        
        loss_plus = loss_fn(theta_plus)
        loss_minus = loss_fn(theta_minus)
        
        # 4. Estimate Gradient
        #    g_est = (L+ - L-) / (2*ck*delta)
        #    Since delta is +/- 1, dividing by delta is same as multiplying
        grad_est = (loss_plus - loss_minus) ./ (2 * c_k * delta)
        
        # 5. Update
        params .-= a_k .* grad_est
        
        push!(history, loss_plus) # logging
        
        if k % 10 == 0
            println("Iter $k: Loss ~ $loss_plus")
        end
    end
    
    return params, history
end
