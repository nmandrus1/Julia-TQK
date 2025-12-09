using LinearAlgebra
using Random
using StableRNGs

"""
    SPSAState

Stores the hyperparameters for the SPSA optimizer.
"""
struct SPSAConfig
    max_iter::Int
    n_resamples::Int
    a::Float64  # Learning rate numerator
    c::Float64  # Perturbation numerator
    A::Float64  # Stability constant
    alpha::Float64 # Learning rate decay
    gamma::Float64 # Perturbation decay
end

# Standard defaults often used in Qiskit/PennyLane
function SPSAConfig(;
    max_iter::Int=100,
    n_resamples::Int = 1,
    a::Float64=0.628,
    c::Float64=0.2,
    A::Float64=0.1 * max_iter, # Note: A can use max_iter
    alpha::Float64=0.602,
    gamma::Float64=0.101
)
    # The inner constructor call must use positional arguments
    # matching the struct's definition order.
    return SPSAConfig(max_iter, n_resamples, a, c, A, alpha, gamma)
end

"""
    optimize_spsa(loss_function, initial_params, config::SPSAConfig, rng::AbstractRNG)

Minimizes 'loss_function(params)' using SPSA.
"""
function optimize_spsa(loss_fn::Function, init_params::Vector{Float64}, config::SPSAConfig, rng::AbstractRNG)
    params = copy(init_params)
    n_params = length(params)
    history = Float64[]
    
    for k in 1:config.max_iter
        # 1. Decay constants
        a_k = config.a / (k + config.A)^config.alpha
        c_k = config.c / k^config.gamma

        
        # Inside optimize_spsa loop...

        grad_accum = zeros(n_params)
        loss_accum = 0

        for _ in 1:config.n_resamples
            # 1. Generate NEW random perturbation
            delta = rand(rng, [-1.0, 1.0], n_params)
    
            # 2. Evaluate
            theta_plus = params .+ c_k .* delta
            theta_minus = params .- c_k .* delta
    
            loss_plus = loss_fn(theta_plus)
            loss_minus = loss_fn(theta_minus)
    
            # 3. Accumulate Gradient Estimate
            # g = (L+ - L-) / (2c * delta)
            # Note: Dividing by delta (which is +/- 1) is same as multiplying
            g_est = (loss_plus - loss_minus) ./ (2 * c_k * delta)
            grad_accum .+= g_est
            loss_accum += loss_plus
        end

        # Average the results
        grad_est = grad_accum ./ config.n_resamples        
        loss_est = loss_accum / config.n_resamples
        
        # 5. Update
        params .-= a_k .* grad_est
        
        push!(history, loss_est) # logging
        
        if k % 10 == 0
            println("Iter $k: Loss ~ $loss_est")
        end
    end
    
    return params, history
end
