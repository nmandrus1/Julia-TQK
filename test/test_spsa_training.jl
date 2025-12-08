using TQK
using Test
using LinearAlgebra
using Statistics
using Random
using Zygote 

# Helper to generate SPSA estimate manually for testing
function spsa_gradient_estimate(loss_fn, params, c)
    n_params = length(params)
    delta = rand([-1.0, 1.0], n_params)
    theta_plus = params .+ c .* delta
    theta_minus = params .- c .* delta
    loss_plus = loss_fn(theta_plus)
    loss_minus = loss_fn(theta_minus)
    return (loss_plus - loss_minus) ./ (2 * c * delta)
end

@testset "SPSA Robust Verification" begin

    # ==========================================
    # Test 1: Deterministic Convergence
    # Why: If it fails here, the update logic (decay, step size) is broken.
    # ==========================================
    @testset "Deterministic Quadratic Bowl" begin
        # Target: [1.0, 1.0]
        quad_loss(x) = sum((x .- 1.0).^2)
        
        config = SPSAConfig(seed=42, max_iter=300, a=1.0, c=0.1, A=10.0)
        init_params = [0.0, 0.0]
        
        final_params, history = optimize_spsa(quad_loss, init_params, config)
        
        # Should be very close to exact solution
        # initial loss is 2, so getting to a loss of 0.01 from
        # 2.0 is better than a 99% improvement
        @test isapprox(final_params, [1.0, 1.0], atol=0.05)
        @test history[end] < 1e-2
    end

    # ==========================================
    # Test 2: Gradient Bias Check (The Math Check)
    # Why: SPSA gradient must correlate with True Gradient (Zygote)
    # ==========================================
    @testset "Gradient Estimator Bias" begin
        # Use your actual Reuploading Kernel config
        n_qubits = 2
        n_features = 2
        n_layers = 1
        X = rand(2, 2)
        y = [1.0, -1.0]
        
        config = ReuploadingConfig(n_qubits, n_features, n_layers)
        params = randn(config.total_params)
        
        # 1. Compute TRUE Gradient via Zygote (Noiseless)
        #    We use variational_kta_loss (the pure function) as ground truth
        true_loss(p) = variational_kta_loss(config, p, X, y)
        true_grad = gradient(true_loss, params)[1]
        
        # 2. Compute Average SPSA Gradient (Noisy Estimator)
        #    We assume the hardware has some shot noise, but for this bias check
        #    we can use the exact function to see if the SPSA math holds.
        n_samples = 1000
        avg_spsa_grad = zeros(length(params))
        c_perturb = 0.01
        
        for _ in 1:n_samples
            g_est = spsa_gradient_estimate(true_loss, params, c_perturb)
            avg_spsa_grad .+= g_est
        end
        avg_spsa_grad ./= n_samples
        
        # 3. Correlation Check
        #    Cosine similarity should be high (close to 1.0)
        similarity = dot(avg_spsa_grad, true_grad) / (norm(avg_spsa_grad) * norm(true_grad))
        
        println("SPSA vs Zygote Cosine Similarity: $similarity")
        @test similarity > 0.8 # It should point in roughly the same direction
    end

    # ==========================================
    # Test 3: Noisy Optimization Trend
    # Why: Proves we can descend the TRUE landscape using only NOISY measurements
    # ==========================================
    @testset "Noisy Optimization Descent" begin
        n_qubits = 2
        X = [0.0 0.0; 0.5 0.5]
        y = [-1.0, 1.0]
        config = ReuploadingConfig(n_qubits, 2, 1)
        params = randn(config.total_params)
        
        # The optimizer sees NOISY loss
        noisy_loss(p) = hardware_compatible_loss(config, p, X, y, 100)
        
        # We verify progress against TRUE loss
        true_loss(p) = variational_kta_loss(config, p, X, y)
        
        start_true_loss = true_loss(params)
        
        # Run SPSA
        spsa_conf = SPSAConfig(seed=42, max_iter=50, a=1.0, c=0.1)
        final_params, history = optimize_spsa(noisy_loss, params, spsa_conf)
        
        end_true_loss = true_loss(final_params)
        
        println("Start True Loss: $start_true_loss")
        println("End True Loss:   $end_true_loss")
        
        # Check descent on the ground truth, not the noisy history
        @test end_true_loss < start_true_loss
    end
end
