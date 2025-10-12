# scripts/minimal_test.jl

# Activate the project environment
using Pkg
Pkg.activate(".")

using TQK
using Yao
using Zygote
using LinearAlgebra
using Printf

"""
This is the minimal test case to verify the gradient calculation of the TQK library.
It compares three methods:
1. Manual Analytical Derivation: The ground truth calculated on paper.
2. Finite Difference: The numerical gold standard.
3. TQK's Adjoint Method: The implementation being tested.
"""
function run_minimal_test()
    println("="^70)
    println("RUNNING MINIMAL GRADIENT VERIFICATION TEST")
    println("="^70)

    # --- 1. SETUP THE TOY PROBLEM ---
    x_data = [π/2, π/4]
    y_data = [1.0, -1.0]
    X_train = reshape(x_data, 2, 1)

    w_initial = 1.0
    b_initial = 0.0
    initial_params = [w_initial, b_initial] # [weights..., biases...]

    loss_fn(K) = (K[1, 2] - (y_data[1] * y_data[2]))^2

    # Use a 1-qubit feature map from TQK
    feature_map = ReuploadingCircuit(1, 1, 1, linear)
    kernel = FidelityKernel(feature_map)

    # --- 2. MANUAL GRADIENT CALCULATION (GROUND TRUTH) ---
    println("--- 1. Manual Calculation (Ground Truth) ---")
    θ₁ = w_initial * x_data[1] + b_initial
    θ₂ = w_initial * x_data[2] + b_initial
    ψ₁_manual = apply(zero_state(1), Ry(θ₁))
    ψ₂_manual = apply(zero_state(1), Ry(θ₂))
    c₁₂_manual = dot(state(ψ₁_manual), state(ψ₂_manual))
    K₁₂_manual = abs2(c₁₂_manual)
    loss_manual = (K₁₂_manual + 1)^2

    dL_dK12 = (K₁₂_manual + 1)
    dK12_dθ1 = -sin(θ₁ - θ₂)
    dθ1_dw = x_data[1]
    dK12_dθ2 = sin(θ₁ - θ₂)
    dθ2_dw = x_data[2]

    manual_grad_w = dL_dK12 * (dK12_dθ1 * dθ1_dw + dK12_dθ2 * dθ2_dw)
    @printf "Manual dL/dw = %.6f\n" manual_grad_w
    println("-"^70)


    # --- 3. FINITE DIFFERENCE CALCULATION (NUMERICAL TRUTH) ---
    println("--- 2. Finite Difference Calculation ---")
    ε = 1e-7
    # Loss at w + ε
    assign_params!(feature_map, [w_initial + ε], [b_initial])
    K_plus = TQK.evaluate(kernel, X_train)
    loss_plus = loss_fn(K_plus)
    # Loss at w - ε
    assign_params!(feature_map, [w_initial - ε], [b_initial])
    K_minus = TQK.evaluate(kernel, X_train)
    loss_minus = loss_fn(K_minus)

    fd_grad_w = (loss_plus - loss_minus) / (2ε)
    @printf "Finite Difference dL/dw = %.6f\n" fd_grad_w
    println("-"^70)


    # --- 4. TQK ADJOINT METHOD CALCULATION (THE TEST) ---
    println("--- 3. TQK Adjoint Method Calculation ---")
    # Reset parameters
    assign_params!(feature_map, [w_initial], [b_initial])
    trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_data)

    # Evaluate the kernel to cache it
    K_cache = TQK.evaluate(trainer.kernel, trainer.X, workspace=trainer.workspace)

    # Call the function we are testing
    loss_analytic, (grad_w_analytic, grad_b_analytic) = TQK.loss_gradient(
        trainer.kernel, K_cache, trainer.loss_fn, trainer.X, trainer.workspace
    )
    tqk_grad_w = grad_w_analytic[1]
    @printf "TQK Analytic dL/dw = %.6f\n" tqk_grad_w
    println("-"^70)


    # --- 5. VERDICT ---
    println("--- Verdict ---")
    rel_error = abs(fd_grad_w - tqk_grad_w) / (abs(fd_grad_w) + abs(tqk_grad_w) + 1e-9)

    if rel_error < 1e-2
        println("\n✅ SUCCESS: The TQK analytic gradient matches the ground truth.")
    else
        println("\n❌ FAILURE: The TQK analytic gradient is incorrect.")
        @printf "Relative Error: %.4f\n" rel_error
        println("The manual and FD methods agree, proving the bug is in the TQK implementation.")
    end
    println("="^70)
end

