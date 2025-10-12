
using TQK
using LinearAlgebra
using Random
using Printf
using Yao

# ==============================================================================
# 1. SETUP: MINIMAL CONFIGURATION
# ==============================================================================

println("=== Minimal Gradient Debugging Script ===\n")

# --- Minimal Dataset (3 samples, 2 features) ---
X_train = [1.0 2.0; 3.0 4.0; 5.0 6.0]
y_train = [1.0, -1.0, 1.0]
println("Using a minimal dataset of 3 samples.")

# --- Minimal Quantum Circuit (2 qubits, 1 layer) ---
n_qubits = 2
n_layers = 1
n_features = 2
feature_map = ReuploadingCircuit(n_qubits, n_features, n_layers, linear)
n_params_circuit = n_params(feature_map)
println("Circuit: $n_qubits qubits, $n_layers layer => $n_params_circuit parameters per set (weights/biases).\n")

# --- Initialize Parameters ---
# Use fixed, non-zero parameters for reproducibility
Random.seed!(123)
initial_weights = randn(n_params_circuit) .* 0.1
initial_biases = randn(n_params_circuit) .* 0.1

println("Initial weights: ", initial_weights)
println("Initial biases: ", initial_biases)

assign_params!(feature_map, initial_weights, initial_biases)
initial_params = vcat(initial_weights, initial_biases)

# --- Kernel and Loss ---
kernel = FidelityKernel(feature_map)
loss_fn(K) = mean((K .- (y_train * y_train')).^2) # Simple Squared Error Loss

# --- Trainer ---
trainer = QuantumKernelTrainer(kernel, loss_fn, X_train, y_train)

# ==============================================================================
# 2. FORWARD PASS: SENSITIVITY CHECK
# ==============================================================================

println("\n" * "="^70)
println("STEP 1: VERIFYING FORWARD PASS SENSITIVITY")
println("="^70)

# --- Calculate base loss ---
assign_params!(trainer.kernel.feature_map, initial_weights, initial_biases)
K_base = TQK.evaluate(trainer.kernel, X_train)
loss_base = trainer.loss_fn(K_base)
println("Initial Loss: $loss_base")

# --- Perturb ONE parameter and recalculate ---
perturb_idx = 2
ε = 0.1 # Large perturbation to see an effect
params_perturbed = copy(initial_params)
params_perturbed[perturb_idx] += ε
weights_perturbed = @view params_perturbed[1:n_params_circuit]
biases_perturbed = @view params_perturbed[n_params_circuit+1:end]

println("\nPerturbing weights[$(perturb_idx)] by $ε...")
assign_params!(trainer.kernel.feature_map, weights_perturbed, biases_perturbed)

# Add a manual check right after assignment
w_check, _ = get_params(trainer.kernel.feature_map)
println("Parameter check: Is weights[$(perturb_idx)] updated in feature_map? -> ", w_check[perturb_idx] ≈ initial_weights[perturb_idx] + ε)

K_pert = TQK.evaluate(trainer.kernel, X_train)
loss_pert = trainer.loss_fn(K_pert)
println("Perturbed Loss: $loss_pert")

loss_change = abs(loss_pert - loss_base)
@info loss_change
println("\nLoss Change: ", loss_change)
if loss_change < 1e-8
    println("❌ FAILURE: The loss is insensitive to parameter changes. The forward pass is not working.")
else
    println("✅ SUCCESS: The loss is sensitive to parameter changes.")
end


# ==============================================================================
# 3. BACKWARD PASS: GRADIENT CALCULATION
# ==============================================================================
# Restore initial parameters for a clean gradient calculation
assign_params!(trainer.kernel.feature_map, initial_weights, initial_biases)
K = TQK.evaluate(trainer.kernel, X_train)

println("\n" * "="^70)
println("STEP 2: COMPUTING GRADIENTS (FINITE DIFFERENCE)")
println("="^70)

# --- Finite Difference for the first parameter ---
ε_fd = 1e-5
# Plus
params_plus = copy(initial_params); params_plus[1] += ε_fd
assign_params!(trainer.kernel.feature_map, @view(params_plus[1:n_params_circuit]), @view(params_plus[n_params_circuit+1:end]))
loss_plus = trainer.loss_fn(TQK.evaluate(trainer.kernel, X_train))
# Minus
params_minus = copy(initial_params); params_minus[1] -= ε_fd
assign_params!(trainer.kernel.feature_map, @view(params_minus[1:n_params_circuit]), @view(params_minus[n_params_circuit+1:end]))
loss_minus = trainer.loss_fn(TQK.evaluate(trainer.kernel, X_train))

fd_grad_w1 = (loss_plus - loss_minus) / (2 * ε_fd)
println("Finite Difference Gradient for weights[1]: ", fd_grad_w1)


println("\n" * "="^70)
println("STEP 3: COMPUTING GRADIENTS (ANALYTIC - ADJOINT METHOD)")
println("="^70)
# Use your custom adjoint method
loss_adj, (grad_w_adj, grad_b_adj) = TQK.loss_gradient(kernel, K, loss_fn, X_train, trainer.workspace)
println("Adjoint Method Gradient for weights[1]: ", grad_w_adj[1])


println("\n" * "="^70)
println("STEP 4: COMPUTING GRADIENTS (ANALYTIC - HYBRID ZYGOTE METHOD)")
println("="^70)
# Use the hybrid Zygote method
loss_hyb, (grad_w_hyb, grad_b_hyb) = TQK.hybrid_loss_gradient(K, X_train, kernel, loss_fn)
println("Hybrid Zygote Gradient for weights[1]: ", grad_w_hyb[1])
println("="^70)

println("\nComparison:")
println(@sprintf("  Finite Diff: %+e", fd_grad_w1))
println(@sprintf("  Adjoint:     %+e", grad_w_adj[1]))
println(@sprintf("  Hybrid:      %+e", grad_w_hyb[1]))
