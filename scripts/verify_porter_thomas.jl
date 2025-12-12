
using DrWatson
@quickactivate "TQK"

using TQK
using LinearAlgebra
using Plots
using Statistics
using Random
using Distributions
using StableRNGs
using Yao


"""
    run_porter_thomas_verification()

The "Nail in the Coffin" experiment.
Verifies if the Kernel Value distribution converges to the Porter-Thomas distribution
P(F) = D * exp(-D * F), which indicates the circuit forms a 2-Design.
"""
function run_porter_thomas_verification()
    println("=== PORTER-THOMAS VERIFICATION ===")
    
    # Configuration
    n_qubits = 6        # D = 64 (Large enough to see stats, small enough to simulate fast)
    n_features = 6      # Encode 6 features
    dim = 2^n_qubits
    
    # We will test increasing depths
    # depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    depths = [1, 2, 4, 8, 12, 16]
    n_samples = 2000    # Number of random pairs to test per depth
    
    # Prepare Plot
    p = Plots.plot(title="Convergence to Haar Randomness (D=$dim)", 
             xlabel="Kernel Value (F)", ylabel="Probability Density",
             xlims=(0, 5/dim), yscale=:log10)
    

    rng = StableRNG(42)

    # 2. Run Sweep
    for l in depths
        println("Testing Depth L=$l...")
        
        # A. Build Circuit
        config = ReuploadingConfig(n_qubits, n_features, l; entanglement=FullEntanglement)
        # Random parameters (untrained kernel)
        params = rand(rng, n_trainable_params(config))
        
        # B. Generate Random Inputs
        #    We sample uniform random inputs in [0, 2π]
        X = rand(rng, n_features, n_samples) .* 2π
        Y = rand(rng, n_features, n_samples) .* 2π
        
        # C. Compute Kernel Values (Fidelities)
        #    We compute element-wise dot products between column i of X and Y
        #    Efficient way: Batch compute all states, then dot product
        states_X = compute_statevectors(config, params, X)
        states_Y = compute_statevectors(config, params, Y)
        
        # Extract raw vectors and compute squared overlap
        # |<psi(x)|psi(y)>|^2
        fidelities = Float64[]
        for i in 1:n_samples
            # Inner product
            overlap = dot(state(states_X[i]), state(states_Y[i]))
            push!(fidelities, abs2(overlap))
        end
        
        # D. Analyze Statistics
        avg_F = mean(fidelities)
        kl_div = abs(avg_F - 1/dim) # Simple proxy for distance from theory
        println("   Avg F: $(round(avg_F, digits=6)) | Expected: $(round(1/dim, digits=6))")
        
        # E. Add to Plot (Histogram)
        stephist!(p, fidelities, normalize=:pdf, label="L=$l", lw=2, alpha=0.7)
    end

    
    #    Plot Theoretical Curve (Porter-Thomas)
    #    P(F) = D * exp(-D * F)
    x_theory = range(0, 10/dim, length=1000)
    y_theory = dim .* exp.(-dim .* x_theory)
    Plots.plot!(p, x_theory, y_theory, label="Theory (Haar)", lw=3, color=:black, linestyle=:dash)

    # Save
    display(p)
    fpath = plotsdir("porter_thomas_convergence.png")
    savefig(fpath)
    println("\n[✓] Plot saved to $fpath")
    println("If the histograms converge to the black dashed line, your kernel is a 2-Design.")
end

# Run immediately
run_porter_thomas_verification()
