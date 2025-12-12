
using Plots

"""
    run_rbf_landscape_baseline()

Visualizes the geometric profile of the RBF Kernel K(x, y) = exp(-γ||x-y||²).
Unlike Quantum Kernels, this is isotropic (depends only on distance), 
so we don't need random vectors—just the distance parameter k.
"""
function run_rbf_landscape_baseline()
    println("=== RBF BASELINE LANDSCAPE ===")
    
    # 1. Configuration (Matching your Quantum Experiment)
    distance_range = 6π  # Same range as quantum plots
    n_steps = 1000
    ks = range(0, distance_range, length=n_steps)
    
    # We test various Gamma values (equivalent to "Training" the RBF)
    # Low Gamma = Broad features (like Low Depth)
    # High Gamma = Sharp features (like High Depth?)
    gammas = [0.01, 0.1, 1.0, 5.0]

    # 2. Plotting
    p = Plots.plot(title="RBF Kernel Landscape (Gold Standard)",
             xlabel="Distance k (along any ray)",
             ylabel="Kernel Value",
             ylims=(0, 1.05),
             xlims=(0, distance_range),
             legend=:topright,
             size=(800, 500))

    for g in gammas
        # RBF Formula: exp(-γ * distance^2)
        profile = exp.(-g .* ks.^2)
        
        Plots.plot!(p, ks, profile, label="Gamma = $g", lw=2.5, alpha=0.8)
    end

    # 3. Save
    # We don't need a "Random Baseline" (1/D) because RBF isn't normalized 
    # to a hypersphere dimension in the same way. It naturally decays to 0.
    mkpath(plotsdir())
    fpath = plotsdir("untrained_vs_trained_periodicity/rbf_landscape_baseline.png")
    savefig(fpath)
    println("-> Plot saved to $fpath")
    println("Notice: No oscillations. Monotonic decay. This is 'Locality'.")
end

run_rbf_landscape_baseline()
