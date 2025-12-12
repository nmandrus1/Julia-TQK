using DrWatson
@quickactivate "TQK"

using TQK
using LinearAlgebra
using Plots
using Random
using StableRNGs
using Yao

# ============================================================================
# SHARED LOGIC
# ============================================================================

"""
    compute_kernel_ray_profile(config, params, x_origin, ray_points)

Computes the kernel fidelity profile along a ray of points relative to an 
origin point for a specific circuit configuration.

Returns:
    Vector{Float64}: The list of squared overlaps |<ψ(origin)|ψ(ray_i)>|²
"""
function compute_kernel_ray_profile(config, params, x_origin, ray_points)
    # 1. Compute Reference State |ψ(x_origin)>
    #    Reshape x_origin to (n_features, 1) for batch processing
    ref_reg = compute_statevectors(config, params, reshape(x_origin, :, 1))[1]
    ref_state = state(ref_reg)
    
    # 2. Compute States along the Ray
    #    ray_points is (n_features x n_steps)
    ray_regs = compute_statevectors(config, params, ray_points)
    
    # 3. Calculate Kernel Profile (Fidelities)
    kernel_profile = Float64[]
    sizehint!(kernel_profile, length(ray_regs))
    
    for r in ray_regs
        # Inner product with reference
        overlap = dot(ref_state, state(r))
        push!(kernel_profile, abs2(overlap))
    end
    
    return kernel_profile
end

"""
    run_periodicity_sweep(configs, labels, baselines; kwargs...)

Generic driver to run a "Ray Tracing" sweep over a list of configurations.
It sets up the random geometry once, then iterates through the provided configs
to trace the kernel landscape.

Arguments:
    configs: Vector of ReuploadingConfig
    labels: Vector of Strings for the legend
    baselines: Vector of Floats for the 1/D theoretical line
"""
function run_periodicity_sweep(configs::Vector, labels::Vector{String}, baselines::Vector{Float64};
                               title::String, filename::String,
                               n_features::Int=4, n_steps::Int=4000, distance_range::Float64=8π, seed::Int=41)
    
    println("=== RUNNING SWEEP: $title ===")
    rng = StableRNG(seed)

    # 1. Geometry Setup (Fixed for all configs in this sweep)
    # Pick a random origin in the feature space
    x_origin = rand(rng, n_features) .* 2π
    
    # Pick a random normalized direction vector
    d_raw = randn(rng, n_features)
    d = d_raw ./ norm(d_raw)
    
    # Generate the path (The "Ray")
    ks = range(0, distance_range, length=n_steps)
    ray_points = hcat([x_origin .+ k .* d for k in ks]...)

    # Prepare Plot
    default_palette = Plots.palette(:default)
    p = Plots.plot(title=title,
             xlabel="Distance k (along direction d)",
             ylabel="Kernel Value K(x, x+kd)",
             ylims=(0, 1.05),
             legend=:topright,
             size=(900, 600))

    # 2. Run Sweep
    for (i, (config, label, baseline)) in enumerate(zip(configs, labels, baselines))
        println("  -> Processing: $label")
        
        # Generate random params for this specific config
        # Note: RNG state advances deterministically here
        params = rand(rng, n_trainable_params(config))
        
        # Compute Profile
        profile = compute_kernel_ray_profile(config, params, x_origin, ray_points)
        
        # Plotting
        # Cycle colors so trace and baseline match
        series_color = default_palette[mod1(i, length(default_palette))]
        
        # Plot Trace
        Plots.plot!(p, ks, profile, label=label, lw=1.5, alpha=0.8, color=series_color)
        
        # Plot Baseline (Theoretical Random Average 1/D)
        # Using label="" hides it from legend to keep it clean, 
        # or we could label it if baselines differ significantly.
        Plots.plot!(p, [0, distance_range], [baseline, baseline], 
                    label="", linestyle=:dash, color=series_color)
    end

    # Save
    display(p)
    fpath = plotsdir(filename)
    savefig(fpath)
    println("\n[✓] Plot saved to $fpath")
end

# ============================================================================
# SPECIFIC EXPERIMENTS
# ============================================================================

function run_periodicity_experiment_random_ray_var_layers()
    n_qubits = 4
    depths = [1, 2, 4, 8, 16]
    
    # Construct Inputs
    configs = [ReuploadingConfig(n_qubits, 4, l; entanglement=FullEntanglement) for l in depths]
    labels = ["L=$l" for l in depths]
    # Baseline is constant (1/16) for all lines since qubits are fixed
    baselines = [1.0/2^n_qubits for _ in depths]
    
    run_periodicity_sweep(configs, labels, baselines;
        title="Kernel Landscape (Varying Layers, N=$n_qubits)",
        filename="kernel_periodicity_ray_layers.png"
    )
end

function run_periodicity_experiment_random_ray_var_qubits()
    n_layers = 2
    nqs = [1, 2, 4, 8]
    
    # Construct Inputs
    configs = [ReuploadingConfig(nq, 4, n_layers; entanglement=FullEntanglement) for nq in nqs]
    labels = ["N=$nq" for nq in nqs]
    # Baseline varies per line (1/2, 1/4, 1/16...)
    baselines = [1.0/2^nq for nq in nqs] 
    
    run_periodicity_sweep(configs, labels, baselines;
        title="Kernel Landscape (Varying Qubits, L=$n_layers)",
        filename="kernel_periodicity_ray_qubits.png"
    )
end

# Run immediately
# run_periodicity_experiment_random_ray_var_layers()
# run_periodicity_experiment_random_ray_var_qubits()
