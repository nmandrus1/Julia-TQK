
using LinearAlgebra
using Statistics
using Random

"""
    generate_kernel_target_data(; 
        n_samples::Int, 
        n_features::Int, 
        n_support_vectors::Int, 
        kernel_fn::Function, # (X_matrix, Y_matrix) -> Kernel_matrix
        rng::AbstractRNG,
        feature_range::Tuple{Float64, Float64}=(0.0, 2π),
        alpha_range::Tuple{Float64, Float64}=(0.1, 2.0)
    )

Universal data generator. Creates a random SVM target function using the provided `kernel_fn`
and labels random inputs.

Process:
1. Generate random Support Vectors (SVs) and Weights.
2. Scale SVs to `feature_range`.
3. Calibrate Bias `b` to center the decision boundary (using a grid search).
4. Generate random Training Inputs `X`, scale them.
5. Label `X` using `f(x) = Σ w_i K(x, sv_i) + b`.
"""
function generate_kernel_target_data(;
    n_samples::Int,
    n_features::Int,
    n_support_vectors::Int,
    kernel_fn::Function,
    rng::AbstractRNG,
    feature_range::Tuple{Float64, Float64}=(0.0, 2π),
    alpha_range::Tuple{Float64, Float64}=(0.1, 2.0)
)
    # --- Helpers ---
    # Scales [0, 1] data to [min, max]
    function scale_data(raw_data)
        min_v, max_v = feature_range
        return raw_data .* (max_v - min_v) .+ min_v
    end

    # --- 1. Teacher Setup ---
    # Generate raw [0,1] support vectors
    sv_raw = rand(rng, n_features, n_support_vectors) 
    sv_phys = scale_data(sv_raw) # Scale BEFORE kernel compute

    # Generate Weights (Alpha * Label)
    alphas = rand(rng, alpha_range[1]:0.01:alpha_range[2], n_support_vectors)
    labels_sv = rand(rng, [-1, 1], n_support_vectors)
    weights = alphas .* labels_sv

    # --- 2. Bias Calibration ---
    # Sample a grid to estimate the distribution of the decision values
    # We use a larger set (e.g., 2000 points) to get a stable mean
    grid_raw = rand(rng, n_features, 2000)
    grid_phys = scale_data(grid_raw)
    
    # Compute Teacher scores on grid
    # K_grid is (2000 x n_sv)
    K_grid = kernel_fn(grid_phys, sv_phys) 
    scores_grid = K_grid * weights
    
    b = -mean(scores_grid) # Center the boundary

    # --- 3. Student Data Generation ---
    X_raw = rand(rng, n_features, n_samples)
    X_phys = scale_data(X_raw)
    
    K_data = kernel_fn(X_phys, sv_phys)
    scores = K_data * weights .+ b
    
    y = sign.(scores)
    
    # Fix exact zeros (rare, but possible)
    y[y .== 0] .= rand(rng, [-1.0, 1.0], count(y .== 0))

    return Dict(
        "X" => X_phys,
        "y" => y,
        "teacher_sv" => sv_phys,
        "teacher_weights" => weights,
        "teacher_bias" => b
    )
end
