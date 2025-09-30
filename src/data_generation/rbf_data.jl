
using Random
using LinearAlgebra

"""
    rbf_kernel(x::AbstractVector, Y::AbstractMatrix, gamma::Float64)

Compute the RBF kernel between a single point `x` and a set of points `Y`.
Each row of `Y` is treated as a data point.

# Arguments
- `x`: A single data point vector.
- `Y`: A matrix where each row is a data point.
- `gamma`: The RBF kernel parameter.

# Returns
A vector of kernel values `K(x, y_i)` for each row `y_i` in `Y`.
"""
function rbf_kernel(x::AbstractVector, Y::AbstractMatrix, gamma::Float64)
    # Calculate squared Euclidean distances between x and each row of Y
    diffs = Y .- x'
    sq_distances = sum(abs2, diffs, dims=2)
    
    return exp.(-gamma .* sq_distances)
end


"""
    generate_pseudo_svm_dataset(; n_samples=10000, n_support_vectors=20, 
                                feature_range=(0, 2π), gamma=1.0, 
                                alpha_range=(0.1, 2.0), bias_range=(-1, 1), 
                                random_state=42)

Generate a dataset using a pseudo-SVM model with a known RBF decision boundary.

This function creates a synthetic dataset by first defining the parameters of an SVM 
decision function (support vectors, alphas, bias) and then classifying randomly 
generated points according to that function.

# Keyword Arguments
- `n_samples`: Number of data points to generate.
- `n_support_vectors`: Number of pseudo support vectors.
- `feature_range`: Tuple `(min, max)` for the feature values.
- `gamma`: The RBF kernel parameter.
- `alpha_range`: Tuple `(min, max)` for the pseudo Lagrange multipliers.
- `bias_range`: Tuple `(min, max)` for the bias term.
- `random_state`: Seed for the random number generator.

# Returns
A `NamedTuple` containing:
- `X`: Feature matrix of size `(n_samples, 2)`.
- `y`: Label vector `{-1, 1}` of length `n_samples`.
- `support_vectors`: The pseudo support vectors of size `(n_support_vectors, 2)`.
- `alphas`: The pseudo Lagrange multipliers.
- `labels_sv`: The labels for the support vectors.
- `bias`: The bias term.
"""
function generate_pseudo_svm_dataset(;
    n_samples::Int=10000, 
    n_support_vectors::Int=20,
    feature_range::Tuple{Float64, Float64}=(0.0, 2π), 
    gamma::Float64=1.0,
    alpha_range::Tuple{Float64, Float64}=(0.1, 2.0), 
    bias_range::Tuple{Float64, Float64}=(-1.0, 1.0),
    seed::Int=42
)
    Random.seed!(seed)
    
    # Helper to scale uniform random numbers to a given range
    rand_in_range(range_tuple, dims...) = range_tuple[1] .+ rand(dims...) .* (range_tuple[2] - range_tuple[1])

    # Step 1: Generate pseudo support vectors
    support_vectors = rand_in_range(feature_range, n_support_vectors, 2)
    
    # Step 2: Assign random labels to support vectors
    labels_sv = rand([-1, 1], n_support_vectors)
    
    # Step 3: Generate pseudo Lagrange multipliers
    alphas = rand_in_range(alpha_range, n_support_vectors)
    
    # Step 4: Generate bias
    bias = rand_in_range(bias_range)
    
    # Step 5: Generate dataset points
    X = rand_in_range(feature_range, n_samples, 2)
    
    # Step 6: Classify using the decision function
    function decision_function(x_point::AbstractVector)
        # f(x) = Σᵢ αᵢ yᵢ K(x, sᵢ) + b
        kernel_values = rbf_kernel(x_point, support_vectors, gamma)
        return sum(alphas .* labels_sv .* kernel_values) + bias
    end
    
    # Classify all points using a comprehension over the rows of X
    y_float = [sign(decision_function(x_row)) for x_row in eachrow(X)]
    
    # Handle boundary points (where f(x) = 0) by assigning a random label
    boundary_mask = y_float .== 0
    num_boundary = sum(boundary_mask)
    if num_boundary > 0
        y_float[boundary_mask] .= rand([-1, 1], num_boundary)
    end
    
    # Convert labels to integers
    y = Int.(y_float)
    
    return Dict(
        :X=>X, 
        :y=>y, 
        :support_vectors=>support_vectors, 
        :alphas=>alphas, 
        :labels_sv=>labels_sv, 
        :bias=>bias,
        :seed=>seed,
    )
end
