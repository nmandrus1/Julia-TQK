
using TQK
using Random
using LinearAlgebra

"
    Generate data JUST like the RBF data,
    but using reuploading kernel instead of rbf kernel
"
function generate_reupload_data(data_config::DataConfig{ReuploadingDataParams})
    # extract DataConfig
    n_samples = data_config.n_samples
    n_support_vectors = data_config.data_params.n_support_vectors
    feature_range = data_config.data_params.feature_range 
    alpha_range = data_config.data_params.alpha_range 
    seed = data_config.seed

    n_qubits = data_config.data_params.n_qubits
    n_features = data_config.data_params.n_features
    n_layers = data_config.data_params.n_layers
    entanglement = data_config.data_params.entanglement

    feature_map = ReuploadingCircuit(n_qubits, n_features, n_layers, entanglement)
    assign_random_params!(feature_map, seed=seed)
    kernel = FidelityKernel(feature_map)

    Random.seed!(seed)
    
    # Helper to scale uniform random numbers to a given range
    rand_in_range(range_tuple, dims...) = range_tuple[1] .+ rand(dims...) .* (range_tuple[2] - range_tuple[1])

    # 1. Generate points first
    X = rand(2, n_samples)

    support_vectors = rand(2, n_support_vectors)
    labels_sv = rand([-1, 1], n_support_vectors)
    alphas = rand_in_range(alpha_range, n_support_vectors)

    X_grid = rand(2, 10000)
    K = evaluate(kernel, X_grid', support_vectors')
    # kernel_matrix = vcat([rbf_kernel(x, support_vectors, gamma) for x in eachcol(X_grid)]...)
    scores = K * (alphas .* labels_sv)
    b = -mean(scores)

    # 3. Compute kernel contributions for all points
    K = evaluate(kernel, X', support_vectors')

    scores = K * (alphas .* labels_sv)
    y_float = sign.(scores .+ b)
    
    # Handle boundary points (where f(x) = 0) by assigning a random label
    boundary_mask = y_float .== 0
    num_boundary = sum(boundary_mask)
    if num_boundary > 0
        y_float[boundary_mask] .= rand([-1, 1], num_boundary)
    end
    
    # Convert labels to integers
    y = Int.(y_float)

    # NOW scale to desired feature range if needed
    scale = feature_range[2] - feature_range[1]
    offset = feature_range[1]
    X_scaled = X .* scale .+ offset
    support_vectors_scaled = support_vectors .* scale .+ offset

    return Dict(
        :X=>X_scaled, 
        :y=>y, 
        :support_vectors=>support_vectors, 
        :alphas=>alphas, 
        :labels_sv=>labels_sv, 
        :bias=>b,
        :seed=>seed,
    )
end
