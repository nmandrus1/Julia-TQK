using MLUtils
using MultivariateStats
using StableRNGs
using distances


# Helper (or just use generic one directly in the closure)
function rbf_kernel_matrix(X, Y, gamma)
    # Distances.pairwise computes column-wise distance matrix efficiently
    D2 = pairwise(SqEuclidean(), X, Y, dims=2)
    return exp.(-gamma .* D2)
end

function data_from_config(config::DataConfig{RBFDataParams}; rng::AbstractRNG)
    params = config.params
    
    # Define the closure: (X, Y) -> K
    # This captures 'gamma' from the config
    rbf_closure(X, Y) = rbf_kernel_matrix(X, Y, params.gamma)
    
    return generate_kernel_target_data(
        n_samples = config.n_samples,
        n_features = 2, # Fixed for this specific dataset type? Or derived?
        n_support_vectors = params.n_support_vectors,
        kernel_fn = rbf_closure,
        rng = rng,
        feature_range = params.feature_range,
        alpha_range = params.alpha_range
    )
end


function data_from_config(config::DataConfig{ReuploadingDataParams}; rng::AbstractRNG)
    params = config.params
    
    # 1. Build Teacher Circuit
    feature_map = ReuploadingConfig(
        params.n_qubits, 
        params.n_features, 
        params.n_layers, 
        params.entanglement
    )
    
    # 2. Assign Teacher Parameters (Randomly)
    teacher_thetas = rand(rng, n_trainable_params(feature_map))
    
    # 3. Define Closure
    # The teacher is FIXED. The kernel is just K(x,y) using FIXED theta.
    quantum_closure(X, Y) = compute_kernel_matrix_pure(feature_map, teacher_thetas, X, Y)
    
    data = generate_kernel_target_data(
        n_samples = config.n_samples,
        n_features = params.n_features,
        n_support_vectors = params.n_support_vectors,
        kernel_fn = quantum_closure,
        rng = rng,
        feature_range = (0.0, 2π), # Full rotation coverage
        alpha_range = params.alpha_range
    )
    
    # Save the teacher's secret params too!
    data["teacher_thetas"] = teacher_thetas
    return data
end

function data_from_config(config::DataConfig{QuantumPauliDataParams}; rng::AbstractRNG)
    params = config.params

    pconfig = PauliConfig(
                    n_features=params.n_features,
                    paulis=params.paulis,
                    reps=params.n_reps,
                    ent=params.ent,
            )
    
    # Closure relying on your Qiskit/Sim interface
    # (Assuming compute_pauli_kernel_matrix_cpu is defined as we discussed)
    pauli_closure(X, Y) = compute_kernel_matrix_pure(pconfig, X, Y)
    
    return generate_kernel_target_data(
        n_samples = config.n_samples,
        n_features = params.n_features,
        n_support_vectors = 20, # Or added to struct
        kernel_fn = pauli_closure,
        rng = rng,
        feature_range = (0.0, 2π)
    )
end

"""
  Produces a Dictionary containing the Train/Test split of a given config  
"""
function produce_data(config::DataConfig, rng::AbstractRNG)
    # NOTE: This is in COLUMN MAJOR format!!!!!
    data = data_from_config(config, rng=rng)
    X = data[:X]
    y = data[:y]
    
    # Split train/test
    (X_train_raw, y_train), (X_test_raw, y_test) = splitobs(rng, (X, y); at=(1 - config.test_size), shuffle=true)    
    
    #Scale features to (0, 2π) range based on training data statistics.
    x_min = minimum(X_train_raw, dims=2)
    x_max = maximum(X_train_raw, dims=2)

    X_train_scaled = (X_train_raw .- x_min) ./ (x_max .- x_min) .* 2π
    X_test_scaled = (X_test_raw .- x_min) ./ (x_max .- x_min) .* 2π

    # scaler = (min=x_min, max=x_max)

    #Apply PCA dimensionality reduction.
    pca_model = fit(PCA , X_train_scaled; maxoutdim=config.n_features)

    # Transform both sets
    X_train = MultivariateStats.predict(pca_model, X_train_scaled)
    X_test = MultivariateStats.predict(pca_model, X_test_scaled)

    return Dict("X_train" => X_train, "y_train" => y_train, "X_test" => X_test, "y_test" => y_test, "config" =>config) 
end

"""
Generates raw synthetic data, scales the data from (0, 2pi), and then uses PCA
to reduce the number of features to the number specified by the data_config.

If the exact data config has already been generated, the ready-to-use data
will be loaded to disk

Returns a dictionary with keys 
    :X_train, :y_train, :X_test, :y_test, and :config 

NOTE: This produces data in COLUMN MAJOR ORDER! To use with python routines
first take the transpose!
"""
function prepare_data!(config::ExperimentConfig, rng::AbstractRNG)   

    # NOTE: This is in COLUMN MAJOR format!!!!!
    data, filepath = produce_or_load(
        produce_data,
        config.data_config,  # Pass as argument
        rng,
        datadir("sims", config.data_config.data_params.dataset_type);
        suffix="jld2",
    )
   
    # config.data_config.data_path = filepath
    return data, filepath
end

