using MLUtils
using Random
using MultivariateStats

function data_from_config(config::DataConfig{RBFDataParams})
    return generate_pseudo_svm_dataset(config)
end

function data_from_config(config::DataConfig{QuantumPauliDataParams})
    return generate_pauli_expectation_data_grid(config)
end

"""
  Produces a Dictionary containing the Train/Test split of a given config  
"""
function produce_data(config::DataConfig)
    # NOTE: This is in COLUMN MAJOR format!!!!!
    data = data_from_config(config)
    X = data[:X]
    y = data[:y]

    # seed rng
    Random.seed!(config.seed)
    rng = Random.default_rng()
    
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
function prepare_data!(config::ExperimentConfig)   

    # NOTE: This is in COLUMN MAJOR format!!!!!
    data, filepath = produce_or_load(
        produce_data,
        config.data_config,  # Pass as argument
        datadir("sims", config.data_config.data_params.dataset_type);
        suffix="jld2",
    )
   
    # config.data_config.data_path = filepath
    return data, filepath
end
