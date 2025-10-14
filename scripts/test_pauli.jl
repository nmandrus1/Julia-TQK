using DrWatson
@quickactivate "TQK"

using TQK
using Random
using Printf
using LIBSVM
using Statistics
using Plots
using PythonCall

"""
Test Pauli kernel combinatorial search on quantum-native data.
"""
function test_pauli_search(;seed::Union{Int, Nothing}=nothing)
    println("="^70)
    println("PAULI KERNEL COMBINATORIAL SEARCH TEST")
    println("="^70)
    
    if isnothing(seed)
        seed = 42
    end
    Random.seed!(seed)
    
    # Generate quantum-separable data (small for fast testing)
    println("\n1. Generating quantum-separable test data...")
    data_config = DataConfig(
        n_samples=300,
        n_features=2,
        test_size=0.3,
        data_params=QuantumPauliDataParams(
            n_qubits=2,
            paulis=["Z", "ZZ"],  # True data-generating Paulis
            reps=2,
            entanglement="full",
            gap=0.3,
            grid_points_per_dim=80,
        ),
        seed=seed
    )
    
    data = produce_data(data_config)
    X_train = data[:X_train]
    y_train = data[:y_train]
    X_test = data[:X_test]
    y_test = data[:y_test]
    
    println("  Train samples: $(length(y_train))")
    println("  Test samples: $(length(y_test))")
    println("  Features: $(size(X_train, 1))")
    println("  True Paulis used: $(data_config.data_params.paulis)")
    
    # Create experiment configuration for search
    println("\n2. Setting up Pauli kernel search...")
    exp_config = ExperimentConfig(
        experiment_name="pauli_search_test",
        data_config=data_config,
        kernel_config=PauliKernelHyperparameterSearchConfig(
            n_qubits=2,
            reps=[1, 2, 3],
            entanglement=["full", "linear"],
            search_strategy="random",
            n_search_iterations=20,  # Small for fast testing
            search_constraints=PauliSearchConstraints(
                base_paulis=["X", "Y", "Z"],
                max_pauli_order=2,
                min_num_terms=1,
                max_num_terms=3
            ),
            seed=seed
        ),
        learning_curve_sizes=Int[],  # Not needed for this test
        c_ranges=[0.1, 1.0, 10.0],
        cv_folds=3,
        seed=seed
    )
    
    println("  Search iterations: $(exp_config.kernel_config.n_search_iterations)")
    println("  Reps to try: $(exp_config.kernel_config.reps)")
    println("  Entanglements: $(exp_config.kernel_config.entanglement)")
    
    # Run the search
    println("\n3. Running combinatorial search (this may take a minute)...")
    best_hyperparams, best_score = search_pauli_hyperparameters(
        exp_config, X_train, y_train
    )
    
    println("\n4. Search Results:")
    println("  Best KTA score: $(round(best_score, digits=4))")
    println("  Best Paulis: $(best_hyperparams.paulis)")
    println("  Best reps: $(best_hyperparams.reps)")
    println("  Best entanglement: $(best_hyperparams.entanglement)")
    println("  Best C: $(best_hyperparams.C)")
    
    # Evaluate on test set
    println("\n5. Evaluating on test set...")
    
    # Create kernel with best hyperparameters
    qiskit_lib = pyimport("qiskit.circuit.library")
    qiskit_kernels = pyimport("qiskit_machine_learning.kernels")
    
    feature_map = qiskit_lib.PauliFeatureMap(
        feature_dimension=size(X_train, 1),
        reps=best_hyperparams.reps,
        entanglement=best_hyperparams.entanglement,
        paulis=pylist(best_hyperparams.paulis)
    )
    
    kernel = qiskit_kernels.FidelityStatevectorKernel(feature_map=feature_map)
    
    # Compute kernel matrices (convert to row-major for Qiskit)
    K_train = pyconvert(Matrix{Float64}, kernel.evaluate(x_vec=X_train'))
    K_test = pyconvert(Matrix{Float64}, kernel.evaluate(x_vec=X_test', y_vec=X_train'))
    
    # Train SVM
    model = svmtrain(K_train, vec(y_train), kernel=Kernel.Precomputed, cost=best_hyperparams.C, verbose=false)
    
    # Predictions
    train_pred = svmpredict(model, K_train)[1]
    test_pred = svmpredict(model, K_test')[1]
    
    train_acc = mean(train_pred .== y_train)
    test_acc = mean(test_pred .== y_test)
    
    println("  Train accuracy: $(round(train_acc * 100, digits=1))%")
    println("  Test accuracy: $(round(test_acc * 100, digits=1))%")
    
    # Create visualization
    println("\n6. Creating visualization...")
    
    # Plot data
    p_data = scatter(X_train[1, y_train.==1], X_train[2, y_train.==1],
                     color=:red, marker=:circle, label="Train (+1)", alpha=0.6)
    scatter!(p_data, X_train[1, y_train.==-1], X_train[2, y_train.==-1],
             color=:blue, marker=:circle, label="Train (-1)", alpha=0.6)
    scatter!(p_data, X_test[1, y_test.==1], X_test[2, y_test.==1],
             color=:red, marker=:diamond, label="Test (+1)", markersize=5)
    scatter!(p_data, X_test[1, y_test.==-1], X_test[2, y_test.==-1],
             color=:blue, marker=:diamond, label="Test (-1)", markersize=5)
    plot!(p_data, xlabel="Feature 1", ylabel="Feature 2",
          title="Quantum-Separable Test Data", legend=:best)
    
    println("\n" * "="^70)
    println("TEST COMPLETE")
    println("="^70)
    println("\nSummary:")
    println("  ✓ Successfully searched $(exp_config.kernel_config.n_search_iterations) configurations")
    println("  ✓ Found best configuration with KTA = $(round(best_score, digits=4))")
    println("  ✓ Test accuracy: $(round(test_acc * 100, digits=1))%")
    
    if test_acc > 0.7
        println("\nPASS: Test accuracy > 70%")
    else
        println("\nWARNING: Test accuracy < 70% (may need more search iterations)")
    end
    
    return Dict(
        :best_hyperparams => best_hyperparams,
        :best_score => best_score,
        :train_accuracy => train_acc,
        :test_accuracy => test_acc,
        :plot => p_data,
        :data => data
    )
end

# Run the test if executed as main script
if abspath(PROGRAM_FILE) == @__FILE__
    results = test_pauli_search()
end
