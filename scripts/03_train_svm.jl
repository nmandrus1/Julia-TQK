# INPUT is a train/test split in the data directory.
# IDK how to get this yet, but we should ALWAYS take in a Train/Test split

# for RBF data, our knob is the gamma parameter for complexity
# Define logarithmic sweep for gamma
rbf_complexities = Dict(
    "very_low"  => 0.001,   # Very smooth decision boundary
    "low"       => 0.01,
    "medium"    => 0.1,
    "high"      => 1.0,
    "very_high" => 10.0,    # Very complex/overfit prone
    "extreme"   => 100.0
)

# quantum is wayyy more complex. We need to generate data
# for different numbers of qubits, entanglements, depths, etc. 

quantum_complexities = [
    # (name, n_qubits, n_layers, entanglement, reps)
    ("minimal",     2, 1, "none",        1),
    ("low",         2, 2, "linear",      1),
    ("medium_low",  4, 2, "linear",      1),
    ("medium",      4, 3, "alternating", 1),
    ("medium_high", 4, 4, "all_to_all",  2),
    ("high",        6, 4, "all_to_all",  2),
    ("very_high",   8, 5, "all_to_all",  3),
]
#I will start with just these, but a rigorous experiment
# would try all combinations. Is there a convienient mechanism
# to do that with Julia/DrWatson? 


# Master experiment configuration
experiment_suite = Dict(
    "data_generation" => Dict(
        "n_samples_train" => [100, 500, 1000, 5000],
        "n_samples_test" => [100, 500, 1000, 2000],  # Fixed ratio or absolute
        "n_features" => [2, 4, 8, 16],
        "noise_levels" => [0.0, 0.01, 0.05, 0.1],
        "class_balance" => [0.5, 0.3, 0.1],  # For imbalanced scenarios
    ),
    
    "kernel_types" => Dict(
        "classical" => ["rbf"],
        "quantum_julia" => ["reuploading", "iqp", "variational"],
        "quantum_qiskit" => ["pauli_feature_map"],
    ),
    
    "evaluation_metrics" => [
        # Training metrics
        "train_accuracy",
        "train_f1_score",
        "train_auroc",
        "kernel_target_alignment",
        
        # Test metrics  
        "test_accuracy",
        "test_f1_score",
        "test_auroc",
        "generalization_gap",
        
        # Kernel properties
        "kernel_condition_number",
        "kernel_effective_rank",
        "kernel_eigenvalue_decay",
        "kernel_frobenius_norm",
        
        # Optimization metrics
        "convergence_iterations",
        "convergence_time",
        "final_loss",
        "loss_trajectory",
        "gradient_norm_trajectory",
        
        # Computational metrics
        "kernel_computation_time",
        "memory_usage",
        "parameter_count",
    ]
)
