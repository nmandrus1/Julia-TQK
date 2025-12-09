
using DrWatson
@quickactivate "TQK"

using TQK
using LinearAlgebra
using Dates
using Printf

include("experiment_runner.jl")

"""
    run_full_grid_verification()

Executes a 3x3 Experiment Grid:
Datasets: [RBF, Reuploading, Pauli]
Models:   [RBF, Reuploading, Pauli]

Total: 9 Experiments.
"""
function run_full_grid_verification()
    println("==========================================================")
    println("   STARTING FULL 3x3 GRID VERIFICATION (Teacher-Student)  ")
    println("==========================================================\n")

    # Common Settings for Speed
    N_SAMPLES = 100        # Small enough to be fast, large enough to learn
    N_FEATURES = 2         # Visualization-friendly 2D
    N_QUBITS = 2           # Fast simulation
    MASTER_SEED::UInt = 1234     # Reproducibility Root

    # ----------------------------------------------------------------
    # 1. Define The Teachers (Datasets)
    # ----------------------------------------------------------------
    datasets = []

    # Teacher A: RBF SVM
    push!(datasets, DataConfig(
        dataset_name = "grid_test_rbf",
        n_samples = N_SAMPLES,
        n_features = N_FEATURES,
        params = RBFDataParams(gamma=2.0, n_support_vectors=10),
        master_seed = MASTER_SEED # Fixed Data Seed
    ))

    # Teacher B: Reuploading Quantum Circuit
    push!(datasets, DataConfig(
        dataset_name = "grid_test_reup",
        n_samples = N_SAMPLES,
        n_features = N_FEATURES,
        params = ReuploadingDataParams(
            n_qubits=N_QUBITS, n_features=N_FEATURES, n_layers=2, 
            ent=LinearEntanglement, n_support_vectors=10
        ),
        master_seed = MASTER_SEED # Fixed Data Seed
    ))

    # Teacher C: Pauli Feature Map
    push!(datasets, DataConfig(
        dataset_name = "grid_test_pauli",
        n_samples = N_SAMPLES,
        n_features = N_FEATURES,
        params = QuantumPauliDataParams(
            n_features=N_FEATURES, paulis=["Z", "ZZ"], reps=2, ent=LinearEntanglement
        ),
        master_seed = MASTER_SEED # Fixed Data Seed
    ))

    # ----------------------------------------------------------------
    # 2. Define The Students (Kernel Methods)
    # ----------------------------------------------------------------
    methods = []

    # Student 1: RBF Kernel (The Classical Baseline)
    push!(methods, RBFMethod(
        name = "student_rbf",
        gamma_grid = [0.1, 1.0, 10.0]
    ))

    # Student 2: Reuploading Kernel (The Gradient Learner)
    push!(methods, ReuploadingMethod(
        name = "student_reup",
        circuit_config = ReuploadingConfig(N_QUBITS, N_FEATURES, 2),
        optimizer = SPSAConfig( max_iter=50, n_resamples=2)
    ))

    # Student 3: Pauli Kernel (The Structure Learner)
    push!(methods, PauliMethod(
        name = "student_pauli",
        n_features = N_FEATURES,
        search_iterations = 5 # Small random search
    ))

    # ----------------------------------------------------------------
    # 3. Execute The Grid
    # ----------------------------------------------------------------
    results_matrix = Matrix{Any}(missing, 3, 3)
    
    for (i, d_conf) in enumerate(datasets)
        for (j, m_conf) in enumerate(methods)
            
            # Construct Unique Experiment
            exp_name = "grid_$(d_conf.dataset_name)_vs_$(m_conf.name)"
            println("\n>> Running: $exp_name")
            
            exp_config = ExperimentConfig(
                name = exp_name,
                master_seed = MASTER_SEED,
                data_config = d_conf,
                method = m_conf,
                tuning_batch_size = 50,
                cv_folds = 3,
                c_grid = [0.1, 1.0, 10.0]
            )

            # Run (Force=true ensures we actually test the code execution)
            try
                artifact = run_experiment(exp_config; force=true);
                results_matrix[i, j] = artifact;
                println("   [SUCCESS] Test Acc: $(round(artifact.test_acc, digits=4)) | Best Param Score: $(round(artifact.tuning_result.best_score, digits=4))")
            catch e
                println("   [FAILED] Error: $e")
                # rethrow(e) # Uncomment to debug specific crashes
            end
        end
    end

    # ----------------------------------------------------------------
    # 4. Final Report
    # ----------------------------------------------------------------
    println("\n\n==========================================================")
    println("                  GRID VERIFICATION REPORT                ")
    println("==========================================================")
    
    # Header
    @printf("%-20s | %-15s %-15s %-15s\n", "Dataset \\ Model", "RBF", "Reuploading", "Pauli")
    println(repeat("-", 75))

    row_names = ["RBF Data", "Reup Data", "Pauli Data"]
    
    for i in 1:3
        # Extract Test Accuracies
        accs = []
        for j in 1:3
            if !ismissing(results_matrix[i, j])
                val = results_matrix[i, j].test_acc
                push!(accs, @sprintf("%.3f", val))
            else
                push!(accs, "FAIL")
            end
        end
        
        @printf("%-20s | %-15s %-15s %-15s\n", row_names[i], accs[1], accs[2], accs[3])
    end
    println("==========================================================")
end

# Run immediately
run_full_grid_verification()
