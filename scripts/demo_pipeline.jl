
# scripts/demo_pipeline.jl
using TQK
using LinearAlgebra
using Random
using StableRNGs
using Dates

function run_demo()
    println("=== Starting TQK Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup: Define a Teacher (Data) and a Student (Method)
    # ---------------------------------------------------------
    
    # Teacher: A simple RBF dataset
    data_conf = DataConfig(
        dataset_name="demo_rbf_data", 
        n_samples=100, 
        params=RBFDataParams(gamma=2.0),
        seed=123 # Seed for Data Generation
    )

    # Student: Reuploading Quantum Kernel
    method_conf = ReuploadingMethod(
        name="student_reup",
        circuit_config=ReuploadingConfig(2, 2, 2; entanglement=LinearEntanglement),
        optimizer=SPSAConfig(max_iter=20, seed=0)
    )

    # Master Config
    exp_config = ExperimentConfig(
        name="demo_run",
        master_seed=UInt(999),
        data_config=data_conf,
        method=method_conf,
        tuning_batch_size=50, # Force batching to test it
        cv_folds=3
    )

    # ---------------------------------------------------------
    # 2. Data Generation Phase
    # ---------------------------------------------------------
    println("-> Generating Data...")
    # In a real run, you'd call produce_data(data_conf). 
    # For this demo, assuming produce_data exists and returns Dict:
    # (Mocking return for demo if produce_data isn't loaded yet)
    data = produce_data(data_conf) 
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    println("   Data Shapes: X=$(size(X_train)), y=$(size(y_train))")

    # ---------------------------------------------------------
    # 3. Tuning Phase (The "Polymorphic" Step)
    # ---------------------------------------------------------
    println("-> Tuning Kernel (Method: $(typeof(method_conf)))...")
    
    # Derive RNG for tuning
    rng_tuning = derive_rng(exp_config.master_seed, 2002) # SALT_TUNING
    tune_conf = TQK.TuningConfig(rng=rng_tuning, batch_size=exp_config.tuning_batch_size)

    # EXECUTE GENERIC TUNING
    result = tune_kernel(method_conf, X_train, y_train, tune_conf)
    
    println("   Best KTA Score: $(result.best_score)")
    println("   History Steps: $(length(result.history))")

    # ---------------------------------------------------------
    # 4. Final Matrices & SVM Phase
    # ---------------------------------------------------------
    println("-> Computing Final Matrices...")
    # Dispatch handles the specific kernel type automatically!
    K_train = compute_final_matrix(result.best_params, X_train)
    K_test  = compute_final_matrix(result.best_params, X_test) # Note: needs (param, X_test, X_train) in real impl usually

    println("-> Tuning SVM C...")
    c_best, acc_cv, _ = tune_svm_c(K_train, y_train, exp_config.c_grid; cv_folds=exp_config.cv_folds)
    println("   Best C: $c_best, CV Acc: $acc_cv")

    # ---------------------------------------------------------
    # 5. Saving Phase
    # ---------------------------------------------------------
    println("-> Saving Artifacts...")
    
    artifacts = TQK.ExperimentArtifacts(
        exp_config,
        result,
        K_train,
        K_test,
        c_best,
        acc_cv,
        0.0, # train_acc placeholder
        0.0, # test_acc placeholder
        string(now())
    )

    # Call the saver (assuming you added src/reporting.jl)
    # path = TQK.save_experiment_artifacts(artifacts)
    # println("   Saved to: $path")
    
    println("\n=== Demo Complete ===")
end

# Run it
run_demo()
