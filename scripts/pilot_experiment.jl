
using DrWatson
@quickactivate "TQK"

using TQK
using LinearAlgebra

function run_pilot_experiment()
    println("=== STARTING PILOT EXPERIMENT: Hypersphere Geometry ===")
    master_seed::UInt = 2024
    tuning_batch_size = 100
    
    # 1. Fixed Dataset (Teacher)
    #    We use a "hard" quantum teacher to force the students to work.
    #    N=500 is enough to get stable kernel statistics.
    teacher_conf = DataConfig(
        dataset_name = "pilot_teacher_reup_8q",
        n_samples = 1000, 
        n_features = 4, 
        params = ReuploadingDataParams(
            n_qubits=4, n_features=4, n_layers=4, 
            ent=FullEntanglement, n_support_vectors=20
        ),
        master_seed = master_seed
    )

    # 2. Student Configurations
    experiments = ExperimentConfig[]

    # Group A: RBF Baseline (Infinite Dimensional?)
    push!(experiments, ExperimentConfig(
        name = "pilot_student_rbf",
        master_seed = master_seed,
        data_config = teacher_conf,
        method = RBFMethod(gamma_grid=[0.01, 0.1, 1.0, 10.0]),
        tuning_batch_size = tuning_batch_size
    ))

    # Group B: Reuploading - Varying Dimension (Qubits)
    # Hypothesis: Avg Kernel value should drop as 1/2^Q
    for q in [2, 4, 6, 8]
        # We assume n_features <= q. If features=4, we need at least 4 qubits 
        # or we use a map that encodes into available qubits. 
        # For simplicity in this pilot, let's fix features=4 and use q >= 4.
        if q < 4 continue end 

        method = ReuploadingMethod(
            name = "pilot_student_reup_q$(q)_l4",
            circuit_config = ReuploadingConfig(q, 4, 4),
            optimizer = SPSAConfig(max_iter=100, n_resamples=5) # Higher resamples for stability
        )
        
        push!(experiments, ExperimentConfig(
            name = "pilot_student_reup_q$(q)",
            master_seed = master_seed,
            data_config = teacher_conf,
            method = method,
            tuning_batch_size = tuning_batch_size
        ))
    end

    # Group C: Reuploading - Varying Depth (Layers)
    # Hypothesis: Deeper circuits = more random = closer to 1/D
    for l in [1, 2, 4, 8]
        method = ReuploadingMethod(
            name = "pilot_student_reup_q4_l$(l)",
            circuit_config = ReuploadingConfig(4, 4, l),
            optimizer = SPSAConfig(max_iter=100, n_resamples=5)
        )
        
        push!(experiments, ExperimentConfig(
            name = "pilot_student_reup_l$(l)",
            master_seed = master_seed,
            data_config = teacher_conf,
            method = method,
            tuning_batch_size = tuning_batch_size
        ))
    end

    # 3. Execution Loop
    println("-> Generated $(length(experiments)) experiments.")
    for (i, config) in enumerate(experiments)
        println("\n[$(i)/$(length(experiments))] Running $(config.name)...")
        try
            run_experiment(config) # Logic inside experiment_runner.jl handles saving
        catch e
            @error "Failed $(config.name)" exception=e
        end
    end
    
    println("\n=== PILOT COMPLETE ===")
end

run_pilot_experiment()
