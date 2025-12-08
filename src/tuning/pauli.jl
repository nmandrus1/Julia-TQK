using StableRNGs

"""
Generate a random valid Pauli string set respecting constraints
"""
function generate_constrained_pauli_set(
    constraints::PauliSearchConstraints;
    rng::AbstractRNG
)   
    base_paulis = constraints.base_paulis
    max_order = constraints.max_pauli_order
    max_terms = constraints.max_num_terms
    min_terms = constraints.min_num_terms
    
    # Ensure we can have entanglement
    if max_order < 2
        error("max_pauli_order must be >= 2 for entanglement")
    end
    
    pauli_set = Set{String}()
    
    # Add one mandatory entangling term
    order = rand(rng, 2:max_order)
    mandatory_term = join(rand(rng, base_paulis, order))
    push!(pauli_set, mandatory_term)
    
    # Add remaining terms
    num_terms = rand(rng, min_terms:max_terms)
    while length(pauli_set) < num_terms
        order = rand(rng, 1:max_order)
        new_term = join(rand(rng, base_paulis, order))
        push!(pauli_set, new_term)
    end
    
    # Return as shuffled vector
    result = collect(pauli_set)
    shuffle!(rng, result)
    return result
end

"""
    tune_kernel(method::PauliMethod, X, y, config::TuningConfig)

Performs Random Search over the space of valid Pauli strings to maximize KTA.
"""
function tune_kernel(method::PauliMethod, X::AbstractMatrix, y::AbstractVector, config::TuningConfig)
    sample_rng = config.sampling_rng # Use the derived RNG
    optimizer_rng = config.optimizer_rng # Use the derived RNG
    best_kta = -Inf
    
    # Default fallback
    best_paulis = ["Z" for _ in 1:method.n_features]
    history = Float64[]

    for i in 1:method.search_iterations
        # 1. Sample Structure using the derived RNG
        current_paulis = generate_constrained_pauli_set(method.constraints; rng=optimizer_rng)
        
        # 2. Define Kernel Function Closure
        # compute_pauli_kernel_matrix must be defined in your feature_maps/pauli.jl
        config = PauliConfig(
                    n_features=method.n_features,
                    paulis=current_paulis,
                    reps=method.n_reps,
                    ent=method.ent,
                )

        kernel_func(data) = compute_kernel_matrix_pure(config, data)
        
        # 3. Compute Score (Batched or Full)
        score = compute_batched_kta(kernel_func, X, y, config.batch_size, sample_rng)
        push!(history, score)
        
        if score > best_kta
            best_kta = score
            best_paulis = current_paulis
        end
    end
    
    return TuningResult(
        TrainedPauliKernel(best_paulis, method.n_features),
        best_kta,
        history
    )
end

# Implement the dispatch for the final matrix computation
function compute_final_matrix(k::TrainedPauliKernel, X)
    return compute_pauli_kernel_matrix(k.paulis, X)
end
