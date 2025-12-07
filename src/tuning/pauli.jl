using Random

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


function tune_kernel(config::PauliMethod, X, y)
    best_kta = -Inf
    best_paulis = nothing
    
    for i in 1:config.n_search_iterations
        # 1. Sample Structure
        current_paulis = generate_constrained_pauli_set(config.constraints; rng=config.rng)
        
        # 2. Build & Eval
        # (Assuming you write a pure function for Pauli too)
        K = compute_pauli_kernel_matrix(current_paulis, X)
        score = kernel_target_alignment(K, y)
        
        if score > best_kta
            best_kta = score
            best_paulis = current_paulis
        end
    end
    return TrainedPauliKernel(best_paulis)
end
