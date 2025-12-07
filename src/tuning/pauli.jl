function tune_kernel(config::PauliMethod, X, y)
    best_kta = -Inf
    best_paulis = nothing
    
    for i in 1:config.n_search_iterations
        # 1. Sample Structure
        current_paulis = generate_constrained_pauli_set(...)
        
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
