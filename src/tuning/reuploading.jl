using StableRNGs

function tune_kernel(method::ReuploadingMethod, X::AbstractMatrix, y::AbstractVector, config::TuningConfig)
    rng = config.rng
    
    # 1. Define Loss Function with Batching
    function loss_fn(params)
        # Create a kernel function closure for the current params
        # Note: We rebuild the circuit factory for every evaluation to be pure/Zygote compatible
        kernel_func(data) = compute_kernel_matrix_hardware(method.circuit_config, params, data)
        
        # Calculate KTA (Negative for minimization)
        score = compute_batched_kta(kernel_func, X, y, config.batch_size, rng)
        return -score 
    end
    
    # 2. Optimize
    init_params = rand(rng, method.circuit_config.total_params)
    opt_params, history = optimize_spsa(loss_fn, init_params, method.optimizer)
    
    # Convert loss history back to KTA scores
    kta_history = -1.0 .* history
    best_score = maximum(kta_history)
    
    return TuningResult(
        TrainedReuploadingKernel(method.circuit_config, opt_params),
        best_score,
        kta_history
    )
end
