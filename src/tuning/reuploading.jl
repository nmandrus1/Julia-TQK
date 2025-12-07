function tune_kernel(config::ReuploadingMethod, X, y)
    # 1. Setup Pure Loss
    # Note: We optimize the CONFIG, not the kernel object
    loss_fn(params) = variational_kta_loss(config, params, X, y)
    
    # 2. Run Optimization (SPSA)
    init_params = rand(config.circuit_config.total_params)
    opt_result = train_pure_optimizer(loss_fn, init_params)
    
    return ReuploadingKernelHyperparameters(thetas=opt_result.u)
end
