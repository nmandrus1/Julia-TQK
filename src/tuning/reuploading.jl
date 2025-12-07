function tune_kernel(config::ReuploadingKernelHyperparameterSearchConfig, X, y)
    # 1. Setup Pure Loss
    # Note: We optimize the CONFIG, not the kernel object
    loss_fn(params) = kernel_target_alignment_loss(config, params, X, y)
    
    # 2. Run Optimization (SPSA)
    init_params = rand(config.total_params)
    opt_result = train_pure_optimizer(loss_fn, init_params)
    
    return ReuploadingKernelHyperparameters(thetas=opt_result.u)
end
