abstract type AbstractKernelTuner end

"""
    tune_kernel(config, X, y) -> OptimizedParams

Universal interface.
- RBF: Performs Grid Search to maximize KTA.
- Pauli: Performs Random Search to maximize KTA.
- Reuploading: Performs Gradient Descent to maximize KTA.
"""
function tune_kernel(config::KernelHyperparameterSearchConfig, X, y)
    error("Not implemented")
end
