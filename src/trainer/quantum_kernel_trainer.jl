using LinearAlgebra
using Optimization
using Random

"""
    QuantumKernelTrainer

Manages quantum kernel training with Optimization.jl integration.
Maintains memory efficiency through workspace reuse.
"""
mutable struct QuantumKernelTrainer
    kernel::FidelityKernel
    loss_fn::Function
    X::Matrix{Float64}
    y::Vector{Float64}
    workspace::AbstractFidelityWorkspace
    # Cache for kernel matrix to avoid recomputation
    K_cache::Matrix{Float64}
    grad_cache::AbstractVector{Float64}
end

"""
    QuantumKernelTrainer(kernel, loss_fn, X, y; workspace=nothing, memory_budget_gb=4.0)

Create a trainer for quantum kernel optimization.

# Arguments
- `kernel`: FidelityKernel to train
- `loss_fn`: Loss function that takes (K, y) and returns scalar
- `X`: Training data (n_samples × n_features)
- `y`: Target labels
- `workspace`: Optional pre-allocated workspace
- `memory_budget_gb`: Memory budget if creating new workspace
"""
function QuantumKernelTrainer(
    kernel::FidelityKernel, 
    loss_fn::Function,
    X::Matrix{Float64}, 
    y::Vector{Float64};
    memory_budget_gb::Float64 = 4.0
)
    # Create workspace if not provided
    workspace = create_preallocated_workspace(
        kernel.feature_map, 
        size(X, 1), 
        memory_budget_gb=memory_budget_gb
    )
    grad_cache = zeros(n_params(kernel.feature_map) * 2)
    
    return QuantumKernelTrainer(kernel, loss_fn, X, y, workspace, zeros(size(X, 1), size(X, 1)), grad_cache)
end

# ==============================
# Optimization.jl Interface
# ==============================

"""
    create_optimization_function(trainer::QuantumKernelTrainer)

Creates an OptimizationFunction compatible with Optimization.jl.
Returns functions for objective and gradient computation.
"""
function create_optimization_function(trainer::QuantumKernelTrainer)
    # Objective function
    function objective!(params, p=nothing)
        # compute everything here and only return gradients in the gradient! function
        # Unpack parameters
        nparams = n_params(trainer.kernel.feature_map)
        weights = @view params[1:nparams]
        biases = @view params[nparams+1:end]
        
        # Update kernel parameters
        assign_params!(trainer.kernel.feature_map, weights, biases)
        
        # Compute loss
        evaluate!(trainer.K_cache, trainer.kernel, trainer.X, trainer.workspace)

        # store gradient of parameters in workspace
        #loss, _ = loss_gradient(trainer.kernel, trainer.K_cache, trainer.loss_fn, trainer.X, trainer.workspace)
        loss = trainer.loss_fn(trainer.K_cache)
        
        return loss
    end
    
    # Gradient function
    function gradient!(grad, params, p=nothing)
        # Unpack parameters
        nparams = n_params(trainer.kernel.feature_map)
       
        weights = @view params[1:nparams]
        biases = @view params[nparams+1:end]
        
        # Update kernel parameters
        assign_params!(trainer.kernel.feature_map, weights, biases)
        
        # Compute loss
        evaluate!(trainer.K_cache, trainer.kernel, trainer.X, trainer.workspace)

        # store gradient of parameters in workspace
        _, _ = loss_gradient(trainer.kernel, trainer.K_cache, trainer.loss_fn, trainer.X, trainer.workspace)
        
        # Pack gradients
        _, _, grad_params = get_grad_buffers!(trainer.workspace)
        grad[1:nparams] .= grad_params[1:nparams]
        grad[nparams+1:end] .= grad_params[nparams+1:end]
        
        return nothing
    end
        
    return objective!, gradient!
end

"""
    pack_parameters(weights::Vector, biases::Vector) -> Vector

Pack weights and biases into single parameter vector for Optimization.jl.
"""
function pack_parameters(weights::Vector{Float64}, biases::Vector{Float64})
    return vcat(weights, biases)
end

"""
    unpack_parameters(params::Vector, n_weights::Int) -> (weights, biases)

Unpack parameter vector into weights and biases.
"""
function unpack_parameters(params::Vector{Float64}, n_weights::Int)
    weights = params[1:n_weights]
    biases = params[n_weights+1:end]
    return weights, biases
end

# ==============================
# Convenience Training Function
# ==============================

"""
    train!(trainer::QuantumKernelTrainer; 
          optimizer=nothing, 
          initial_params=nothing,
          iterations=100,
          callback=nothing,
          kwargs...)

Train the quantum kernel using Optimization.jl.

# Arguments
- `trainer`: QuantumKernelTrainer instance
- `optimizer`: Optimization.jl optimizer (e.g., LBFGS(), Adam())
- `initial_params`: Initial parameter vector (defaults to current kernel params)
- `iterations`: Maximum iterations
- `callback`: Optional callback function(state, loss_val)
- `kwargs`: Additional arguments passed to solve()

# Returns
- Solution from Optimization.jl
"""
function train!(trainer::QuantumKernelTrainer; 
               optimizer=nothing,
               initial_params=nothing,
               iterations=100,
               callback=nothing,
               kwargs...)
    
    # Get current parameters if not provided
    if isnothing(initial_params)
        weights, biases = get_params(trainer.kernel.feature_map)
        initial_params = pack_parameters(weights, biases)
    end
    
    # Create optimization function
    obj!, grad! = create_optimization_function(trainer)
        
    # Create OptimizationFunction
    optf = OptimizationFunction(obj!, grad=grad!)
    
    # Create OptimizationProblem
    prob = OptimizationProblem(optf, initial_params)
    
    # Default optimizer if not provided
    if optimizer === nothing
        optimizer = OptimizationOptimJL.LBFGS()
    end
    
    # Create callback wrapper if provided
    opt_callback = nothing
    if callback !== nothing
        opt_callback = (state, loss_val) -> begin
            callback(state, loss_val)
            return false  # Don't stop
        end
    end
    
    # Solve
    sol = solve(prob, optimizer; 
                maxiters=iterations, 
                callback=opt_callback,
                kwargs...)
    
    # Update kernel with final parameters
    nparams = n_params(trainer.kernel.feature_map)
    final_weights, final_biases = unpack_parameters(sol.u, nparams)
    assign_params!(trainer.kernel.feature_map, final_weights, final_biases)
    
    return sol
end

# ==============================
# Loss Functions
# ==============================

"""
    kernel_alignment_loss(K::Matrix, y::Vector)

MSE loss between kernel and ideal kernel y*y'.
"""
function kernel_alignment_loss(K::Matrix, y::Vector)
    y_outer = y * y'
    return sum((K - y_outer).^2)
end

"""
    kernel_target_alignment(K::Matrix, y::Vector)

Kernel-target alignment for classification.
Higher is better (minimize negative).
"""
function kernel_target_alignment(K::Matrix, y::Vector)
    y_outer = y * y'
    return -tr(K * y_outer) / (sqrt(tr(K * K)) * sqrt(tr(y_outer * y_outer)))
end

# ==============================
# Example Usage
# ==============================

function example_usage()
    # Setup
    Random.seed!(42)
    
    # Create feature map and kernel
    fm = ReuploadingCircuit(4, 2, 2, linear)
    assign_random_params!(fm)
    kernel = FidelityKernel(fm)
    
    # Create data
    n_samples = 50
    X = rand(n_samples, 2)
    y = [i <= n_samples÷2 ? -1.0 : 1.0 for i in 1:n_samples]
    
    # Create trainer with pre-allocated workspace
    trainer = QuantumKernelTrainer(
        kernel, 
        kernel_alignment_loss,
        X, 
        y,
        memory_budget_gb=2.0
    )
    
    # Option 1: Use convenience function
    sol = train!(trainer, 
                optimizer=LBFGS(),
                iterations=50,
                callback=(state, loss) -> println("Iter $(state.iter): loss = $loss"))
    
    
    # Get optimization functions
    obj, grad!, obj_and_grad! = create_optimization_function(trainer)
    
    # Create problem
    weights, biases = get_params(kernel.feature_map)
    initial_params = pack_parameters(weights, biases)
    
    optf = OptimizationFunction(obj_and_grad!, grad=grad!)
    prob = OptimizationProblem(optf, initial_params)
    
    # # Solve with different optimizer
    # using OptimizationOptimisers
    # sol2 = solve(prob, Adam(0.01), maxiters=100)
    
    return sol, sol2
end
