using ArgCheck
using LinearAlgebra
using LinearAlgebra.BLAS
using Yao
using YaoBlocks
using Logging
using Zygote

"""
    FidelityKernel

A quantum kernel implementation using the uncomputation method.
Computes kernel values as K(x,y) = |⟨0|U†(x)U(y)|0⟩|².

# Fields
- `feature_map`: A quantum circuit that maps input data to a quantum state
"""
mutable struct FidelityKernel
    feature_map::AbstractQuantumFeatureMap
end

# --- Core Helper Functions ---

"""
    In-place modification of a Yao statevector to the |0⟩ state
"""
@inline function zero!(statevec::ArrayReg)
    s = state(statevec)
    fill!(s, 0)
    s[1] = 1.0
end

"""
    create_statevec!(register::ArrayReg, feature_map, x::AbstractVector)

Computes the statevector |ψ(x)⟩ = U(x)|0⟩ in-place in the provided register.
"""
@inline function create_statevec!(register::ArrayReg, feature_map, x::AbstractVector)
    zero!(register)
    map_inputs!(feature_map, x)
    apply!(register, feature_map.circuit)
    return register
end


# --- Main Evaluation Functions ---

@inline kernel_fidelity(ψ1::ArrayReg, ψ2::ArrayReg) = abs2(dot(state(ψ1), state(ψ2)))

"""
    evaluate(kernel::FidelityKernel, X::Matrix)

Compute the kernel matrix K(X,X) for training data.
"""
function evaluate(kernel::FidelityKernel, X::AbstractMatrix; workspace::Union{Nothing, AbstractFidelityWorkspace} = nothing)
    n_samples = size(X, 1)

    if isnothing(workspace)
        workspace = create_preallocated_workspace(kernel.feature_map, n_samples, (n_samples, n_samples))
    end

    evaluate!(workspace.K_cache, kernel, X, workspace)
    return workspace.K_cache
end

"""
    evaluate(kernel::FidelityKernel, X::Matrix, Y::Matrix)

Compute the kernel matrix K(X,Y) between two datasets.
"""
function evaluate(kernel::FidelityKernel, X::AbstractMatrix, Y::AbstractMatrix; workspace::Union{Nothing, AbstractFidelityWorkspace}=nothing)
    n_x = size(X, 1)
    n_y = size(Y, 1)

    if isnothing(workspace)
        workspace = create_preallocated_workspace(kernel.feature_map, n_x + n_y, (n_x, n_y))
    end

    evaluate!(workspace.K_cache, kernel, X, Y, workspace)
    return workspace.K_cache
end

"""
    evaluate(kernel::FidelityKernel, x::Vector, y::Vector)

Computes the kernel between two data points.
"""
function evaluate(kernel::FidelityKernel, x::Vector, y::Vector)
    fm = kernel.feature_map
    map_inputs!(kernel.feature_map, x)
    x_statevec = apply!(zero_state(n_qubits(fm)), fm.circuit)

    map_inputs!(kernel.feature_map, y)
    y_statevec = apply!(zero_state(n_qubits(fm)), fm.circuit)
    
    return kernel_fidelity(x_statevec, y_statevec)
end


# Symmetric kernel K(X,X)
function evaluate!(K::Matrix, kernel::FidelityKernel, X::AbstractMatrix, 
                  workspace::AbstractFidelityWorkspace)
    n = size(X, 1)
    statevecs = @view workspace.statevec_pool[1:n]
    
    # Compute all statevectors
    for i in 1:n
        create_statevec!(statevecs[i], kernel.feature_map, @view X[i, :])
    end

    
    # Kernel matrix (upper triangle + diagonal)
    @inbounds for i in 1:n
        K[i,i] = 1.0
        for j in (i+1):n
            val = kernel_fidelity(statevecs[i], statevecs[j])
            K[i,j] = K[j,i] = val
        end
    end
end

# Asymmetric kernel K(X,Y)
function evaluate!(K::Matrix, kernel::FidelityKernel, X::AbstractMatrix, 
                  Y::AbstractMatrix, workspace::AbstractFidelityWorkspace)

    nx, ny = size(X, 1), size(Y, 1)
    x_vecs = @view workspace.statevec_pool[1:nx]
    y_vecs = @view workspace.statevec_pool[nx+1:nx+ny]
    
    for i in 1:nx
        create_statevec!(x_vecs[i], kernel.feature_map, @view X[i, :])
    end
    for j in 1:ny
        create_statevec!(y_vecs[j], kernel.feature_map, @view Y[j, :])
    end
    
    @inbounds for i in 1:nx, j in 1:ny
        K[i,j] = kernel_fidelity(x_vecs[i], y_vecs[j])
    end
end

# --- Gradient Implementations ---

"""
    loss_gradient_finite_diff(kernel, K_cache, loss_fn, X, workspace; ε=1e-5)

Compute gradients using finite differences on parameters.
Simple and robust, requires 2P kernel evaluations.
"""
function loss_gradient_finite_diff(
    kernel::FidelityKernel,
    K_cache::AbstractMatrix,
    loss_fn::Function,
    X::AbstractMatrix,
    workspace::PreallocatedWorkspace;
    ε::Float64=1e-5
)

    fm = kernel.feature_map
    n_p = n_params(fm)
    grad_w = @view workspace.grad_buffer[1:n_p]
    grad_b = @view workspace.grad_buffer[n_p+1:end]
    
    # grad_w = zeros(n_p)
    # grad_b = zeros(n_p)
    weights, biases = get_params(fm)

    
    # Gradients w.r.t. weights
    for i in 1:n_p
        w_plus = copy(weights)
        w_plus[i] += ε
        assign_params!(fm, w_plus, biases)
        evaluate!(K_cache, kernel, X, workspace)
        # K_cache contains K_plus
        loss_plus = loss_fn(K_cache)
        
        w_minus = copy(weights)
        w_minus[i] -= ε
        assign_params!(fm, w_minus, biases)
        evaluate!(K_cache, kernel, X, workspace)
        # K_cache contains K_minus
        loss_minus = loss_fn(K_cache)
        
        grad_w[i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Gradients w.r.t. biases
    for i in 1:n_p
        b_plus = copy(biases)
        b_plus[i] += ε
        assign_params!(fm, weights, b_plus)
        evaluate!(K_cache, kernel, X, workspace)
        # K_cache contains K_plus
        loss_plus = loss_fn(K_cache)
        
        b_minus = copy(biases)
        b_minus[i] -= ε
        assign_params!(fm, weights, b_minus)

        evaluate!(K_cache, kernel, X, workspace)
        # K_cache contains K_minus
        loss_minus = loss_fn(K_cache)
        
        grad_b[i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Restore and compute final loss
    assign_params!(fm, weights, biases)
    evaluate!(K_cache, kernel, X, workspace)
    loss = loss_fn(K_cache)
    
    return loss, (grad_w, grad_b)
end


function loss_gradient_finite_diff(
    kernel::FidelityKernel,
    K_cache::AbstractMatrix,  # For final loss only
    loss_fn::Function,
    X::AbstractMatrix,
    workspace::ThreadAwareWorkspace;
    ε::Float64=1e-5
)
    fm = kernel.feature_map
    n_p = n_params(fm)
    weights, biases = get_params(fm)
       
    # Parallel gradient computation for weights
    Threads.@threads for i in 1:n_p
        tid = Threads.threadid()
        K_local = get_K_cache(workspace, tid)
        grad_local = get_grad_buffer(workspace, tid)
        fm_local = get_feature_map(workspace, tid) # Use thread-local copy
        kernel_local = FidelityKernel(fm_local)
        
        # w[i] + ε
        w_plus = copy(weights)
        w_plus[i] += ε
        assign_params!(fm_local, w_plus, biases)
        evaluate!(K_local, kernel_local, X, get_workspace(workspace, tid))
        loss_plus = loss_fn(K_local)
        
        # w[i] - ε
        w_minus = copy(weights)
        w_minus[i] -= ε
        assign_params!(fm_local, w_minus, biases)
        evaluate!(K_local, kernel_local, X, get_workspace(workspace,tid))
        loss_minus = loss_fn(K_local)
        
        grad_local[i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Parallel gradient computation for biases
    Threads.@threads for i in 1:n_p
        tid = Threads.threadid()
        K_local = get_K_cache(workspace, tid)
        grad_local = get_grad_buffer(workspace, tid)
        fm_local = get_feature_map(workspace, tid) # Use thread-local copy
        kernel_local = FidelityKernel(fm_local)
        
        # b[i] + ε
        b_plus = copy(biases)
        b_plus[i] += ε
        assign_params!(fm_local, weights, b_plus)
        evaluate!(K_local, kernel_local, X, get_workspace(workspace, tid))
        loss_plus = loss_fn(K_local)
        
        # b[i] - ε
        b_minus = copy(biases)
        b_minus[i] -= ε
        assign_params!(fm_local, weights, b_minus)
        evaluate!(K_local, kernel_local, X, get_workspace(workspace, tid))
        loss_minus = loss_fn(K_local)
        
        grad_local[n_p + i] = (loss_plus - loss_minus) / (2ε)
    end
    
    # Combine thread-local gradients
    combined_grad = sum(workspace.thread_grad_buffers)
    grad_w = @view combined_grad[1:n_p]
    grad_b = @view combined_grad[n_p+1:end]
    
    # Compute final loss with original parameters
    assign_params!(fm, weights, biases)
    evaluate!(K_cache, kernel, X, workspace)
    loss = loss_fn(K_cache)
    
    return loss, (grad_w, grad_b)
end
