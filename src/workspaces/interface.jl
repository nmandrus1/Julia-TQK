
"""
Abstract interface for quantum kernel workspaces.
All workspace implementations must provide memory management for statevectors and gradients.
"""
abstract type AbstractFidelityWorkspace end

# Required interface functions
function get_vectors! end
function get_grad_buffers! end
function reset! end
function get_forward_tile_size end
function get_backward_tile_size end


# --- Helper functions for memory views ---

"""
    WorkspaceViews

Convenience struct for managing different memory regions during computation.
"""
struct WorkspaceViews{T<:AbstractVector{<:ArrayReg}}
    statevecs::T
    adjoints::Union{T,Nothing}
    col_statevecs::Union{T,Nothing}  # For tiled operations
end

"""
    get_forward_views(workspace::AbstractFidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for forward pass computation.
"""
function get_forward_views(workspace::AbstractFidelityWorkspace, n_samples::Int)
    statevecs = get_vectors!(workspace, n_samples)
    return WorkspaceViews(statevecs, nothing, nothing)
end

"""
    get_backward_views(workspace::AbstractFidelityWorkspace, n_samples::Int) -> WorkspaceViews

Get memory views configured for backward pass computation (2-way split).
"""
function get_backward_views(workspace::AbstractFidelityWorkspace, n_samples::Int)
    statevecs = get_vectors!(workspace, n_samples)
    adjoints = get_vectors!(workspace, n_samples)
    return WorkspaceViews(statevecs, adjoints, nothing)
end

"""
    get_backward_tiled_views(workspace::AbstractFidelityWorkspace, row_size::Int, col_size::Int) -> WorkspaceViews

Get memory views configured for tiled backward pass (3-way split).
"""
function get_backward_tiled_views(workspace::AbstractFidelityWorkspace, row_size::Int, col_size::Int)
    row_statevecs = get_vectors!(workspace, row_size)
    adjoints = get_vectors!(workspace, row_size)
    col_statevecs = get_vectors!(workspace, col_size)
    return WorkspaceViews(row_statevecs, adjoints, col_statevecs)
end

"""
    extract_gradients(workspace::AbstractFidelityWorkspace) -> (weights, biases)

Extract weight and bias gradients from the interleaved gradient buffer.
"""
function extract_gradients(workspace::AbstractFidelityWorkspace)
    _, _, grad_params = get_grad_buffers!(workspace)
    # Slicing already creates a copy, no need for additional copy()
    d_weights = @view grad_params[1:2:end]
    d_biases = grad_params[2:2:end]
    return d_weights, d_biases
end
