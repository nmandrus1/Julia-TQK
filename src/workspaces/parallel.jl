
"""
    ThreadAwareWorkspace <: AbstractFidelityWorkspace

Workspace optimized for parallel finite difference gradient computation.

# Fields
- `statevec_pool::Vector{ArrayReg}`: Shared read-only quantum registers
- `thread_K_caches::Vector{Matrix{Float64}}`: One K matrix per thread
- `thread_grad_buffers::Vector{Vector{Float64}}`: One gradient buffer per thread
- `n_threads::Int`: Number of threads
"""
struct ThreadAwareWorkspace <: AbstractFidelityWorkspace
    thread_statevec_pool::Vector{Vector{ArrayReg}}
    thread_K_caches::Vector{Matrix{Float64}}
    thread_grad_buffers::Vector{Vector{Float64}}
    thread_feature_maps::Vector{ReuploadingCircuit}
    n_threads::Int
end

struct ThreadLocalWorkspace <: AbstractFidelityWorkspace
    statevec_pool::AbstractVector{ArrayReg}
    K_cache::AbstractMatrix{Float64}
    grad_buffer::AbstractVector{Float64}
end

"""
    create_thread_aware_workspace(feature_map, max_samples, K_dims; 
                                   memory_fraction=0.7)

Create workspace with thread-local caches and memory validation.

# Errors
Throws if required memory exceeds `memory_fraction` of system memory.
"""
function create_thread_aware_workspace(
    feature_map::AbstractQuantumFeatureMap,
    max_samples::Int,
    K_dims::Tuple{Int, Int};
    memory_fraction::Float64=0.7
)
    num_qubits = n_qubits(feature_map)
    num_params = n_params(feature_map)
    n_threads = Threads.nthreads()
    
    # Memory calculations (bytes)
    bytes_per_statevec = 2^num_qubits * sizeof(ComplexF64)
    statevec_memory = max_samples * bytes_per_statevec
    
    K_memory_per_thread = prod(K_dims) * sizeof(Float64)
    grad_memory_per_thread = 2 * num_params * sizeof(Float64)
    total_required = (statevec_memory + K_memory_per_thread + grad_memory_per_thread) * n_threads
    
    available = Sys.total_memory() * memory_fraction
    
    if total_required > available
        gb_required = total_required / 1024^3
        gb_available = available / 1024^3
        error("""
        Insufficient memory for $n_threads threads:
          Required: $(round(gb_required, digits=2)) GB
          Available ($(Int(memory_fraction*100))%): $(round(gb_available, digits=2)) GB
          
        Solutions:
          - Reduce threads: export JULIA_NUM_THREADS=<smaller number>
          - Reduce max_samples: $max_samples -> $(div(max_samples, 2))
          - Increase memory_fraction (risky): $memory_fraction -> $(memory_fraction + 0.1)
        """)
    end
    
    @info "Thread-aware workspace" n_threads gb_total=round(total_required/1024^3, digits=2)
    
    # Allocate resources
    thread_statevec_pools = [[zero_state(ComplexF64, num_qubits) for _ in 1:max_samples] for _ in 1:n_threads]
    thread_K_caches = [zeros(Float64, K_dims) for _ in 1:n_threads]
    thread_grad_buffers = [zeros(Float64, 2 * num_params) for _ in 1:n_threads]
    thread_feature_maps = [deepcopy(feature_map) for _ in 1:n_threads]
    
    return ThreadAwareWorkspace(
        thread_statevec_pools,
        thread_K_caches,
        thread_grad_buffers,
        thread_feature_maps,
        n_threads
    )
end

# Interface implementations
get_K_cache(ws::ThreadAwareWorkspace, thread_id::Int=Threads.threadid()) = 
    ws.thread_K_caches[thread_id-1]

get_grad_buffer(ws::ThreadAwareWorkspace, thread_id::Int=Threads.threadid()) = 
    ws.thread_grad_buffers[thread_id-1]

get_feature_map(ws::ThreadAwareWorkspace, thread_id::Int=Threads.threadid()) = 
    ws.thread_feature_maps[thread_id-1]

function get_statevectors(ws::ThreadAwareWorkspace, thread_id::Int=Threads.threadid())
    if thread_id == 1
        return ws.thread_statevec_pool[1]
    else
        return ws.thread_statevec_pool[thread_id-1]
    end
end

reset!(ws::ThreadAwareWorkspace) = 
    foreach(fill!, ws.thread_grad_buffers, 0.0)

get_workspace(ws::ThreadAwareWorkspace, thread_id::Int=Threads.threadid()) =
    ThreadLocalWorkspace(get_statevectors(ws, thread_id), get_K_cache(ws, thread_id), get_grad_buffer(ws, thread_id))


get_statevectors(ws::ThreadLocalWorkspace) = ws.statevec_pool
