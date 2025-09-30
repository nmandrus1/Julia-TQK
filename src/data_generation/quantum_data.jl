# src/data_generation/quantum_expectation.jl
using LinearAlgebra
using Random
using PythonCall
using StatsBase

"""
Generate Pauli operators
"""
pauli_I() = [1 0; 0 1]
pauli_X() = [0 1; 1 0]
pauli_Y() = [0 -im; im 0]
pauli_Z() = [1 0; 0 -1]

function pauli_string_to_operator(pauli_string::String, n_qubits::Int)
    """Convert a Pauli string like 'ZXY' to the full operator on n_qubits"""
    ops = Dict('I' => pauli_I(), 'X' => pauli_X(), 'Y' => pauli_Y(), 'Z' => pauli_Z())
    
    # Start with identity
    result = Complex{Float64}(1)
    
    # Build the operator for specified qubits
    for i in 1:n_qubits
        if i <= length(pauli_string)
            result = kron(result, ops[pauli_string[i]])
        else
            result = kron(result, pauli_I())  # Identity on remaining qubits
        end
    end
    
    return result
end

function random_unitary(dim::Int; seed::Union{Int,Nothing}=nothing)
    """Generate random unitary via QR decomposition of random complex matrix"""
    !isnothing(seed) && Random.seed!(seed)
    
    # Random complex matrix
    A = randn(ComplexF64, dim, dim)
    
    # QR decomposition
    Q, R = qr(A)
    
    # Ensure unitarity by fixing phases
    Lambda = Diagonal(sign.(diag(R)))
    U = Matrix(Q * Lambda)
    
    return U
end

# Alternative: Grid-based sampling for better coverage
function generate_pauli_expectation_data_grid(
    config::PauliFeatureMapConfig,
    n_samples::Int;
    gap::Float64 = 0.3,
    observable::Union{String,Nothing} = nothing,
    grid_points_per_dim::Int = 20,
    seed::Int = 42
)
    Random.seed!(seed)
    
    n_qubits = config.n_qubits
    dim = 2^n_qubits
    
    # Choose observable
    if isnothing(observable)
        observable = config.paulis[argmax(length.(config.paulis))]
    end
    
    O = pauli_string_to_operator(observable, n_qubits)
    V = random_unitary(dim, seed=seed+1)
    M = V' * O * V

    feature_map = create_pauli_feature_map(config)

    "Get the statevector that corresponds to feature transform"
    function statevec_from_feature_map(feature_map, datavec::Vector{Float64})
        Statevector = pyimport("qiskit.quantum_info").Statevector
        sv = Statevector(feature_map.assign_parameters(datavec))
        return pyconvert(Vector{ComplexF64}, sv.data)
    end
    
    # Create grid
    grid_1d = range(0, 2π, length=grid_points_per_dim)
    
    # For efficiency, only create full grid if n_features <= 3
    if config.n_features <= 3
        grid_points = Iterators.product([grid_1d for _ in 1:config.n_features]...)
        
        valid_points = []
        labels = []
        exp_vals = []
        
        for x_tuple in grid_points
            x = collect(x_tuple)
            psi = statevec_from_feature_map(feature_map, x)
            exp_val = real(psi' * M * psi)
            
            if abs(exp_val) > gap
                push!(valid_points, x)
                push!(labels, sign(exp_val))
                push!(exp_vals, exp_val)
            end
        end
        
        # Sample from valid points
        n_per_class = n_samples ÷ 2
        pos_indices = findall(labels .== 1)
        neg_indices = findall(labels .== -1)
        
        if length(pos_indices) < n_per_class || length(neg_indices) < n_per_class
            @error"Insufficient separable points"
        end
        
        selected_pos = sample(pos_indices, n_per_class, replace=false)
        selected_neg = sample(neg_indices, n_per_class, replace=false)
        
        X = vcat([valid_points[i]' for i in selected_pos]...,
                 [valid_points[i]' for i in selected_neg]...)
        y = vcat(ones(n_per_class), -ones(n_per_class))
        
        return Dict(:X => X, :y => y, :config => config, :gap => gap, :seed=> seed)
    else
        @error "Data dimension must be <= 3"
    end
end
