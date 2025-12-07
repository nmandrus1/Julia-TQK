using TQK
using Yao
using Test
using PythonCall
using LinearAlgebra
using Random

# --- 1. Qiskit Interop Helper ---

"""
    get_qiskit_matrix(config, x)

Calls Qiskit via PythonCall to generate the 'Ground Truth' unitary matrix.
"""
function get_qiskit_matrix(config::PauliConfig, x::Vector{Float64})
    # 1. Imports
    # We use 'PauliFeatureMap' class from circuit library
    # We use 'Operator' from quantum_info to get the matrix
    library = pyimport("qiskit.circuit.library")
    quantum_info = pyimport("qiskit.quantum_info")
    
    # 2. Convert Entanglement Enum to Qiskit String
    ent_str = if config.ent == LinearEntanglement
        "linear"
    elseif config.ent == CircularEntanglement
        "circular"
    elseif config.ent == FullEntanglement
        "full"
    else
        error("Unsupported entanglement for Qiskit comparison: $(config.ent )")
    end

    # 3. Construct the Qiskit Circuit Object
    # Note: Qiskit expects a Python list for paulis
    pfm = library.PauliFeatureMap(
        feature_dimension = config.n_features,
        reps = config.reps,
        entanglement = ent_str,
        paulis = pylist(config.paulis),
        alpha = config.alpha
    )

    # 4. Bind the parameters (x) to the circuit
    # Qiskit's assign_parameters takes a list/array
    bound_pfm = pfm.assign_parameters(x)

    # 5. Extract Unitary Matrix
    # We use the Operator class to calculate the unitary of the circuit
    op = quantum_info.Operator(bound_pfm)
    
    # Convert Python numpy array to Julia Matrix{ComplexF64}
    # Note: Qiskit data is row-major (C-style), Julia is col-major (Fortran-style),
    # but pyconvert usually handles the layout correctly for 2D arrays.
    U_py = op.data
    U_jl = pyconvert(Matrix{ComplexF64}, U_py)

    return U_jl
end

# --- 2. Equivalence Checker ---

"""
    check_unitary_equivalence(U_yao, U_qiskit)

Checks if two unitaries are identical up to a global phase.
Returns true if |Tr(U_yao' * U_qiskit)| / 2^n ≈ 1.0
"""
function check_unitary_equivalence(U_yao::AbstractMatrix, U_qiskit::AbstractMatrix)
    n_dim = size(U_yao, 1)
    
    # Dimension sanity check
    if size(U_yao) != size(U_qiskit)
        @error "Dimension mismatch" size(U_yao) size(U_qiskit)
        return false
    end
    
    # Calculate Operator Fidelity
    # Trace(A' * B) / D
    overlap = tr(U_yao' * U_qiskit) / n_dim
    fidelity = abs(overlap)
    
    # Debug print if test fails
    if !isapprox(fidelity, 1.0; atol=1e-8)
        println("    Mismatch! Fidelity: $fidelity")
        println("    Yao[1,1]: $(U_yao[1,1])")
        println("    Qiskit[1,1]: $(U_qiskit[1,1])")
    end

    return isapprox(fidelity, 1.0; atol=1e-8)
end

# --- 3. The Test Suite ---

@testset "PauliFeatureMap Rigorous Verification" begin
    
    # Set seed for reproducibility of 'x'
    Random.seed!(42)

    @testset "Standard Z-ZZ Feature Maps" begin
        # Case 1.1: 2 Qubits, Linear (Hello World)
        n = 2
        x = rand(n)
        config = PauliConfig(n, ["Z", "ZZ"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)

        # Case 1.2: 3 Qubits, Full Entanglement (Tests connectivity logic)
        n = 3
        x = rand(n)
        config = PauliConfig(n, ["Z", "ZZ"]; reps=1, ent=FullEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    @testset "Higher Repetitions & Custom Alpha" begin
        # Case 2: Deep circuit with scaling
        n = 2
        x = rand(n)
        # alpha=0.5, reps=2
        config = PauliConfig(n, ["Z", "ZZ"]; reps=2, ent=LinearEntanglement, alpha=0.5)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    @testset "Alternative Pauli Basis (X, Y, Mixed)" begin
        # This tests the Basis Change logic (Hadamards, SqrtX, etc.)
        
        # Case 3.1: Single Qubit X and Y rotations
        n = 2
        x = rand(n)
        config = PauliConfig(n, ["X", "Y"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)

        # Case 3.2: Mixed Entanglement (XX, YY, ZX)
        # This validates the native gate selection and ladder logic
        n = 3
        x = rand(n)
        # Using a weird mix to stress test the builder
        config = PauliConfig(n, ["XX", "YY", "ZX"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    @testset "Circular Entanglement" begin
        # Case 4: 3 Qubits Circular (1-2, 2-3, 3-1)
        n = 3
        x = rand(n)
        config = PauliConfig(n, ["Z", "ZZ"]; reps=1, ent=CircularEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end
    
    @testset "Edge Case: Single Qubit" begin
        # Case 5: 1 Qubit (Entanglement settings should be ignored safely)
        n = 1
        x = rand(n)
        config = PauliConfig(n, ["Z", "X"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end
end
