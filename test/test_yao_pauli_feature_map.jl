using TQK
using Yao
using Test
using PythonCall
using LinearAlgebra
using Random
using Zygote

# --- 1. Qiskit Interop Helper ---


"""
    finite_difference_gradient(f, x; ε=1e-5)

Computes the gradient of scalar function f at x using central finite differences.
"""
function finite_difference_gradient(f, x::Float64; ε=1e-5)
    # Perturb +ε
    y_plus = f(x + ε)
    # Perturb -ε
    y_minus = f(x - ε)
    
    # Central difference
    grad = (y_plus - y_minus) / (2ε)
    return grad
end


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

    
    @testset "Diagnostic: XX Only" begin
        # Case D.1: 2 Qubits, Single XX interaction
        # Qiskit uses RXX(theta) here. Yao uses H-CNOT-Rz-CNOT-H.
        # They must match.
        n = 2
        x = rand(n)
        config = PauliConfig(n, ["XX"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    
    @testset "Diagnostic: XX Only" begin
        # Case D.1: 2 Qubits, Single XX interaction
        # Qiskit uses RXX(theta) here. Yao uses H-CNOT-Rz-CNOT-H.
        # They must match.
        n = 2
        x = rand(n)
        config = PauliConfig(n, ["XX"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    
    @testset "Diagnostic: XZ Only" begin
        # Case D.3: 2 Qubits, Single XZ interaction
        # Just like ZX, but indices swapped. 
        # X on q1, Z on q2.
        n = 2
        x = rand(n)
        config = PauliConfig(n, ["XZ"]; reps=1, ent=LinearEntanglement)
        
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
        
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    @testset "Diagnostic: Ladder Parity" begin
        n = 2
        indices = [1, 2]
        k = 2
        
        # Manually rebuild the ladder logic from pauli_evolution_block
        ladder_ops = [cnot(indices[i+1], indices[i]) for i in 1:(k-1)]
        blk = chain(n, ladder_ops...) # Should be cnot(2, 1)
        
        # Test Parity Logic: |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
        # (Remember Yao is Little Endian: q1 is LSB. 
        # But indices=[1,2] means q1=1, q2=2. 
        # State |q2 q1>. |10> means q2=1, q1=0).
        
        # Case A: Input |10> (q2=1, q1=0) -> Expect |11> (q2=1, q1=1)
        reg = product_state(bit"10") |> blk
        @test isapprox(statevec(reg)[4], 1.0) # Index 4 is |11>
        
        # Case B: Input |00> -> Expect |00>
        reg = product_state(bit"00") |> blk
        @test isapprox(statevec(reg)[1], 1.0)
    end
    
    @testset "Diagnostic: Rotation Only" begin
        n = 2
        theta = π # Full rotation for easy checking (Z -> -Z)
        
        # Manually rebuild rotation logic
        indices = [1, 2]
        target_qubit = indices[1] # Qubit 1
        rot_op = put(n, target_qubit => Rz(theta))
                
        # Apply to |00> (q1=0). Expect -i phase (wait, Rz definition)
        # Yao Rz(theta) = exp(-i theta/2 Z).
        # theta=pi -> exp(-i pi/2 Z) = -i Z.
        # Z|0> = |0>. exp -> -i.
        reg = product_state(bit"00") |> rot_op
        @test isapprox(statevec(reg)[1], 0.0 - 1.0im)
    end


    @testset "Diagnostic: Deep Inspection" begin
        println("\n--- DEEP INSPECTION ---")
        n = 2
        # Use a fixed X to ensure deterministic matrices
        x = [0.5, 0.5] 
        # "XZ": Qiskit likely does Z on q0, X on q1. Yao does X on q1, Z on q2.
        config = PauliConfig(n, ["XZ"]; reps=1, ent=LinearEntanglement)
        
        # Build circuits
        circ = build_circuit(config, [], x)
        U_yao = mat(circ)
        U_qiskit = get_qiskit_matrix(config, x)

        println("Yao Block Structure:")
        println(circ) 
        
        println("\nYao Matrix [1:2, 1:2]:")
        display(U_yao[1:2, 1:2])
        
        println("\nQiskit Matrix [1:2, 1:2]:")
        display(U_qiskit[1:2, 1:2])
        
        # If my hypothesis is right, these will look totally different.
        @test !isapprox(U_yao, U_qiskit) # Expect failure
    end    

    
    @testset "Final Verification: Endianness Fix" begin
        # This previously failed with fidelity ~0.78
        n = 2
        x = rand(n)
        # "XZ" -> Should now map to Z on 1, X on 2 (Matching Qiskit)
        config = PauliConfig(n, ["XZ"]; reps=1, ent=LinearEntanglement)
    
        U_yao = mat(build_circuit(config, [], x))
        U_qiskit = get_qiskit_matrix(config, x)
    
        @test check_unitary_equivalence(U_yao, U_qiskit)
    end

    # --- 2. The Verification Test ---
    @testset "Pauli KTA Gradient Verification" begin
        # Data Setup
        n_qubits = 2
        n_samples = 4
        X = rand(n_samples, n_qubits) # Random features
        y = sign.(randn(n_samples))   # Random labels (+1/-1)
    
        # Initial Alpha (The parameter we want to optimize)
        alpha_init = 2.0
    
        # Define Loss Function w.r.t Alpha
        # We must reconstruct the config inside the function so Zygote sees 
        # 'a' as a variable entering the graph.
        function loss_fn(a)
            c = PauliConfig(n_qubits, ["Z", "ZZ"]; reps=1, alpha=a)
            # Pass empty [] for params, as standard Pauli has no trainable weights yet
            return variational_kta_loss(c, [], X, y)
        end

        println("\n--- Starting KTA Gradient Check ---")

        # 1. Forward Pass
        loss_val = loss_fn(alpha_init)
        println("Forward Loss (Negative KTA): $loss_val")
        @test isa(loss_val, Real)

        # 2. Zygote Gradient
        println("Computing Zygote Gradient...")
        grad_zygote = Zygote.gradient(loss_fn, alpha_init)[1]
        println(" -> Zygote Grad: $grad_zygote")
    
        # 3. Finite Difference Gradient
        println("Computing Finite Difference Gradient...")
        grad_fd = finite_difference_gradient(loss_fn, alpha_init)
        println(" -> FiniteDiff Grad: $grad_fd")

        # 4. Comparison
        # We use reasonable tolerances for floating point arithmetic
        @test isapprox(grad_zygote, grad_fd; rtol=1e-4, atol=1e-4)
    
        diff = abs(grad_zygote - grad_fd)
        println("Absolute Difference: $diff")
    
        if diff < 1e-4
            println("SUCCESS: Gradients match! The pipeline is differentiable.")
        else
            println("FAILURE: Gradients diverge.")
        end
    end
end
