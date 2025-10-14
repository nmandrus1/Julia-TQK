
using LinearAlgebra
using Parameters
using PythonCall


# --- Your Functions (Refactored) ---

"Create a Qiskit PauliFeatureMap circuit from Julia configuration."
function create_pauli_feature_map(params::PauliKernelHyperparameters)
    # The constructor is called directly like a Julia function.
    # Note: Keyword arguments are passed using Julia's syntax.
    pauli_feature_map = pyimport("qiskit.circuit.library").pauli_feature_map
    feature_map = pauli_feature_map(
        feature_dimension=params.n_qubits,
        reps=params.reps,
        entanglement=params.entanglement,
        paulis=pylist(params.paulis),
    )

    return feature_map
end

"Compute the quantum kernel matrix for data X using PauliFeatureMap."
function compute_pauli_kernel_matrix(params::PauliKernelHyperparameters, X::Matrix{Float64}, Y::Union{Nothing, Matrix{Float64}}=nothing)
    feature_map = create_pauli_feature_map(params)
    
    # Create the quantum kernel object
    FidelityStatevectorKernel = pyimport("qiskit_machine_learning.kernels").FidelityStatevectorKernel
    quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
    
    # Evaluate the kernel. No need to convert X to a numpy array!
    # PythonCall handles the conversion automatically.
    kernel_matrix_py = if isnothing(Y)
        quantum_kernel.evaluate(x_vec=X)
    else
        quantum_kernel.evaluate(x_vec=X, y_vec=Y)
    end
    
    # Convert back to a Julia matrix. `pyconvert` is the explicit way.
    return pyconvert(Matrix{Float64}, kernel_matrix_py)
end
