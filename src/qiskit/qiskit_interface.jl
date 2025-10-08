
using LinearAlgebra
using Parameters
using PythonCall


# --- Your Functions (Refactored) ---

"Create a Qiskit PauliFeatureMap circuit from Julia configuration."
function create_pauli_feature_map(config::PauliKernelHyperparameterSearchConfig)
    # The constructor is called directly like a Julia function.
    # Note: Keyword arguments are passed using Julia's syntax.
    pauli_feature_map = pyimport("qiskit.circuit.library").pauli_feature_map
    feature_map = pauli_feature_map(
        feature_dimension=config.n_features,
        reps=config.reps,
        entanglement=config.entanglement,
        paulis=pylist(config.paulis),
    )

    return feature_map
end

"Compute the quantum kernel matrix for data X using PauliFeatureMap."
function compute_pauli_kernel_matrix(config::PauliKernelHyperparameterSearchConfig, X::Matrix{Float64})
    feature_map = create_pauli_feature_map(config)
    
    # Create the quantum kernel object
    FidelityStatevectorKernel = pyimport("qiskit_machine_learning.kernels").FidelityStatevectorKernel
    quantum_kernel = FidelityStatevectorKernel(feature_map=feature_map)
    
    # Evaluate the kernel. No need to convert X to a numpy array!
    # PythonCall handles the conversion automatically.
    kernel_matrix_py = quantum_kernel.evaluate(x_vec=X)
    println(kernel_matrix_py)
    
    # Convert back to a Julia matrix. `pyconvert` is the explicit way.
    return pyconvert(Matrix{Float64}, kernel_matrix_py)
end

"Get properties of the PauliFeatureMap circuit."
function get_circuit_properties(config::PauliKernelHyperparameterSearchConfig)
    feature_map = create_pauli_feature_map(config)
    
    # Access attributes and call methods directly using dot notation!
    # No py"..." needed.
    gate_counts_py = feature_map.count_ops()

    properties = Dict(
        :depth => feature_map.depth(),
        :n_qubits => feature_map.num_qubits,
        :n_parameters => feature_map.num_parameters,
        # pyconvert automatically handles the Python dict -> Julia Dict
        :gate_counts => pyconvert(Dict{String, Int}, gate_counts_py)
    )
    return properties
end

"Generate a visual representation of the circuit."
function visualize_circuit(config::PauliKernelHyperparameterSearchConfig; filename=nothing)
    feature_map = create_pauli_feature_map(config)
    decomposed_map = feature_map.decompose()

    if isnothing(filename)
        # Call the method directly and print the result
        println(decomposed_map.draw(output="text"))
    else
        # PythonCall knows how to handle Julia strings and keyword arguments
        decomposed_map.draw(output="mpl", filename=filename)
        println("Circuit diagram saved to $(filename)")
    end
end
