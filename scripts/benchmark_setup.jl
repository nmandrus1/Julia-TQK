using DrWatson
@quickactivate "TQK"

# simple script that setups variables in qiskit ordering
# for a simple benchmark in the REPL

using Yao
using BenchmarkTools

function setup_benchmark_vars(; num_qubits = 4, num_features = 6, num_layers = 2, entanglement::EntanglementBlock = linear, use_sequential_weights = true, random_data = false)

    reup = ReuploadingCircuit(num_qubits, num_features, num_layers, entanglement)

    if use_sequential_weights   
        num_multiples = ceil(num_features/3) - 1
        repeating_assignment = [w for n in 0:num_multiples for w in [3.0, 1.0, 2.0].+ 3n]
        weights = [elem for n in 0:(num_qubits * num_layers - 1) for elem in repeating_assignment .+ num_features*n]
        x = ones(num_features)
    else
        weights = ones(n_parameters(reup))
        x = [float(i) for i in 1:num_features]
    end

    assign_params!(reup, weights, reup.biases)

    kernel = FidelityKernel(reup, use_cache=false, parallel=false)

    if random_data
        X_data = rand(500, num_features)
    else
        X_data = [num_features*r + c for r in 0:499, c in 0:num_features-1]
    end
    return reup, kernel, weights, x, X_data
end
