using DrWatson
@quickactivate "TQK"

using TQK 
using Yao
using LinearAlgebra
using Random
using StatsBase
using Plots

# We want to test if a quantum feature map is producing "random" vectors
# given a feature map F(x), we are going to generate datapoints, and compute
#     the feature vectors. Then we compute the density matrices for them all
#     and we use these to test for randomness by checking how much the
#     density matrix looks maximally mixed (maximum classical uncertainty)

function create_reuploading_circuit_with_random_params(n_qubits::Int, n_features::Int, n_layers::Int, ent::EntanglementBlock; seed=42)
    fm = ReuploadingCircuit(n_qubits, n_features, n_layers, ent)
    assign_random_params!(fm)
    return fm
end

"""
    generate_random_vectors(dim::Int=4, num_samples::Int=100, normalize::Bool=true)

Generates a matrix of random vectors (size dim x num_samples).

If normalize is true, each column vector is normalized to have unit Euclidean norm, 
placing the vectors on the unit sphere, which is required for comparing to the RBF 
and standard QKernel frameworks.
"""
function generate_random_vectors(dim::Int=4, num_samples::Int=100, normalize::Bool=true)
    # Generate random numbers from a standard normal distribution N(0, 1)
    # The matrix X has dimensions (dim, num_samples)
    X = randn(dim, num_samples)

    if normalize
        # Iterate over columns (each sample vector) and normalize its L2 norm
        for i in 1:num_samples
            # Use @views for efficiency to avoid creating a new vector for each column
            @views X[:, i] = X[:, i] / norm(X[:, i])
        end
    end

    return X
end


function compute_density_matrices(fm::AbstractQuantumFeatureMap, X::AbstractMatrix)
    n = n_qubits(fm)
    
    # Use a generator/comprehension to map each input to a density matrix
    # The result, `density_matrices`, is a Vector{Matrix{ComplexF64}}
    density_matrices = [
        begin
            # Map input to the feature map parameters
            map_inputs!(fm, input)
            
            # Apply the circuit to the initial state (need to ensure initial_state is reset or copied)
            # NOTE: We assume 'apply!' modifies a copy of the state internally 
            # or that the user handles the state reset outside this function.
            # A safer approach is to apply to a new state each time:
            encoding = apply!(zero_state(n), fm.circuit)
            
            # Compute and return the density matrix
            dm = density_matrix(encoding)
            dm.state
        end
        for input in eachcol(X)
    ]
    
    return density_matrices
end


"""
    average_and_plot_density_matrices(density_matrices::Vector{Matrix{ComplexF64}})

Computes the average density matrix from a collection of matrices and displays a 
heatmap of the magnitude of the entries.

Returns the average density matrix.
"""
function average_and_plot_density_matrices(density_matrices::Vector{Matrix{ComplexF64}})
    # 1. Input Validation and Setup
    if isempty(density_matrices)
        error("The input vector of density matrices is empty.")
    end
    
    N_samples = length(density_matrices)
    D = size(density_matrices[1], 1) # Dimension of the Hilbert space (2^n)
    
    # 2. Compute the Average Density Matrix (Averaging is linear)
    # Start with a zero matrix of the correct size
    avg_rho = mean(density_matrices)

    # 3. Visualization (Heatmap)
    
    # Density matrices are complex, so we plot the magnitude (|rho_ij|)
    plot_data = abs.(avg_rho) 

    # Create the heatmap
    p = heatmap(
        plot_data,
        aspect_ratio = :equal,
        title = "Average Density Matrix Magnitude (|ρ|)",
        xlabel = "Basis State Index",
        ylabel = "Basis State Index",
        color = :viridis, # A good color scheme for magnitude plots
        xticks = (1:D),
        yticks = (1:D),
        cbar_title = "Magnitude",
        size = (600, 600)
    )
    
    display(p)

    # 4. Return the result for further numerical analysis
    return avg_rho
end

"""
    run_randomness_test(
        n_qubits::Int, 
        n_layers::Int, 
        ent::Entanglement, 
        num_samples::Int; 
        seed::Int=42
    )

Performs the full experiment to test the randomness of a ReuploadingCircuit:
1. Generates the feature map (circuit) with random parameters.
2. Generates random, normalized data vectors.
3. Computes the density matrix for each input.
4. Computes and plots the average density matrix.

Returns a dictionary containing the input parameters and the average density matrix.
"""
function run_randomness_test(
    n_qubits::Int, 
    n_layers::Int, 
    ent::EntanglementBlock, 
    num_samples::Int; 
    seed::Int=42
)
    # 0. Set up environment and parameters
    Random.seed!(seed)
    D_hilbert = 2^n_qubits
    
    # Assuming n_features equals n_qubits for simplicity in this general test
    n_features = n_qubits 
    
    # 1. Generate Feature Map (Circuit)
    println("1. Generating feature map (N=$n_qubits, L=$n_layers)...")
    fm = create_reuploading_circuit_with_random_params(n_qubits, n_features, n_layers, ent; seed=seed)
    
    # 2. Generate Data
    println("2. Generating $num_samples random, normalized data vectors (Dim=$n_features)...")
    X = generate_random_vectors(n_features, num_samples, true)
    
    # 3. Compute Density Matrices
    println("3. Computing density matrices for all samples...")
    # NOTE: This function requires the TQK types to be available in the environment
    density_matrices = compute_density_matrices(fm, X)
    
    # 4. Compute Average and Plot
    println("4. Computing and plotting average density matrix...")
    avg_rho = average_and_plot_density_matrices(density_matrices)
    
    # 5. Calculate Diagnostic Metrics
    
    # Purity (Measure of mixedness/entropy. Purity=1 for pure state, 1/D for maximally mixed)
    purity = tr(avg_rho^2) 
    
    # Max entry (should be near 1/D for random states)
    max_mag = maximum(abs.(avg_rho))

    # 6. Return Results Dictionary
    results = @dict(
        n_qubits,
        n_features,
        n_layers,
        ent,
        num_samples,
        D_hilbert,
        avg_rho,
        purity,
        max_mag
    )
    
    println("\n--- Experiment Complete ---")
    println("Average Purity (Tr(ρ²)): ", round(purity, digits=5))
    println("Target Purity (1/D): ", round(1/D_hilbert, digits=5))
    
    return results
end

# Example Usage to test 2 qubits:
# (Requires the TQK environment and all helper functions to be active)
# 
# results_2q = run_randomness_test(
#     2,               # n_qubits (N=2)
#     3,               # n_layers (L=3, reasonably deep)
#     LinearEntangler, # ent (Type of entanglement)
#     200              # num_samples
# )
# 
# # You can now save the results using DrWatson:
# # @tagsave(datadir("randomness_tests", savename(results_2q, "bson")), results_2q)


"""
    compute_kernel_perturbation(fm::AbstractQuantumFeatureMap, X::AbstractMatrix, delta::Float64=1e-3)

Computes the quantum kernel value between a data point `x` and a perturbed
version `x + delta_x`, where the perturbation `delta_x` is applied only to
the first element of `x`.

The quantum kernel is defined as K(x, x') = |<ψ(x)|ψ(x')>|^2, where |ψ(x)> is
the feature vector (final state) of the circuit.

Returns a vector of kernel values K(x_i, x_i + dx) for all samples.
"""
function compute_kernel_perturbation(fm::AbstractQuantumFeatureMap, X::AbstractMatrix, delta::Float64=1e-3)
    n = n_qubits(fm)
    N_samples = size(X, 2)
    kernel_values = Float64[]

    # The initial state is always |0>^n
    initial_state = zero_state(n)

    for i in 1:N_samples
        x = @view X[:, i]
        
        # 1. Compute the Feature State for the original vector |ψ(x)>
        map_inputs!(fm, x)
        ψ_x = apply!(copy(initial_state), fm.circuit)
        
        # 2. Compute the Perturbed Vector x' = x + dx
        x_perturbed = copy(x)
        # Perturb only the first element (farthest from the origin)
        x_perturbed[1] += delta
        
        # NOTE: Re-normalize the perturbed vector if the original data was normalized,
        # to ensure it lies on the unit sphere for a fair comparison.
        x_perturbed = x_perturbed / norm(x_perturbed)

        # 3. Compute the Feature State for the perturbed vector |ψ(x')>
        map_inputs!(fm, x_perturbed)
        ψ_xp = apply!(copy(initial_state), fm.circuit)

        # 4. Compute the Kernel Value K(x, x') = |<ψ(x)|ψ(x')>|^2
        # The inner product <ψ(x)|ψ(x')> is the overlap
        overlap = ψ_x'ψ_xp
        
        # The kernel value is the squared magnitude of the overlap (Hilbert-Schmidt norm)
        kernel_value = abs2(overlap)
        
        push!(kernel_values, kernel_value)
    end

    return kernel_values
end

"""
    run_sensitivity_test(...)

Performs the full experiment to test the local sensitivity of a ReuploadingCircuit:
1. Generates the feature map (circuit) with random parameters.
2. Generates random, normalized data vectors.
3. Computes the quantum kernel value K(x, x+dx) for each input.
4. Plots the distribution of the resulting kernel values.

Returns a dictionary containing the input parameters and the computed kernel values.
"""
function run_sensitivity_test(
    n_qubits::Int,
    n_layers::Int,
    ent::EntanglementBlock,
    num_samples::Int;
    delta::Float64=1e-3, # Magnitude of the perturbation
    seed::Int=42
)
    # 0. Set up environment and parameters
    Random.seed!(seed)
    n_features = n_qubits # Assuming n_features equals n_qubits
    
    println("1. Generating feature map (N=$n_qubits, L=$n_layers)...")
    fm = create_reuploading_circuit_with_random_params(n_qubits, n_features, n_layers, ent; seed=seed)
    
    println("2. Generating $num_samples random, normalized data vectors (Dim=$n_features)...")
    X = generate_random_vectors(n_features, num_samples, true)
    
    println("3. Computing kernel values K(x, x+dx) for all samples (delta=$delta)...")
    kernel_values = compute_kernel_perturbation(fm, X, delta)
    
    # --- Visualization (Histogram) ---
    println("4. Plotting distribution of kernel values...")
    
    # A smaller mean/median for K(x, x+dx) indicates higher sensitivity/less local robustness.
    median_kernel = median(kernel_values)
    mean_kernel = mean(kernel_values)
    
    p = histogram(
        kernel_values,
        title = "Kernel Sensitivity (K(x, x+dx))",
        xlabel = "Kernel Value K(x, x+dx)",
        ylabel = "Frequency",
        legend = false,
        bins = 20,
        normalize = :probability,
        size = (700, 500)
    )
    
    vline!([median_kernel], line=(:dash, 2), label="Median", color=:red)
    
    display(p) 
    
    # --- Return Results Dictionary ---
    results = @dict(
        n_qubits,
        n_features,
        n_layers,
        ent,
        num_samples,
        delta,
        median_kernel,
        mean_kernel,
        kernel_values
    )
    
    println("\n--- Sensitivity Test Complete ---")
    println("Median Kernel Value: ", round(median_kernel, digits=5))
    println("Mean Kernel Value: ", round(mean_kernel, digits=5))
    
    # Interpretation Note: A value close to 1 indicates the circuit is smooth and locally
    # robust, meaning small changes in x don't change the feature vector much.
    # A value close to 0 indicates high sensitivity and potential "barren plateau" issues
    # or the feature map is highly randomized.
    
    return results
end

# Example Usage to test 2 qubits:
# (Requires the TQK environment and all helper functions to be active)
# 
# sensitivity_results_2q = run_sensitivity_test(
#     2,               # n_qubits (N=2)
#     3,               # n_layers (L=3, reasonably deep)
#     LinearEntangler, # ent (Type of entanglement)
#     200,             # num_samples
#     delta=1e-3       # small perturbation magnitude
# )
# 
# # You can now save the results using DrWatson:
# # @tagsave(datadir("sensitivity_tests", savename(sensitivity_results_2q, "bson")), sensitivity_results_2q)
