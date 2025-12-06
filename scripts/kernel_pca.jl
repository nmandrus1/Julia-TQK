using DrWatson
@quickactivate "TQK"

using TQK 
using LinearAlgebra
using Random
using Plots
using Optimization 
using OptimizationOptimisers
using MultivariateStats
using StatsBase
using NPZ
using Dates
using Printf

"""
Run quantum kernel training for KPCA reconstruction.
Tracks reconstruction quality and saves results for sharing.
"""
function run_kpca_reconstruction_experiment(::Type{T};    
    # Quantum circuit parameters
    n_qubits::Int = 4,
    n_layers::Int = 2,
    entanglement::EntanglementBlock = all_to_all,
    
    # KPCA parameters
    kpca_dim::Int = 223,
    alpha::Float64 = 1e-8,
    
    # Training parameters
    opt_iters::Int = 50,
    learning_rate::Float64 = 0.01,
    
    # Reproducibility
    seed::Union{Int, Nothing} = 42,
    
    # Output
    save_results::Bool = true,
    output_name::String = "kpca_reconstruction"
) where {T <: AbstractFloat}
    
    # Set seed
    if isnothing(seed)
        seed = rand(Int)
    end
    Random.seed!(seed)
    
    println("="^60)
    println("Quantum Kernel KPCA Reconstruction Experiment")
    println("="^60)
    println("Seed: $seed")
    println("Circuit: $n_qubits qubits, $n_layers layers")
    println("KPCA dim: $kpca_dim")
    println("Optimizer: AMSGrad (lr=$learning_rate)")
    println("="^60)
    
    # Load/generate data
    println("\nLoading data...")
    X_data = load_satellite_data(T)
    
    println("Data shape: $(size(X_data))")
    
    n_input_features = size(X_data, 2)

    # Initialize quantum circuit
    println("\nInitializing quantum circuit...")
    feature_map = ReuploadingCircuit(n_qubits, n_input_features, n_layers, entanglement)
    assign_random_params!(feature_map, seed=seed)
    kernel = FidelityKernel(feature_map)
    
    n_params_total = 2 * n_params(feature_map)
    println("Total parameters: $n_params_total ($(n_params(feature_map)) weights + $(n_params(feature_map)) biases)")
    
    # Initial kernel and reconstruction
    println("\nComputing initial reconstruction...")
    K_initial = TQK.evaluate(kernel, X_data)
    X_initial_recon = direct_kpca_reconstruct(K_initial, X_data; kpca_dim=kpca_dim, α=alpha)
    initial_mse = mean((X_data - X_initial_recon).^2)
    initial_rmse = sqrt(initial_mse)
    
    println("Initial RMSE: $(round(initial_rmse, digits=6))")
    
    # Set up loss function
    loss_fn = K -> kernel_pca_reconstruction_loss(K, X_data; kpca_dim=kpca_dim, α=alpha)
    
    # Create trainer
    trainer = QuantumKernelTrainer(
        kernel,
        loss_fn,
        X_data,
        Vector{T}(undef, 0),
    )
    
    # Track metrics
    metrics = Dict(
        "iteration" => Int[],
        "loss" => Float64[],
        "rmse" => Float64[],
        "rel_improvement" => Float64[],
    )
    
    function training_callback(state, loss)
        iter = state.iter
        
        # Compute reconstruction
        K = trainer.K_cache
        X_recon = direct_kpca_reconstruct(K, X_data; kpca_dim=kpca_dim, α=alpha)
        rmse = sqrt(mean((X_data- X_recon).^2))
        rel_improvement = 100 * (1 - rmse/initial_rmse)
        
        push!(metrics["iteration"], iter)
        push!(metrics["loss"], loss)
        push!(metrics["rmse"], rmse)
        push!(metrics["rel_improvement"], rel_improvement)
        
        if iter % max(1, opt_iters ÷ 10) == 0 || iter == 1
            @printf "Iter %3d: loss=%.6f, RMSE=%.6f (%.1f%% improvement)\n" iter loss rmse rel_improvement
        end
    end
    
    # Train
    println("\nStarting training...")
    println("─"^50)
    
    optimizer = OptimizationOptimisers.AMSGrad(eta=learning_rate)
    sol = train!(trainer,
                 optimizer=optimizer,
                 iterations=opt_iters,
                 callback=training_callback)
    
    println("─"^50)
    println("Training completed!")
    
    # Final reconstruction
    println("\nComputing final reconstruction...")
    K_final = TQK.evaluate(kernel, X_data)
    X_final_recon = direct_kpca_reconstruct(K_final, X_data; kpca_dim=kpca_dim, α=alpha)
    final_mse = mean((X_data - X_final_recon).^2)
    final_rmse = sqrt(final_mse)
    improvement = 100 * (1 - final_rmse/initial_rmse)
    
    println("\nReconstruction Summary:")
    println("  Initial RMSE: $(round(initial_rmse, digits=6))")
    println("  Final RMSE:   $(round(final_rmse, digits=6))")
    println("  Improvement:  $(round(improvement, digits=2))%")
    
    # Create plots
    p_loss = plot(metrics["iteration"], metrics["loss"],
                  xlabel="Iteration", ylabel="Loss",
                  title="Training Loss", label="Loss",
                  lw=2, marker=:circle, markersize=2)
    
    p_rmse = plot(metrics["iteration"], metrics["rmse"],
                  xlabel="Iteration", ylabel="RMSE",
                  title="Reconstruction RMSE", label="RMSE",
                  lw=2, marker=:circle, markersize=2, color=:red)
    
    p_kernels = plot(
        heatmap(K_initial, title="Initial Kernel", c=:viridis, clim=(0,1)),
        heatmap(K_final, title="Trained Kernel", c=:viridis, clim=(0,1)),
        layout=(1,2), size=(800, 300)
    )
    
    # Package results
    results = Dict(
        :params => Dict(
            :type => T,
            :seed => seed,
            :n_qubits => n_qubits,
            :n_layers => n_layers,
            :n_features => n_input_features,
            :kpca_dim => kpca_dim,
            :alpha => alpha,
            :opt_iters => opt_iters,
            :learning_rate => learning_rate,
            :data_type => data_type,
            :timestamp => now(),
        ),
        :trained_kernel => kernel,
        :metrics => metrics,
        :X_original => X_data,
        :X_initial_recon => X_initial_recon,
        :X_final_recon => X_final_recon,
        :K_initial => K_initial,
        :K_final => K_final,
        :initial_rmse => initial_rmse,
        :final_rmse => final_rmse,
        :improvement_percent => improvement,
        :plots => Dict(
            :loss => p_loss,
            :rmse => p_rmse,
            :kernels => p_kernels,
        ),
        :scaler => scaler,
    )
    
    # Save results
    if save_results
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        filename = "$(output_name)_$(timestamp)"
        
        # Save to DrWatson structure
        saved_path = safesave(datadir("kernel-pca", filename * ".jld2"), results)
        println("\nResults saved to: $saved_path")
        
        # Export reconstructed matrix for sharing
        export_path = datadir("kernel-pca", filename * "_reconstructed.npy")
        npzwrite(export_path, results[:X_final_recon])
        println("Reconstructed matrix exported to: $export_path")
        
        # Save plots
        savefig(p_loss, datadir("kernel-pca", filename * "_loss.png"))
        savefig(p_rmse, datadir("kernel-pca", filename * "_rmse.png"))
        savefig(p_kernels, datadir("kernel-pca", filename * "_kernels.png"))
        println("Plots saved.")
    end
    
    return results
end

"""
Load satellite data (from document 1)
"""
function load_satellite_data(::Type{T}) where {T<:AbstractFloat}
    data_path = datadir("kernel-pca", "prediction_6_matrix_224x4032.npy")
    X = npzread(data_path)'
    

    # Preprocessing
    println("Preprocessing...")
    scaler = StatsBase.fit(ZScoreTransform, X)
    X_scaled = StatsBase.transform(scaler, X)
    
    # Optional PCA pre-reduction
    pca_model = fit(PCA , X_scaled; maxoutdim=224, pratio=1)

    X_data = predict(pca_model, X_scaled)
    X_data = convert(Matrix{T}, X_data)
    return permutedims(X_data)
end

"""
Direct KPCA reconstruction (from document 1)
"""
function direct_kpca_reconstruct(
    K::AbstractMatrix, 
    X_input::AbstractMatrix; 
    kpca_dim::Int, 
    α::Float64=1e-10
)
    n = size(K, 1)
    J = ones(n, n)

    # Center the kernel matrix
    K_centered = K - (1/n)*K*J - (1/n)*J*K + (1/n^2)*J*K*J
    
    # Ensure symmetry and add regularization
    K_clean = Symmetric(K_centered) + α * I
    
    # Get eigenvectors
    λ, V = eigen(K_clean)
    
    # Sort by eigenvalue (descending)
    idx = sortperm(λ, rev=true)[1:kpca_dim]
    V_sorted = V[:, idx]
    
    # Reconstruct
    temp = V_sorted * X_input' 
    X_reconstructed = temp * V_sorted
    
    return X_reconstructed
end

function kernel_pca_reconstruction_loss(K::AbstractMatrix, X::AbstractMatrix; kpca_dim::Int, α::Float64)
    X_reconstructed = direct_kpca_reconstruct(K, X; kpca_dim=kpca_dim, α=α)
    return mean((X-X_reconstructed).^2)
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    results = run_kpca_reconstruction_experiment(
        Float32;
        n_samples=200,
        n_features=10,
        data_type=:rbf,
        n_qubits=4,
        n_layers=3,
        kpca_dim=5,
        opt_iters=100,
        learning_rate=0.01,
        seed=42,
        save_results=true,
        output_name="test_kpca"
    )
    
    println("\nExperiment complete!")
    println("Access results: results[:X_final_recon] for reconstructed matrix")
end
