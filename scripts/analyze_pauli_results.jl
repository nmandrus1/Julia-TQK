using DrWatson
@quickactivate "TQK"

using Yao, YaoBlocks, YaoAPI, LinearAlgebra, JLD2, Distances, Statistics

include("analysis.jl")

# @const_gate SX = 1/2 * ComplexF64[1+im 1-im; 1-im 1+im]
# @const_gate SXD = 1/2 * (ComplexF64[1+im 1-im; 1-im 1+im])'

function analyze_pauli_results()
    exps = exp_list()
    for expr in eachrow(exps)
        println(expr[:data_path])
        data = load(expr[:data_path])

        X_train = permutedims(data["X_train"])
        y_train = Vector(data["y_train"])

        # get pauli parameters
        all_results_for_exp = find_experiments(experiment_id=expr[:experiment_id])
        for kernel_name in ["pauli", "reup", "rbf"]
            kernel_results = filter(row -> contains(row.kernel_id, kernel_name), all_results_for_exp)
            sort!(kernel_results, :best_test_acc, rev=true)

            best_kernel = first(kernel_results, 1)
            kernel = best_kernel.kernel_id[1]

            hyperparams = load_hyperparams(expr[:experiment_id], kernel)
            if kernel_name == "pauli"
                display(hyperparams)
                println()
            end

            # Compute kernel
            if hyperparams isa RBFHyperparameters
                display(hyperparams)

                # 1. Calculate pairwise squared Euclidean distances.
                #    NOTE: Distances.jl expects features in columns, so we transpose (').
                D_train_sq = pairwise(SqEuclidean(), X_train', dims=2)
                K_train = exp.(-hyperparams.gamma .* D_train_sq)
        
            elseif hyperparams isa PauliKernelHyperparameters
                K_train = compute_pauli_kernel_matrix(hyperparams, X_train)
        
            elseif hyperparams isa ReuploadingKernelHyperparameters
                ent_map = Dict("linear"=>linear, "alternating"=>alternating, "all_to_all"=>all_to_all)
        
                # Reconstruct feature map with trained params
                nqubits = hyperparams.nqubits
                nlayers = hyperparams.nlayers
                feature_map = ReuploadingCircuit(nqubits, size(X_train, 2), nlayers , linear)
                assign_params!(feature_map, hyperparams.thetas, hyperparams.biases)
        
                kernel = FidelityKernel(feature_map)
                K_train = TQK.evaluate(kernel, X_train)
            end

            println("Rank of K_train (", kernel_name, "): ", rank(K_train))
            println("Var of K_train (", kernel_name, "): ", var(K_train))
            println()
        end
        
    end
end

function pauli_Y_YY_X_kernel(x1, x2)
    return chain(2,
        put(1 => H),
        put(2 => H),
        put(1 => SXGate),
        put(2 => SXGate),
        put(1 => Rx(2*x1)),
        put(2 => Rx(2*x2)),
        control(1, 2=>X),
        put(2=>Rx(2*(pi-x1)*(pi-x2))),
        control(1, 2=>X),
        put(1 => Daggered(SXGate)),
        put(2 => Daggered(SXGate)),
        put(1 => H),
        put(2 => H),
        put(1 => Rx(2*x1)),
        put(2 => Rx(2*x2)),
        put(1 => H),
        put(2 => H),
    )
end
