using Test
using StableRNGs
using TQK

@testset "Data Consistency & Reproducibility" begin    
    # Define a config
    dc = DataConfig(
        dataset_name="test_consistency",
        n_samples=50,
        master_seed=UInt(42),
        params=RBFDataParams(gamma=0.5),
    )

    # Test 1: Deterministic Generation
    println("Generating Data 1...")
    data1 = produce_data(dc)
    println("Generating Data 2 (Same Seed)...")
    data2 = produce_data(dc)
    
    @test data1["X_train"] == data2["X_train"]
    @test data1["y_test"] == data2["y_test"]
    
    # Test 2: DrWatson Caching
    # Save to a temp dir to test loading
    mktempdir() do tmpdir
        # Mock datadir to point to tmpdir
        # (In practice, just call produce_or_load manually)
        d1, f1 = DrWatson.produce_or_load(produce_data, dc, tmpdir)
        d2, f2 = DrWatson.produce_or_load(produce_data, dc, tmpdir)
        
        @test f1 == f2 # Should be same file
        @test d1["X_train"] == d2["X_train"]
    end
end

@testset "Tuning Batch Consistency" begin
    # Setup
    X = rand(2, 100)
    y = rand([-1, 1], 100)
    
    # Two identical configs
    rng_samp1 = derive_rng(UInt(42), TQK.SALT_DATAGEN)
    rng_samp2 = derive_rng(UInt(42), TQK.SALT_DATAGEN)
    
    # Simulate batch selection
    batch1 = TQK.compute_batched_kta(x->x'*x, X, y, 10, rng_samp1)
    batch2 = TQK.compute_batched_kta(x->x'*x, X, y, 10, rng_samp2)
    
    # Since the RNGs were derived identically and operations are same, results must match
    @test batch1 == batch2
end
