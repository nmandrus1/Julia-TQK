using Yao
using YaoAPI
using Zygote
using ChainRulesCore
using Test

# --- MOCK HELPERS ---

# 1. A standard rotation (Differentiable)
function make_rot_layer(n::Int, theta::Real)
    return put(n, 1=>Rz(theta))
end

# 2. A fixed layer (Simulating your Basis Change)
# We test two ways of making this "constant"
function make_fixed_layer_raw(n::Int)
    return chain(n, put(n, i=>H) for i in 1:n)
end

function make_fixed_layer_ignored(n::Int)
    # This simulates what we tried in the main code
    return Zygote.ignore() do
        chain(n, put(n, i=>H) for i in 1:n)
    end
end

@testset "Granular AD Debug V2 (Functional)" begin

    # --- TEST 1: Baseline ---
    println("\n--- 1. Testing Rotation Only (Baseline) ---")
    function test_rot(theta)
        n = 2
        c = make_rot_layer(n, theta)
        reg = apply(zero_state(n), c)
        return real(expect(put(n, 1=>Z), reg))
    end

    try
        g = gradient(test_rot, 0.5)
        println("PASS: Rot gradient: ", g)
    catch e
        println("FAIL: Rot.")
        showerror(stdout, e, catch_backtrace())
    end

    # --- TEST 2: Zygote.ignore on its own ---
    println("\n--- 2. Testing Zygote.ignore block isolated ---")
    function test_ignore(theta)
        n = 2
        # This block is constant, but we return a value derived from it to check flow
        c = make_fixed_layer_ignored(n)
        reg = apply(zero_state(n), c)
        # We multiply by theta so there is a gradient to track
        return real(expect(put(n, 1=>Z), reg)) * theta
    end

    try
        g = gradient(test_ignore, 0.5)
        println("PASS: Ignore gradient: ", g)
    catch e
        println("FAIL: Ignore.")
        showerror(stdout, e, catch_backtrace())
    end

    # --- TEST 3: The Sandwich (Raw construction) ---
    println("\n--- 3. Sandwich with Raw Construction (No ignore) ---")
    # Does Zygote fail simply because the block is complex, even without ignore?
    function test_sandwich_raw(theta)
        n = 2
        # Recreating the blocks every time, allowing Zygote to trace construction
        # (This usually works but is slow/allocating)
        left = make_fixed_layer_raw(n)
        right = make_fixed_layer_raw(n)
        mid = make_rot_layer(n, theta)
        
        c = chain(left, mid, right)
        reg = apply(zero_state(n), c)
        return real(expect(put(n, 1=>Z), reg))
    end

    try
        g = gradient(test_sandwich_raw, 0.5)
        println("PASS: Raw Sandwich: ", g)
    catch e
        println("FAIL: Raw Sandwich.")
        showerror(stdout, e, catch_backtrace())
    end

    # --- TEST 4: The Sandwich (With Zygote.ignore) ---
    println("\n--- 4. Sandwich with Zygote.ignore (The likely crash) ---")
    function test_sandwich_ignore(theta)
        n = 2
        # This matches your current Pauli implementation
        left = make_fixed_layer_ignored(n)
        right = make_fixed_layer_ignored(n)
        mid = make_rot_layer(n, theta)
        
        c = chain(left, mid, right)
        reg = apply(zero_state(n), c)
        return real(expect(put(n, 1=>Z), reg))
    end

    try
        g = gradient(test_sandwich_ignore, 0.5)
        println("PASS: Ignored Sandwich: ", g)
    catch e
        println("FAIL: Ignored Sandwich.")
        # This is where we expect "tuple must be non-empty"
        showerror(stdout, e, catch_backtrace())
    end
end
