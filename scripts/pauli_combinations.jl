
using Combinatorics
using Base.Iterators

"""
    generate_special_combinations()

Generates special combinations of strings from the characters 'x', 'y', 'z'
under the following constraints:
1. All generated terms must have 'order >= 2' (length >= 2).
2. The maximum order (length) of any single term must be 3.
3. There can be at most 3 terms in the final combination (list).
"""
function generate_special_combinations()
    # Define the character set
    chars = ["X", "Y", "Z"]

    # --- 1. Generate all valid 'order >= 2' terms with length <= 3 ---

    # Terms of length 2 (order = 2): 3^2 = 9 terms
    terms_len2 = vec([join(p) for p in product(chars, chars)])

    # Terms of length 3 (order = 3): 3^3 = 27 terms
    # terms_len3 = vec([join(p) for p in product(chars, chars, chars)])


    # All valid terms (length 2 or 3)
    # valid_terms = vcat(terms_len2, terms_len3)
    valid_terms = vcat(terms_len2)
    # Note: If the example ["Z", "ZZ"] suggests that single-character
    # terms are allowed *if* a higher-order term is present, the problem
    # is different. Based on "There must be at least one 'order >= 2' term",
    # and the examples, I'll generate all possible terms and filter later.

    # Re-reading the problem statement and examples:
    # "There must be at least one "order >= 2" term: "xx", "xy", "zy", "zx", etc"
    # "The maximum order must be 3"
    # "There can be at most 3 terms in the final combination."

    # The examples ["Z", "ZZ"] and ["XYY"] suggest single-character terms (like "Z")
    # are allowed, as long as at least *one* term has length >= 2.

    # Let's redefine 'valid_terms' to include all terms of length 1, 2, and 3.
    terms_len1 = [string(c) for c in chars]
    # valid_terms_all = vcat(terms_len1, terms_len2, terms_len3)
    valid_terms_all = vcat(terms_len1, terms_len2)

    # All valid high-order terms (length >= 2)
    # high_order_terms = vcat(terms_len2, terms_len3)
    high_order_terms = terms_len2

    # --- 2. Combine terms into final combinations (1, 2, or 3 terms) ---
    combinations = Set{Vector{String}}()

    # Case 1: Combinations with 1 term
    # This term must be high-order (length >= 2)
    for term in high_order_terms
        push!(combinations, [term])
    end

    # Case 2: Combinations with 2 terms
    # At least one of the two terms must be high-order
    for term1 in valid_terms_all
        for term2 in valid_terms_all
            comb = sort!([term1, term2]) # Sort to avoid duplicates like ["X","Y"] and ["Y","X"]
            # Check for the constraint: at least one term is high-order (length >= 2)
            if length(term1) >= 2 || length(term2) >= 2
                push!(combinations, comb)
            end
        end
    end

    # Case 3: Combinations with 3 terms
    # At least one of the three terms must be high-order
    # Using 'combinations' from the Combinatorics package for unique triplets
    # The order of terms in the final list matters for the examples you gave,
    # but to avoid generating the same set of terms multiple times, we'll
    # use combinations and then generate all permutations of the result.
    for comb_tuple in with_replacement_combinations(valid_terms_all, 3)
        comb = collect(comb_tuple)
        # Check for the constraint: at least one term is high-order (length >= 2)
        if any(length(t) >= 2 for t in comb)
            # Generate all unique permutations for this set of terms
            for p in unique(collect(permutations(comb)))
                 push!(combinations, p)
            end
        end
    end

    # The above approach for 3 terms using `unique(collect(permutations(comb)))`
    # is complex. Let's simplify the 2 and 3 term cases by not sorting/permuting
    # and just generating all ordered combinations, which matches your examples.
    # The set will handle duplicates of the final list structure, e.g.,
    # ["XY"] and ["XY"] will only be stored once.

    combinations_ordered = Set{Vector{String}}()
    high_order_terms_set = Set(high_order_terms)

    # Case 1: 1 term (must be high-order)
    for term in high_order_terms
        push!(combinations_ordered, [term])
    end

    # Case 2: 2 terms
    for term1 in valid_terms_all
        for term2 in valid_terms_all
            comb = [term1, term2]
            # Check for the constraint: at least one term is high-order
            if term1 in high_order_terms_set || term2 in high_order_terms_set
                push!(combinations_ordered, comb)
            end
        end
    end

    # Case 3: 3 terms
    for term1 in valid_terms_all
        for term2 in valid_terms_all
            for term3 in valid_terms_all
                comb = [term1, term2, term3]
                # Check for the constraint: at least one term is high-order
                if term1 in high_order_terms_set || term2 in high_order_terms_set || term3 in high_order_terms_set
                    push!(combinations_ordered, comb)
                end
            end
        end
    end

    return collect(combinations_ordered)
end

# Example usage:
# all_combinations = generate_special_combinations()
# println("Total combinations found: ", length(all_combinations))
# println("A few examples:")
# for i in 1:min(10, length(all_combinations))
#     println(all_combinations[i])
# end


function count_combination_patterns(combs::Vector{Vector{String}})
    # A dictionary to store the results for different patterns
    pattern_counts = Dict{String, Int}(
        "1_term_order_2"            => 0,
        "1_term_order_3"            => 0,
        "2_term_max_order_2"        => 0,
        "3_term_max_order_2"        => 0,
        "3_term_max_order_3"        => 0,
    )

    for comb in combs
        num_terms = length(comb)

        # Calculate the maximum length (order) within the current combination
        # The length of an empty list is 0, so we initialize max_order to 0
        max_order = isempty(comb) ? 0 : maximum(length, comb)

        # --------------------------------------------------------------------
        # Pattern 1: 1 term combinations of order 2
        # Example: ["xy"]
        if num_terms == 1 && max_order == 2
            pattern_counts["1_term_order_2"] += 1
        end

        # Pattern 2: 1 term combinations of order 3
        # Example: ["xyz"]
        if num_terms == 1 && max_order == 3
            pattern_counts["1_term_order_3"] += 1
        end

        # --------------------------------------------------------------------
        # Pattern 3: 2 term combinations with max order 2
        # This means BOTH terms must have length <= 2, and at least one must be >= 2
        # The overall constraint ensures at least one term is order >= 2.
        # Example: ["x", "yz"], ["xy", "zz"]
        if num_terms == 2 && max_order == 2
            pattern_counts["2_term_max_order_2"] += 1
        end

        # --------------------------------------------------------------------
        # Pattern 4: 3 term combinations where ALL terms are order 1
        # This is a special case. Note that this pattern *will not exist* # in your 'combs' list because your generation constraints required 
        # at least one order >= 2 term. We include it for completeness.
        # Example: ["x", "y", "z"] (If constraints allowed it)
        if num_terms == 3 && max_order == 2
            pattern_counts["3_term_max_order_2"] += 1
        end

        if num_terms == 3 && max_order == 3
            pattern_counts["3_term_max_order_3"] += 1
        end
    end

    return pattern_counts
end
