using Combinatorics

function grlex_lt(α, β)
    da = sum(α)
    db = sum(β)

    da < db || (da == db && α < β)
end

"""
    monomial_indices(n, r) -> Vector{Vector{Int}}

Generate exponents for multivariate polynomials with `n` 
variables up to a total degree of `r` (excluding the degree 0 constant).

The output is sorted in graded lexicographic (grlex) order. Ordering was
chosen since it's the default in SumOfSquares's `monomials` function.
"""
function monomial_indices(n::Int64, r::Int64)::Vector{Vector{Int}}
    indices = Vector{Vector{Int}}()
    for d in 1:r                              # degree 0 excluded, since Phi_{1:r}
        for α in multiexponents(n, d)     # all multi-indices of total degree d
            push!(indices, collect(α))
        end
    end

    indices = sort(indices, lt=grlex_lt)

    return indices
end