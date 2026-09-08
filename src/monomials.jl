using Combinatorics

function grlex_lt(α, β)
    da = sum(α)
    db = sum(β)

    return da < db || (da == db && α < β)
end

"""
    monomial_exponents(n, r, include_constant=false)

Generate exponent tuples for multivariate polynomials with `n` variables up to
a total degree of `r`. The degree 0 constant is included when
`include_constant` is `true`.

The output is sorted in graded lexicographic (grlex) order. Ordering was
chosen since it matches DynamicPolynomials' default `monomials` order.
"""
function monomial_exponents(n::Integer, r::Integer, include_constant::Bool = false)
    if n < 0 || r < 0
        throw(ArgumentError("n and r must be non-negative, got n = $n, and r = $r"))
    end

    start = include_constant ? 0 : 1

    exponents = NTuple{n, Int}[]
    for d in start:r
        for α in multiexponents(n, d)
            push!(exponents, Tuple(α))
        end
    end

    sort!(exponents, lt = grlex_lt)

    return exponents
end

"""
    exponent_index(exponents)

Map each immutable exponent tuple to its position in `exponents`.
"""
function exponent_index(exponents)
    return Dict(Tuple(α) => i for (i, α) in enumerate(exponents))
end
