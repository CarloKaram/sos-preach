using LinearAlgebra: tr

"""
    drift_polynomial(Q, λ, β, region, e, v, M_w, basis_exponents)

Construct the regional drift polynomial
`λ * V(e) + β - E[V(Āe + B̄v + d̄ + w)]`, where `Q` represents
`V` in the nonconstant monomial basis described by `basis_exponents`.

`M_w` must use the noise monomial basis including constant monomial.
"""
function drift_polynomial(Q, λ, β, region, e, v, M_w, basis_exponents)
    f = region.Ā * e + region.B̄ * v + region.d̄
    Φ = [prod(e[k]^α[k] for k in eachindex(e)) for α in basis_exponents]
    T = _translation_matrix(f, basis_exponents)

    V = sum(Q[i, j] * Φ[i] * Φ[j] for i in eachindex(Φ), j in eachindex(Φ))
    expected_successor = tr(Q * T * M_w * transpose(T))

    return λ * V + β - expected_successor
end

function _translation_matrix(f, basis_exponents)
    # Φ(w) needs constant monomial, appending it (equiv. appending 0 exponents)
    zero_exponent = ntuple(_ -> 0, length(f))
    noise_exponents = [zero_exponent; basis_exponents]

    return [
        all(ν[k] <= α[k] for k in eachindex(f)) ?
        prod(binomial(α[k], ν[k]) * f[k]^(α[k] - ν[k]) for k in eachindex(f)) :
        zero(f[1])
        for α in basis_exponents, ν in noise_exponents
    ]
end
