using Distributions

using LinearAlgebra: diag, isdiag

const _MultiIndex = Union{AbstractVector{<:Integer}, Tuple{Vararg{Integer}}}

"""
    moment_matrix(noise, basis_exponents)

Compute `E[Φ(w) Φ(w)']`, where `basis_exponents` contains the exponent
of each monomial in `Φ`. The rows and columns follow the order of
`basis_exponents`.
"""
function moment_matrix(
        noise::UnivariateDistribution,
        basis_exponents::AbstractVector{<:_MultiIndex},
    )
    return moment_matrix([noise], basis_exponents)
end

function moment_matrix(
        noise::AbstractVector{<:UnivariateDistribution},
        basis_exponents::AbstractVector{<:_MultiIndex},
    )
    return [
        prod(_raw_moment(dist, α[k] + β[k]) for (k, dist) in enumerate(noise))
            for α in basis_exponents, β in basis_exponents
    ]
end

function moment_matrix(
        noise::Distributions.Product,
        basis_exponents::AbstractVector{<:_MultiIndex},
    )
    return moment_matrix(noise.v, basis_exponents)
end

function moment_matrix(noise::MvNormal, basis_exponents::AbstractVector{<:_MultiIndex})
    isdiag(noise.Σ) || throw(
        ArgumentError(
            "moment_matrix only supports MvNormal distributions with diagonal covariance",
        )
    )

    variances = diag(noise.Σ)
    return [
        prod(
            _normal_raw_moment(noise.μ[k], variances[k], α[k] + β[k])
                for k in eachindex(noise.μ)
        )
            for α in basis_exponents, β in basis_exponents
    ]
end

function _raw_moment(dist::Normal, k::Integer)
    μ, σ = params(dist)
    return _normal_raw_moment(μ, σ^2, k)
end

function _raw_moment(dist::Gamma, k::Integer)
    α, θ = params(dist)
    moment = one(promote_type(typeof(α), typeof(θ)))

    for j in 1:k
        moment *= θ * (α + (j - 1))
    end

    return moment
end

function _raw_moment(dist::LocationScale, k::Integer)
    location, scale, base = params(dist)
    return sum(
        binomial(k, j) * location^(k - j) * scale^j * _raw_moment(base, j)
            for j in 0:k
    )
end

function _normal_raw_moment(μ, variance, k::Integer)
    T = promote_type(typeof(μ), typeof(variance))
    previous = one(T)
    k == 0 && return previous

    moment = convert(T, μ)
    for j in 2:k
        previous, moment = moment, μ * moment + (j - 1) * variance * previous
    end

    return moment
end
