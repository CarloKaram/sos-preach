using Distributions

struct SaturatedSystem{T,D}
    A::Matrix{T}
    B::Matrix{T}
    K::Matrix{T}
    u_min::Vector{T}
    u_max::Vector{T}
    H::Matrix{T}
    h::Vector{T}
    noise::D
end

function SaturatedSystem(A, B, K, H, h, noise; u_min=nothing, u_max=nothing)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("A must be square"))

    size(B, 1) == n || throw(DimensionMismatch("B must have $n rows"))
    m = size(B, 2)
    size(K) == (m, n) || throw(DimensionMismatch("K must have size ($m, $n)"))

    size(H, 2) == n || throw(DimensionMismatch("H must have $n columns"))
    length(h) == size(H, 1) || throw(DimensionMismatch(
        "h must have length $(size(H, 1))",
    ))

    noise_dimension = _noise_dimension(noise)
    noise_dimension == n || throw(DimensionMismatch(
        "noise has dimension $noise_dimension, expected $n",
    ))

    T = promote_type(eltype(A), eltype(B), eltype(K), eltype(H), eltype(h))
    u_min !== nothing && (T = promote_type(T, eltype(u_min)))
    u_max !== nothing && (T = promote_type(T, eltype(u_max)))

    lower = u_min === nothing ? fill(-one(T), m) : Vector{T}(u_min)
    upper = u_max === nothing ? fill(one(T), m) : Vector{T}(u_max)

    length(lower) == m || throw(DimensionMismatch("u_min must have length $m"))
    length(upper) == m || throw(DimensionMismatch("u_max must have length $m"))
    all(lower .< upper) || throw(ArgumentError("u_min must be strictly less than u_max"))

    return SaturatedSystem{T,typeof(noise)}(
        Matrix{T}(A),
        Matrix{T}(B),
        Matrix{T}(K),
        lower,
        upper,
        Matrix{T}(H),
        Vector{T}(h),
        noise,
    )
end

function saturated_error_dynamics(system::SaturatedSystem, e, v)
    command = clamp.(system.K * e + v, system.u_min, system.u_max)
    return system.A * e + system.B * (command - v)
end

_noise_dimension(::UnivariateDistribution) = 1
_noise_dimension(noise::MultivariateDistribution) = length(noise)
_noise_dimension(noise::AbstractVector{<:UnivariateDistribution}) = length(noise)
_noise_dimension(noise) = throw(ArgumentError(
    "noise must be a distribution or a vector of univariate distributions, got $(typeof(noise))",
))
