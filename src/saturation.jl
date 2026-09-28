using LinearAlgebra

struct SaturationRegion{T}
    Ξ::Vector{Int}
    D::Diagonal{T, Vector{T}}
    Ā::Matrix{T}
    B̄::Matrix{T}
    d::Vector{T}
    d̄::Vector{T}
    R::Matrix{T}
    c::Vector{T}
end

function build_region(system::SaturatedSystem{T}, Ξ::Vector{Int})::SaturationRegion where {T}
    m = size(system.B, 2)
    n = size(system.A, 1)

    # might be might be unnecessary
    length(Ξ) == m || throw(
        DimensionMismatch(
            "Ξ has length $(length(Ξ)), but B has m = $m columns! Input dimension disagreement!"
        )
    )

    D = Diagonal(one(T) .- T.(abs.(Ξ)))
    Ā = system.A + system.B * D * system.K
    B̄ = system.B * (D - Matrix{T}(I, m, m))
    d = [Ξ[i] == 1 ? system.u_max[i] : Ξ[i] == -1 ? system.u_min[i] : zero(T) for i in 1:m]

    d̄ = system.B * d

    R_sat, c_sat = saturation_inequalities(system.K, system.u_min, system.u_max, Ξ)

    # Nominal input constraints: v ∈ [v_min, v_max]
    I_m = Matrix{T}(I, m, m)
    R_input = [zeros(T, m, n) I_m; zeros(T, m, n) -I_m]
    c_input = [system.v_max; -system.v_min]

    R = [R_sat; R_input]
    c = [c_sat; c_input]

    return SaturationRegion(Ξ, D, Ā, B̄, d, d̄, R, c)
end

function generate_regions(system::SaturatedSystem; prune_symmetric=false)
    m = size(system.B, 2)
    patterns = sort!(vec(collect(Iterators.product(fill([-1, 0, 1], m)...))))
    if prune_symmetric && system.u_min == -system.u_max &&
            system.v_min == -system.v_max
        filter!(Ξ -> Ξ <= map(-, Ξ), patterns)
    end
    return [build_region(system, collect(Ξ)) for Ξ in patterns]
end

"""
    semialgebraic_saturation_generators(system, e, s; prune_symmetric=false)

Construct normalized polynomial generators for the exact semialgebraic saturation
domain. The general generators are ordered as the `m` interval products, the
`m` lower-end sector constraints, and the `m` upper-end sector constraints.

With `prune_symmetric=true`, construct the conservative full-actuator-domain
generators in the existing order: upper bounds, lower bounds, and sector
constraints.
"""
function semialgebraic_saturation_generators(
        system::SaturatedSystem,
        e,
        s;
        prune_symmetric=false,
    )
    n = size(system.A, 1)
    m = size(system.B, 2)
    length(e) == n || throw(DimensionMismatch("e must have length $n"))
    length(s) == m || throw(DimensionMismatch("s must have length $m"))

    T = eltype(system.K)
    sector_scales = [
        max(one(T), maximum(abs, view(system.K, i, :))) for i in 1:m
    ]
    Ke = [sum(system.K[i, k] * e[k] for k in 1:n) for i in 1:m]

    if prune_symmetric
        widths = system.u_max - system.u_min
        linear_scales = max.(one(T), widths)
        upper = [(widths[i] - s[i]) / linear_scales[i] for i in 1:m]
        lower = [(widths[i] + s[i]) / linear_scales[i] for i in 1:m]
        sector = [
            s[i] * (Ke[i] - s[i]) / sector_scales[i] for i in 1:m
        ]
        return [upper; lower; sector]
    end

    outer_lower = system.u_min - system.v_max
    outer_upper = system.u_max - system.v_min
    inner_lower = system.u_min - system.v_min
    inner_upper = system.u_max - system.v_max
    lower_scales = max.(one(T), abs.(outer_lower))
    upper_scales = max.(one(T), abs.(outer_upper))
    inner_lower_scales = max.(one(T), abs.(inner_lower))
    inner_upper_scales = max.(one(T), abs.(inner_upper))

    interval = [
        (s[i] - outer_lower[i]) * (outer_upper[i] - s[i]) /
        (lower_scales[i] * upper_scales[i]) for i in 1:m
    ]
    lower_sector = [
        (Ke[i] - s[i]) * (s[i] - inner_lower[i]) /
        (sector_scales[i] * inner_lower_scales[i]) for i in 1:m
    ]
    upper_sector = [
        (s[i] - Ke[i]) * (inner_upper[i] - s[i]) /
        (sector_scales[i] * inner_upper_scales[i]) for i in 1:m
    ]

    return [interval; lower_sector; upper_sector]
end

function saturation_inequalities(K, u_min, u_max, Ξ)
    n = size(K, 2)
    m = length(Ξ)
    T = promote_type(eltype(K), eltype(u_min), eltype(u_max))
    K = Matrix{T}(K)
    rows_R = Matrix{T}[]
    rows_c = T[]
    for i in 1:m
        # build rows for actuator i
        if Ξ[i] == 1                        # -K_i e - v_i <= -u_max
            row = zeros(T, 1, n + m)
            row[1, 1:n] = -K[i, :]'
            row[1, n + i] = -one(T)

            push!(rows_R, row)
            push!(rows_c, -u_max[i])
        elseif Ξ[i] == 0
            row = zeros(T, 1, n + m)
            row[1, 1:n] = K[i, :]'
            row[1, n + i] = one(T)

            push!(rows_R, row)
            push!(rows_c, u_max[i])

            row = zeros(T, 1, n + m)
            row[1, 1:n] = -K[i, :]'
            row[1, n + i] = -one(T)

            push!(rows_R, row)
            push!(rows_c, -u_min[i])
        else  # Xi[i] == -1                # K_i e + v_i <= u_min
            row = zeros(T, 1, n + m)
            row[1, 1:n] = K[i, :]'
            row[1, n + i] = one(T)

            push!(rows_R, row)
            push!(rows_c, u_min[i])
        end
    end
    return vcat(rows_R...), vcat(rows_c...)
end
