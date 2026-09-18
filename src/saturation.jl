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

    # Input constraints: v ∈ [u_min, u_max]
    I_m = Matrix{T}(I, m, m)
    R_input = [zeros(T, m, n) I_m; zeros(T, m, n) -I_m]
    c_input = [system.u_max; -system.u_min]

    R = [R_sat; R_input]
    c = [c_sat; c_input]

    return SaturationRegion(Ξ, D, Ā, B̄, d, d̄, R, c)
end

function generate_regions(system::SaturatedSystem; prune_symmetric=false)
    m = size(system.B, 2)
    patterns = sort!(vec(collect(Iterators.product(fill([-1, 0, 1], m)...))))
    if prune_symmetric && system.u_min == -system.u_max
        filter!(Ξ -> Ξ <= map(-, Ξ), patterns)
    end
    return [build_region(system, collect(Ξ)) for Ξ in patterns]
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
