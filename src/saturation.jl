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

function build_region(A, B, K, Ξ::Vector{Int}, u_min = nothing, u_max = nothing)::SaturationRegion
    m = size(B, 2)
    n = size(A, 1)

    T = promote_type(eltype(A), eltype(B), eltype(K))
    if u_min !== nothing
        T = promote_type(T, eltype(u_min))
    end
    if u_max !== nothing
        T = promote_type(T, eltype(u_max))
    end

    if u_min === nothing
        u_min = fill(-one(T), m)
    else
        u_min = Vector{T}(u_min)
    end
    if u_max === nothing
        u_max = fill(one(T), m)
    else
        u_max = Vector{T}(u_max)
    end

    length(Ξ) == m || throw(
        DimensionMismatch(
            "Ξ has length $(length(Ξ)), but B has m = $m columns! Input dimension disagreement!"
        )
    )

    A = Matrix{T}(A)
    B = Matrix{T}(B)
    K = Matrix{T}(K)
    D = Diagonal(one(T) .- T.(abs.(Ξ)))
    Ā = A + B * D * K
    B̄ = B * (D - Matrix{T}(I, m, m))
    d = [Ξ[i] == 1 ? u_max[i] : Ξ[i] == -1 ? u_min[i] : zero(T) for i in 1:m]

    d̄ = B * d

    R_sat, c_sat = saturation_inequalities(K, Ξ, u_min, u_max)

    # Input constraints: v ∈ [u_min, u_max]
    I_m = Matrix{T}(I, m, m)
    R_input = [zeros(T, m, n) I_m; zeros(T, m, n) -I_m]
    c_input = [u_max; -u_min]

    R = [R_sat; R_input]
    c = [c_sat; c_input]

    return SaturationRegion(Ξ, D, Ā, B̄, d, d̄, R, c)
end

function generate_regions(A, B, K, u_min = nothing, u_max = nothing)
    m = size(B, 2)
    patterns = sort!(vec(collect(Iterators.product(fill([-1, 0, 1], m)...))))
    return [build_region(A, B, K, collect(Ξ), u_min, u_max) for Ξ in patterns]
end


function saturation_inequalities(K, Ξ, u_min, u_max)
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
