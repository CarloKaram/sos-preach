using LinearAlgebra

struct SaturationRegion
    Ξ::Vector{Int}
    D::Diagonal{Float64}
    Ā::Matrix{Float64}
    B̄::Matrix{Float64}
    d::Vector{Float64}
    R::Matrix{Float64}
    c::Vector{Float64}
end

function build_region(A, B, K, Ξ::Vector{Int}, u_min = nothing, u_max = nothing)::SaturationRegion
    m = size(B, 2)
    n = size(A, 1)

    if u_min === nothing
        u_min = -ones(m)
    end
    if u_max === nothing
        u_max = ones(m)
    end

    @assert length(Ξ) == m "Ξ has length $(length(Ξ)), but B has $m columns"
    
    D = Diagonal(ones(Int, m) - abs.(Ξ))
    Ā = A + B * D * K
    B̄ = B * (D - I)
    d = [Ξ[i] == 1 ? u_max[i] : Ξ[i] == -1 ? u_min[i] : 0.0 for i in 1:m]

    R_sat, c_sat = saturation_inequalities(K, Ξ, u_min, u_max)
    
    # Input constraints: v ∈ [u_min, u_max]
    R_input = [zeros(m, n) I; zeros(m, n) -I]
    c_input = [u_max; -u_min]

    R = [R_sat; R_input]
    c = [c_sat; c_input]

    return SaturationRegion(Ξ, D, Ā, B̄, d, R, c)
end

function generate_regions(A, B, K)
    m = size(B, 2)
    patterns = Iterators.product(fill([-1, 0, 1], m)...)
    return [build_region(A, B, K, collect(Ξ)) for Ξ in patterns]
end


function saturation_inequalities(K, Ξ, u_min, u_max)
    n = size(K, 2)
    m = length(Ξ)
    rows_R = []
    rows_c = []
    for i in 1:m
        # build rows for actuator i
        if Ξ[i] == 1                        # -K_i e - v_i <= -u_max 
            row = zeros(1, n + m)
            row[1, 1:n] = -K[i, :]'
            row[1, n+i] = -1.0

            push!(rows_R, row)  
            push!(rows_c, -u_max[i])
        elseif Ξ[i] == 0
            row = zeros(1, n + m)
            row[1, 1:n] = K[i, :]'
            row[1, n+i] = 1.0
            
            push!(rows_R, row)
            push!(rows_c, u_max[i])

            row = zeros(1, n + m)
            row[1, 1:n] = -K[i, :]'
            row[1, n+i] = -1.0
            
            push!(rows_R, row)
            push!(rows_c, -u_min[i])
        else  # Xi[i] == -1                # K_i e + v_i <= u_min
            row = zeros(1, n + m)
            row[1, 1:n] = K[i, :]'
            row[1, n+i] = 1.0

            push!(rows_R, row)
            push!(rows_c, u_min[i])
        end
    end
    return vcat(rows_R...), vcat(rows_c...)
end