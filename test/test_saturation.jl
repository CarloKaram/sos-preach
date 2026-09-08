using Test
include("../src/saturation.jl")

# Test system
A = [0.8 0.1; 0 0.7]
B = [0; 1;;]
K = [0.5 -0.3]

A_2 = [0.9 0.1; 0.0 0.8]
B_2 = [0.5 0.0; 0.3 0.7]
K_2 = [0.2 -0.1; -0.4 0.3]

function saturated_dynamics(A, B, K, e, v, u_min, u_max)
    return A * e + B * (clamp.(K * e + v, u_min, u_max) - v)
end


@testset "build_region" begin
    up_sat = build_region(A, B, K, [1])
    low_sat = build_region(A, B, K, [-1])
    lin_reg = build_region(A, B, K, [0])

    @testset "Correct matrices for each Ξ" begin
        @test up_sat.D == Diagonal([0])
        @test up_sat.Ā ≈ A
        @test up_sat.B̄ ≈ -B
        @test up_sat.d ≈ [1.0]
        @test up_sat.d̄ ≈ vec(B)

        @test lin_reg.D == Diagonal([1])
        @test lin_reg.Ā ≈ A + B * K
        @test lin_reg.B̄ ≈ zeros(2, 1)
        @test lin_reg.d ≈ [0.0]
        @test lin_reg.d̄ ≈ zeros(2)

        @test low_sat.D == Diagonal([0])
        @test low_sat.Ā ≈ A
        @test low_sat.B̄ ≈ -B
        @test low_sat.d ≈ [-1.0]
        @test low_sat.d̄ ≈ -vec(B)
    end

    @testset "Point-inclusion for each region" begin
        v = [0.5]
        e = [1000.0, 1000.0]
        @test all(up_sat.R * [e; v] .≤ up_sat.c)
        @test !all(low_sat.R * [e; v] .≤ low_sat.c)
        @test !all(lin_reg.R * [e; v] .≤ lin_reg.c)


        e = [0.0, 0.0]
        @test !all(up_sat.R * [e; v] .≤ up_sat.c)
        @test !all(low_sat.R * [e; v] .≤ low_sat.c)
        @test all(lin_reg.R * [e; v] .≤ lin_reg.c)

        e = [-10000.0, -1.0]
        @test !all(up_sat.R * [e; v] .≤ up_sat.c)
        @test all(low_sat.R * [e; v] .≤ low_sat.c)
        @test !all(lin_reg.R * [e; v] .≤ lin_reg.c)

        v = [1.5]
        @test !all(up_sat.R * [e; v] .≤ up_sat.c)
        @test !all(low_sat.R * [e; v] .≤ low_sat.c)
        @test !all(lin_reg.R * [e; v] .≤ lin_reg.c)
    end

    @testset "Row counts" begin
        @test size(up_sat.R, 1) == 3
        @test size(low_sat.R, 1) == 3
        @test size(lin_reg.R, 1) == 4
    end
end

@testset "generate_regions" begin
    regions_m1 = generate_regions(A, B, K)
    regions_m2 = generate_regions(A_2, B_2, K_2)

    @testset "Region count" begin
        @test regions_m1 isa Vector{SaturationRegion{Float64}}
        @test regions_m2 isa Vector{SaturationRegion{Float64}}
        @test length(regions_m1) == 3
        @test length(regions_m2) == 9
    end

    @testset "Region ordering" begin
        expected_m1 = [[ξ] for ξ in -1:1]
        expected_m2 = [[ξ₁, ξ₂] for ξ₁ in -1:1 for ξ₂ in -1:1]

        @test [region.Ξ for region in regions_m1] == expected_m1
        @test [region.Ξ for region in regions_m2] == expected_m2
    end

    @testset "Asymmetric bounds and affine dynamics" begin
        u_min = [-2.0, -3.0]
        u_max = [1.0, 4.0]
        regions = generate_regions(A_2, B_2, K_2, u_min, u_max)
        v = zeros(2)

        for region in regions
            command = [
                ξ == -1 ? u_min[i] - 1 : ξ == 1 ? u_max[i] + 1 : (u_min[i] + u_max[i]) / 2
                    for (i, ξ) in enumerate(region.Ξ)
            ]
            e = K_2 \ command

            @test all(region.R * [e; v] .≤ region.c)
            @test region.d̄ ≈ B_2 * region.d
            @test region.Ā * e + region.B̄ * v + region.d̄ ≈
                saturated_dynamics(A_2, B_2, K_2, e, v, u_min, u_max)
        end
    end

    @testset "Coefficient types" begin
        A_exact = [9 // 10 1 // 10; 0 // 1 4 // 5]
        B_exact = [1 // 2 0 // 1; 3 // 10 7 // 10]
        K_exact = [1 // 5 -1 // 10; -2 // 5 3 // 10]
        u_min_exact = [-2 // 1, -3 // 1]
        u_max_exact = [1 // 1, 4 // 1]

        for T in (Rational{Int}, BigFloat)
            regions = generate_regions(
                T.(A_exact), T.(B_exact), T.(K_exact),
                T.(u_min_exact), T.(u_max_exact),
            )

            @test regions isa Vector{SaturationRegion{T}}
            @test all(region -> region.d̄ == T.(B_exact) * region.d, regions)
        end
    end

end
