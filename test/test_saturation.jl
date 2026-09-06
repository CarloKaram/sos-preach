using Test
include("../src/saturation.jl")

# Test system
A = [0.8 0.1; 0 0.7]
B = [0; 1;;]
K = [0.5 -0.3]

A_2 = [0.9 0.1; 0.0 0.8]
B_2 = [0.5 0.0; 0.3 0.7]
K_2 = [0.2 -0.1; -0.4 0.3]


@testset "build_region" begin
    up_sat = build_region(A, B, K, [1])
    low_sat = build_region(A, B, K, [-1])
    lin_reg = build_region(A, B, K, [0])
    
    @testset "Correct matrices for each Ξ" begin
        @test up_sat.D == Diagonal([0])
        @test up_sat.Ā ≈ A
        @test up_sat.B̄ ≈ -B
        @test up_sat.d ≈ [1.0]

        @test lin_reg.D == Diagonal([1])
        @test lin_reg.Ā ≈ A + B*K
        @test lin_reg.B̄ ≈ zeros(2, 1)
        @test lin_reg.d ≈ [0.0]

        @test low_sat.D == Diagonal([0])
        @test low_sat.Ā ≈ A
        @test low_sat.B̄ ≈ -B
        @test low_sat.d ≈ [-1.0]
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
        @test length(regions_m1) == 3
        @test length(regions_m2) == 9
    end

    @testset "Region ordering" begin
        @test regions_m1[1].Ξ == [-1]
        @test regions_m1[2].Ξ == [0]
        @test regions_m1[3].Ξ == [1]

        @test regions_m2[1].Ξ == [-1; -1]
        @test regions_m2[2].Ξ == [-1; 0]
        # @test regions_m2[3].Ξ == [1]
    end

end