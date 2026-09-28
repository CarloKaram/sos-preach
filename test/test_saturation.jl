using Distributions
using DynamicPolynomials
using Test
include("../src/system.jl")
include("../src/saturation.jl")

# Test system
A = [0.8 0.1; 0 0.7]
B = [0; 1;;]
K = [0.5 -0.3]

A_2 = [0.9 0.1; 0.0 0.8]
B_2 = [0.5 0.0; 0.3 0.7]
K_2 = [0.2 -0.1; -0.4 0.3]

system_m1 = SaturatedSystem(
    A, B, K, zeros(0, 2), Float64[], Normal.(zeros(2), ones(2)),
)
system_m2 = SaturatedSystem(
    A_2, B_2, K_2, zeros(0, 2), Float64[], Normal.(zeros(2), ones(2)),
)

@testset "build_region" begin
    up_sat = build_region(system_m1, [1])
    low_sat = build_region(system_m1, [-1])
    lin_reg = build_region(system_m1, [0])

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

    @testset "Region membership" begin
        regions = (up_sat, low_sat, lin_reg)
        cases = (
            ([1000.0, 1000.0], [0.5], (true, false, false)),
            ([0.0, 0.0], [0.5], (false, false, true)),
            ([-10000.0, -1.0], [0.5], (false, true, false)),
            ([-10000.0, -1.0], [1.5], (false, false, false)),
        )
        for (e, v, expected) in cases
            @test Tuple(all(region.R * [e; v] .≤ region.c) for region in regions) ==
                  expected
        end
    end

    @testset "Separate nominal input bounds" begin
        system = SaturatedSystem(
            A, B, K, zeros(0, 2), Float64[], Normal.(zeros(2), ones(2));
            u_min=[-2.0], u_max=[3.0], v_min=[-0.5], v_max=[0.75],
        )
        up_sat = build_region(system, [1])

        @test up_sat.c[1] == -3.0
        @test up_sat.c[end-1:end] == [0.75, 0.5]
    end
end

@testset "generate_regions" begin
    regions_m1 = generate_regions(system_m1)
    regions_m2 = generate_regions(system_m2)

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

    @testset "Symmetric region pruning" begin
        @test [
            region.Ξ for region in generate_regions(system_m1; prune_symmetric=true)
        ] == [[-1], [0]]

        @test [
            region.Ξ for region in generate_regions(system_m2; prune_symmetric=true)
        ] == [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0]]

        asymmetric_nominal = SaturatedSystem(
            A, B, K, zeros(0, 2), Float64[], Normal.(zeros(2), ones(2));
            v_min=[-0.5], v_max=[1.0],
        )
        @test length(generate_regions(asymmetric_nominal; prune_symmetric=true)) == 3
    end

    @testset "Asymmetric bounds and affine dynamics" begin
        u_min = [-2.0, -3.0]
        u_max = [1.0, 4.0]
        system = SaturatedSystem(
            A_2,
            B_2,
            K_2,
            zeros(0, 2),
            Float64[],
            Normal.(zeros(2), ones(2));
            u_min,
            u_max,
        )
        regions = generate_regions(system)
        @test [
            region.Ξ for region in generate_regions(system; prune_symmetric=true)
        ] == [region.Ξ for region in regions]
        v = zeros(2)

        for region in regions
            # choose an input or "command" that activates saturation pattern Ξ
            command = [
                ξ == -1 ? u_min[i] - 1 : ξ == 1 ? u_max[i] + 1 : (u_min[i] + u_max[i]) / 2
                    for (i, ξ) in enumerate(region.Ξ)
            ]
            # solve for points e such that K_2 e + v = command
            e = K_2 \ command
            # verify that (e, v) pair belongs in fact to the region
            @test all(region.R * [e; v] .≤ region.c)
            @test region.d̄ ≈ B_2 * region.d
            @test region.Ā * e + region.B̄ * v + region.d̄ ≈
                saturated_error_dynamics(system, e, v)
        end
    end

    @testset "Coefficient types" begin
        A_exact = [9 // 10 1 // 10; 0 // 1 4 // 5]
        B_exact = [1 // 2 0 // 1; 3 // 10 7 // 10]
        K_exact = [1 // 5 -1 // 10; -2 // 5 3 // 10]
        u_min_exact = [-2 // 1, -3 // 1]
        u_max_exact = [1 // 1, 4 // 1]

        for T in (Rational{Int}, BigFloat)
            system = SaturatedSystem(
                T.(A_exact),
                T.(B_exact),
                T.(K_exact),
                zeros(T, 0, 2),
                T[],
                Normal.(zeros(2), ones(2));
                u_min=T.(u_min_exact),
                u_max=T.(u_max_exact),
            )
            regions = generate_regions(system)

            @test regions isa Vector{SaturationRegion{T}}
            @test all(region -> region.d̄ == T.(B_exact) * region.d, regions)
        end
    end

end

@testset "Semialgebraic saturation generators" begin
    symmetric_system = SaturatedSystem(
        A, B, [2.0 -3.0], zeros(0, 2), Float64[], Normal.(zeros(2), ones(2)),
    )
    @polyvar e_symmetric[1:2] s_symmetric[1:1]
    general_symmetric_generators = semialgebraic_saturation_generators(
        symmetric_system, e_symmetric, s_symmetric,
    )

    Ke_symmetric = 2.0e_symmetric[1] - 3.0e_symmetric[2]
    @test general_symmetric_generators == [
        (s_symmetric[1] + 2.0) * (2.0 - s_symmetric[1]) / 4.0,
        (Ke_symmetric - s_symmetric[1]) * s_symmetric[1] / 3.0,
        (s_symmetric[1] - Ke_symmetric) * (-s_symmetric[1]) / 3.0,
    ]

    pruned_symmetric_generators = semialgebraic_saturation_generators(
        symmetric_system, e_symmetric, s_symmetric; prune_symmetric=true,
    )
    @test pruned_symmetric_generators == [
        (2.0 - s_symmetric[1]) / 2.0,
        (2.0 + s_symmetric[1]) / 2.0,
        s_symmetric[1] * (Ke_symmetric - s_symmetric[1]) / 3.0,
    ]

    u_min = [-2.0, -3.0]
    u_max = [1.0, 4.0]
    v_min = [-1.0, -2.0]
    v_max = [0.5, 2.0]
    system = SaturatedSystem(
        A_2,
        B_2,
        K_2,
        zeros(0, 2),
        Float64[],
        Normal.(zeros(2), ones(2));
        u_min,
        u_max,
        v_min,
        v_max,
    )
    @polyvar e_semialgebraic[1:2] s_semialgebraic[1:2]
    generators = semialgebraic_saturation_generators(
        system, e_semialgebraic, s_semialgebraic,
    )

    @test length(generators) == 6
    Ke_1 = 0.2e_semialgebraic[1] - 0.1e_semialgebraic[2]
    Ke_2 = -0.4e_semialgebraic[1] + 0.3e_semialgebraic[2]
    @test generators[1] ==
          (s_semialgebraic[1] + 2.5) * (2.0 - s_semialgebraic[1]) / 5.0
    @test generators[2] ==
          (s_semialgebraic[2] + 5.0) * (6.0 - s_semialgebraic[2]) / 30.0
    @test generators[3] ==
          (Ke_1 - s_semialgebraic[1]) * (s_semialgebraic[1] + 1.0)
    @test generators[4] ==
          (Ke_2 - s_semialgebraic[2]) * (s_semialgebraic[2] + 1.0)
    @test generators[5] ==
          (s_semialgebraic[1] - Ke_1) * (0.5 - s_semialgebraic[1])
    @test generators[6] ==
          (s_semialgebraic[2] - Ke_2) * (2.0 - s_semialgebraic[2]) / 2.0

    for e in ([0.0, 0.0], [4.0, -3.0], [-7.0, 2.0]),
            v in (v_min, v_max, (v_min + v_max) / 2)
        s = clamp.(system.K * e + v, u_min, u_max) - v
        @test all(semialgebraic_saturation_generators(system, e, s) .>= -1e-12)
    end

    scalar_system = SaturatedSystem(
        A, B, [1.0 0.0], zeros(0, 2), Float64[], Normal.(zeros(2), ones(2));
        u_min=[-2.0], u_max=[3.0], v_min=[-1.0], v_max=[0.5],
    )
    @test semialgebraic_saturation_generators(scalar_system, [0.0, 0.0], [5.0])[1] < 0
    @test semialgebraic_saturation_generators(scalar_system, [0.0, 0.0], [0.5])[2] < 0
    @test semialgebraic_saturation_generators(scalar_system, [0.0, 0.0], [-0.5])[3] < 0

    full_nominal_system = SaturatedSystem(
        A_2, B_2, K_2, zeros(0, 2), Float64[], Normal.(zeros(2), ones(2));
        u_min, u_max,
    )
    @test semialgebraic_saturation_generators(
        system, e_semialgebraic, s_semialgebraic; prune_symmetric=true,
    ) == semialgebraic_saturation_generators(
        full_nominal_system, e_semialgebraic, s_semialgebraic; prune_symmetric=true,
    )

    @test_throws DimensionMismatch semialgebraic_saturation_generators(
        system, zeros(1), zeros(2),
    )
    @test_throws DimensionMismatch semialgebraic_saturation_generators(
        system, zeros(2), zeros(1),
    )
end
