using Distributions
using LinearAlgebra
using Test

include("../src/system.jl")
include("../src/moments.jl")
include("../src/monomials.jl")

A = [0.8 0.1; 0.0 0.7]
B = [0.0; 1.0;;]
K = [0.5 -0.3]
H = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
h = fill(10.0, 4)
noise = Normal.(zeros(2), ones(2))

@testset "SaturatedSystem" begin
    @testset "Construction" begin
        system = SaturatedSystem(A, B, K, H, h, noise)

        @test system isa SaturatedSystem{Float64,typeof(noise)}
        @test system.u_min == [-1.0]
        @test system.u_max == [1.0]
        @test size(system.A, 1) == 2
        @test size(system.B, 2) == 1
        @test size(system.H, 1) == 4
        @test system.noise === noise

        asymmetric = SaturatedSystem(A, B, K, H, h, noise; u_min=[-2.0], u_max=[3.0])
        @test asymmetric.u_min == [-2.0]
        @test asymmetric.u_max == [3.0]
    end

    @testset "Invalid data" begin
        @test_throws DimensionMismatch SaturatedSystem(A[:, 1:1], B, K, H, h, noise)
        @test_throws DimensionMismatch SaturatedSystem(A, zeros(1, 1), K, H, h, noise)
        @test_throws DimensionMismatch SaturatedSystem(A, B, zeros(2, 2), H, h, noise)
        @test_throws DimensionMismatch SaturatedSystem(A, B, K, zeros(4, 3), h, noise)
        @test_throws DimensionMismatch SaturatedSystem(A, B, K, H, h[1:3], noise)
        @test_throws DimensionMismatch SaturatedSystem(A, B, K, H, h, noise; u_min=[-1.0, -1.0])
        @test_throws DimensionMismatch SaturatedSystem(A, B, K, H, h, [Normal()])
        @test_throws ArgumentError SaturatedSystem(A, B, K, H, h, noise; u_min=[1.0], u_max=[1.0])
        @test_throws ArgumentError SaturatedSystem(A, B, K, H, h, "noise")
    end

    @testset "Saturated error dynamics" begin
        system = SaturatedSystem(A, B, K, H, h, noise)
        e = [10.0, 0.0]
        v = [0.2]

        @test saturated_error_dynamics(system, e, v) ≈ A * e + B * ([1.0] - v)
    end

    @testset "Noise moment matrix" begin
        A₁ = [0.9;;]
        B₁ = [1.0;;]
        K₁ = [0.0;;]
        H₁ = [1.0; -1.0;;]
        h₁ = [10.0, 10.0]
        exponents = monomial_exponents(1, 2)

        gaussian = SaturatedSystem(A₁, B₁, K₁, H₁, h₁, Normal(0.0, 1.0))
        @test moment_matrix(gaussian.noise, exponents) ≈ [1.0 0.0; 0.0 3.0]

        gamma = SaturatedSystem(
            A₁,
            B₁,
            K₁,
            H₁,
            h₁,
            Gamma(0.1, 1.0) - 0.1,
        )
        @test moment_matrix(gamma.noise, exponents) ≈ [0.1 0.2; 0.2 0.63]
    end
end
