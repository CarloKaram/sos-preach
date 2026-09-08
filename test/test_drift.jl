using Distributions
using DynamicPolynomials
using JuMP
using LinearAlgebra
using SumOfSquares
using Test

include("../src/system.jl")
include("../src/saturation.jl")
include("../src/moments.jl")
include("../src/monomials.jl")
include("../src/drift.jl")

@polyvar e v x[1:2] u

@testset "Regional drift polynomial" begin
    basis_1d = monomial_exponents(1, 2)
    noise_basis_1d = monomial_exponents(1, 2, true)

    @testset "Translation identity" begin
        f = e + v
        T = _translation_matrix([f], basis_1d)

        @test size(T) == (2, 3)
        @test T == [f 1 0; f^2 2f 1]
    end

    A = [1.0;;]
    B = [0.0;;]
    K = [0.0;;]
    H = [1.0; -1.0;;]
    h = [10.0, 10.0]
    system = SaturatedSystem(A, B, K, H, h, Normal())
    region = build_region(system, [0])

    @testset "Gaussian drift" begin
        M_w = moment_matrix(Normal(), noise_basis_1d)
        Q = [1.0 0.0; 0.0 1.0]

        Δ = drift_polynomial(Q, 0.5, 2.0, region, [e], [v], M_w, basis_1d)
        expected = -0.5e^4 - 6.5e^2 - 2.0

        @test Δ ≈ expected
    end

    @testset "Shifted Gamma drift" begin
        noise = Gamma(0.1, 1.0) - 0.1
        M_w = moment_matrix(noise, noise_basis_1d)
        Q = [0.0 0.0; 0.0 1.0]

        Δ = drift_polynomial(Q, 0.5, 1.0, region, [e], [v], M_w, basis_1d)
        expected = -0.5e^4 - 0.6e^2 - 0.8e + 0.37

        @test Δ ≈ expected
    end

    @testset "Regional affine dynamics" begin
        affine_system = SaturatedSystem(
            [0.8;;], [1.0;;], [0.5;;], H, h, Normal();
            u_min=[-2.0], u_max=[1.0]
        )
        upper_region = build_region(affine_system, [1])
        M_w = moment_matrix(Normal(), noise_basis_1d)
        f = 0.8e - v + 1.0

        Δ = drift_polynomial([1.0 0.0; 0.0 0.0], 0.5, 0.3,
                             upper_region, [e], [v], M_w, basis_1d)

        @test Δ ≈ 0.5e^2 + 0.3 - (f^2 + 1.0)
    end

    @testset "Multivariate ordering" begin
        A₂ = Matrix{Float64}(I, 2, 2)
        B₂ = zeros(2, 1)
        K₂ = zeros(1, 2)
        H₂ = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        h₂ = fill(10.0, 4)
        noise = [Normal(0.0, 1.0), Normal(0.0, 2.0)]
        system₂ = SaturatedSystem(A₂, B₂, K₂, H₂, h₂, noise)
        region₂ = build_region(system₂, [0])
        basis₂ = monomial_exponents(2, 1)
        noise_basis₂ = monomial_exponents(2, 1, true)
        M_w = moment_matrix(noise, noise_basis₂)
        Q = [2.0 1.0; 1.0 3.0]

        V = 2.0x[2]^2 + 2.0x[1] * x[2] + 3.0x[1]^2
        Δ = drift_polynomial(Q, 0.5, 0.7, region₂, x, [u], M_w, basis₂)
        expected_successor = 2.0(x[2]^2 + 4.0) +
                             2.0x[1] * x[2] +
                             3.0(x[1]^2 + 1.0)

        @test basis₂ == [(0, 1), (1, 0)]
        @test Δ ≈ 0.5V + 0.7 - expected_successor
    end

    @testset "SOS compatibility" begin
        M_w = moment_matrix(Normal(), noise_basis_1d)
        model = SOSModel()
        @variable(model, Q[1:2, 1:2], PSD)

        Δ = drift_polynomial(Q, 0.5, 1.0, region, [e], [v], M_w, basis_1d)
        constraint = @constraint(model, Δ in SOSCone())

        @test constraint isa JuMP.ConstraintRef
    end
end
