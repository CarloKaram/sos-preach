using Distributions
using LinearAlgebra
using Test

include("../src/moments.jl")
include("../src/monomials.jl")

@testset "moment_matrix" begin
    basis_exponents = monomial_exponents(1, 2)

    @testset "Gaussian noise" begin
        M = moment_matrix(Normal(0.0, 1.0), basis_exponents)
        @test M ≈ [1.0 0.0; 0.0 3.0]
    end

    @testset "Shifted Gamma noise" begin
        noise = Gamma(0.1, 1.0) - 0.1
        M = moment_matrix(noise, basis_exponents)

        @test M ≈ [0.1 0.2; 0.2 0.63]

        reversed_exponents = reverse(basis_exponents)
        @test moment_matrix(noise, reversed_exponents) ≈ [0.63 0.2; 0.2 0.1]
    end

    @testset "Independent multivariate noise" begin
        noise = ContinuousUnivariateDistribution[
            Normal(0.0, 1.0),
            Gamma(0.1, 1.0) - 0.1,
        ]
        exponents = monomial_exponents(2, 1)
        expected = [0.1 0.0; 0.0 1.0]

        @test moment_matrix(noise, exponents) ≈ expected
        @test moment_matrix(product_distribution(noise), exponents) ≈ expected
    end

    @testset "Diagonal multivariate Gaussian" begin
        noise = MvNormal([1.0, -2.0], Diagonal([4.0, 9.0]))
        exponents = monomial_exponents(2, 1)

        @test moment_matrix(noise, exponents) ≈ [13.0 -2.0; -2.0 5.0]

        correlated = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])
        @test_throws ArgumentError moment_matrix(correlated, exponents)
    end
end
