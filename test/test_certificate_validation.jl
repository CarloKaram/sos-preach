using DynamicPolynomials
using LinearAlgebra
using SumOfSquares
using Test

include("../src/certificate_validation.jl")

@polyvar x

@testset "SOS certificate validation" begin
    basis = SumOfSquares.MB.SubBasis{SumOfSquares.MB.Monomial}(monomials([x], 0:1))
    exact_gram = GramMatrix([1.0 0.0; 0.0 1.0], basis)

    exact = _gram_check("exact", 1 + x^2, exact_gram, 1e-10)
    @test exact.passed
    @test exact.margin == 1.0
    @test exact.residual == 0.0

    mismatch = _gram_check("mismatch", 1 + x + x^2, exact_gram, 1e-10)
    @test !mismatch.passed
    @test mismatch.residual == 1.0

    indefinite_gram = GramMatrix([1.0 0.0; 0.0 -0.01], basis)
    indefinite = _gram_check(
        "indefinite", polynomial(indefinite_gram), indefinite_gram, 1e-10,
    )
    @test !indefinite.passed
    @test indefinite.margin == -0.01

    nonsymmetric = _matrix_check("nonsymmetric", [1.0 1.0; 0.0 1.0], 1e-10)
    @test !nonsymmetric.passed
    @test nonsymmetric.margin == -Inf

    nonfinite = _matrix_check("nonfinite", [Inf;;], 1e-10)
    @test !nonfinite.passed
    @test nonfinite.margin == -Inf
end
