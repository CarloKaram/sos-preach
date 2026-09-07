using Test
using DynamicPolynomials
include("../src/monomials.jl")

function polynomial_exponents(n, r, include_constant=false)
    @polyvar x[1:n]
    degrees = include_constant ? (0:r) : (1:r)
    return [Tuple(degree(monomial, variable) for variable in x)
            for monomial in monomials(x, degrees)]
end

@testset "monomial_exponents" begin
    @testset "Generation and ordering" begin
        @test monomial_exponents(1, 4) == [(1,), (2,), (3,), (4,)]

        expected = [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
        @test monomial_exponents(2, 2) == expected
        @test monomial_exponents(2, 2, true) == vcat([(0, 0)], expected)
    end

    @testset "Boundary cases" begin
        @test monomial_exponents(2, 0) == NTuple{2, Int}[]
        @test monomial_exponents(2, 0, true) == [(0, 0)]
        @test monomial_exponents(0, 2) == NTuple{0, Int}[]
        @test monomial_exponents(0, 2, true) == [()]

        @test_throws ArgumentError monomial_exponents(-1, 2)
        @test_throws ArgumentError monomial_exponents(2, -1)
    end

    @testset "Exponent lookup" begin
        exponents = monomial_exponents(2, 2)
        index = exponent_index(exponents)

        @test all(index[α] == i for (i, α) in enumerate(exponents))
    end

    @testset "DynamicPolynomials ordering" begin
        for (n, r) in ((2, 3), (3, 2))
            @test monomial_exponents(n, r) == polynomial_exponents(n, r)
            @test monomial_exponents(n, r, true) == polynomial_exponents(n, r, true)
        end
    end
end
