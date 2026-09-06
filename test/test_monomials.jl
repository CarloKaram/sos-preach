using Test
include("../src/monomials.jl")

@testset "monomial_indices" begin
    n, r = 2, 1
    indices = monomial_indices(n, r)

    @test length(indices) == binomial(n + r, r) - 1

    @test all(length(α) == n for α in indices)
    @test all(sum(α) == r for α in indices)

    @test indices == [[0, 1], [1, 0]]

    @test monomial_indices(0, 0) == Vector{Int}[]
    @test monomial_indices(0, 2) == Vector{Int}[]
    @test monomial_indices(2, 0) == Vector{Int}[]

    n, r = 2, 2
    indices = monomial_indices(n, r)
    @test all(sum(α) <= r && sum(α) > 0 for α in indices)
    @test monomial_indices(2, 2) == [[0, 1], [1, 0], [0, 2], [1,1], [2, 0]]
end