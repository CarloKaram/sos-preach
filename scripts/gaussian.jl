using Distributions
using LinearAlgebra
using MosekTools
using Serialization

include("../src/system.jl")
include("../src/saturation.jl")
include("../src/monomials.jl")
include("../src/moments.jl")
include("../src/drift.jl")
include("../src/certificate_validation.jl")
include("../src/sos_program.jl")
include("lambda_search.jl")
include("plotting_utility.jl")

# Model, K, and original constrains
const A = [0.89 0.10; 0.10 0.89]
const B = [0.0; 1.0;;]
const K = [-0.282 -0.8415]
# const K = 0.01*[-0.41407  -0.917]
# const K = [-0.8415 -0.282]
# const K = 0.9*[-0.38909  -0.850484]
const H = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
const h = fill(10.0, 4)

# Set noise as Gaussian
noise = MvNormal(zeros(2), Diagonal(fill(0.1, 2)))

system = SaturatedSystem(A, B, K, H, h, noise; u_min=[-1.0], u_max=[1.0])

γ_floor = 1e-5
τ_∞ = 1

gaussian_results = []

for r in [1, 2, 4]
    println("\n STARTING CASE: r = ", r)

    μ_init = 5

    solve_at_λ = λ -> begin
        result = solve_sos_program(
            system,
            MosekTools.Optimizer;
            λ,
            ε = 0.2,
            τ_inf = τ_∞,
            ζ = 1e-5,
            γ_floor,
            r,
            σ_half_degree = r,
            μ_half_degree = 1,
            initial_μ = μ_init,
            max_iterations = 5,
            validation_tolerance = 2e-7,
            silent=true,
            even=false,
            saturation_formulation=:semialgebraic,
        )

        println("\nCASE: r = ", r, ", λ = ", λ)

        return result
    end

    winner = golden_section_search(solve_at_λ, eigmax(A)^(2*r)-0.01, 0.999)
    best_result = winner.result
    best_λ = winner.λ
    best_ρ = winner.ρ

    print_timing(winner)

    push!(gaussian_results, best_result)

    println("\nWinner: r = ", r)
    if best_result === nothing
        println("  no solution was found")
    else
        println("  λ:                     ", best_λ)
        println("  ρ:                     ", best_ρ)
        println("  γ:                     ", best_result.solution.γ)
        println("  status:                ", best_result.status)
        println(
            "  numerically validated: ",
            best_result.validation !== nothing && best_result.validation.passed,
        )
    end
end

Random.seed!(1234)
fig = plot_pub(
    gaussian_results; system, t=100, n_samples=10_000, sample_at_bounds=true,
)
display(fig)
