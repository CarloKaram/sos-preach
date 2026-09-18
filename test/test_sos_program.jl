using CSDP
using Distributions
using DynamicPolynomials
using JuMP
using Test

include("../src/system.jl")
include("../src/saturation.jl")
include("../src/monomials.jl")
include("../src/moments.jl")
include("../src/drift.jl")
include("../src/certificate_validation.jl")
include("../src/sos_program.jl")
include("../scripts/lambda_search.jl")

mutable struct CountingCSDPOptimizer
    calls::Int
    fail_on::Int
end

function (factory::CountingCSDPOptimizer)()
    factory.calls += 1
    optimizer = CSDP.Optimizer()
    JuMP.MOI.set(optimizer, JuMP.MOI.RawOptimizerAttribute("axtol"), 1e-6)
    JuMP.MOI.set(optimizer, JuMP.MOI.RawOptimizerAttribute("atytol"), 1e-4)
    JuMP.MOI.set(optimizer, JuMP.MOI.RawOptimizerAttribute("objtol"), 1e-6)
    if factory.calls == factory.fail_on
        JuMP.MOI.set(optimizer, JuMP.MOI.RawOptimizerAttribute("maxiter"), 0)
    end
    optimizer
end

test_system() = SaturatedSystem(
    [0.2;;],
    [0.0;;],
    [0.0;;],
    [1.0; -1.0;;],
    [100.0, 100.0],
    Normal(),
)

struct StalledFeasibleModel end
JuMP.termination_status(::StalledFeasibleModel) = JuMP.MOI.SLOW_PROGRESS
JuMP.primal_status(::StalledFeasibleModel) = JuMP.MOI.FEASIBLE_POINT
JuMP.has_values(::StalledFeasibleModel) = true

function solve_test_system(optimizer; kwargs...)
    solve_sos_program(
        test_system(),
        optimizer;
        λ=0.5,
        ε=0.5,
        τ_inf=10.0,
        ζ=1.0,
        γ=0.01,
        r=1,
        σ_half_degree=0,
        μ_half_degree=0,
        validation_tolerance=1e-4,
        kwargs...,
    )
end

@testset "Lean SOS program" begin
    @testset "Theoretical inputs" begin
        optimizer = CountingCSDPOptimizer(0, 0)
        common = (
            ε=0.5,
            τ_inf=10.0,
            ζ=1.0,
            γ=0.01,
            r=1,
            σ_half_degree=0,
            μ_half_degree=0,
        )

        @test_throws ArgumentError solve_sos_program(
            test_system(), optimizer; common..., λ=1.0,
        )
        @test_throws ArgumentError solve_sos_program(
            test_system(), optimizer; common..., λ=0.5, ε=1.0,
        )
        @test_throws ArgumentError solve_sos_program(
            test_system(), optimizer; common..., λ=0.5, ζ=10.0,
        )
        @test_throws ArgumentError solve_sos_program(
            test_system(), optimizer; common..., λ=0.5, γ=0.0,
        )
        @test optimizer.calls == 0
    end

    @testset "Normalized regional generators" begin
        system = SaturatedSystem(
            [0.2;;], [0.0;;], [0.5;;], [1.0; -1.0;;], [100.0, 100.0], Normal();
            u_min=[-2.0], u_max=[3.0],
        )
        region = build_region(system, [1])
        @polyvar e[1:1] v[1:1]
        generators = _normalized_region_generators(region, e, v)

        @test generators[1] == (-3.0 + 0.5e[1] + v[1]) / 3.0
        @test generators[2] == 1.0 - v[1] / 3.0
        @test generators[3] == 1.0 + v[1] / 2.0
    end

    @testset "Complete alternating pair" begin
        optimizer = CountingCSDPOptimizer(0, 0)
        result = solve_test_system(
            optimizer; max_iterations=1, convergence_tolerance=0.0,
        )

        @test optimizer.calls == 2
        @test result.status == :numerically_validated
        @test result.stop_reason == :iteration_limit
        @test result.validation.passed
        @test isempty(result.validation.failed_checks)
        @test result.solution !== nothing
        @test result.solution.β == 2.25
        @test result.solution.basis_exponents == [(1,)]
        @test size(result.solution.Q) == (1, 1)
        @test all(isfinite, result.solution.Q)
        @test 0 <= result.solution.ρ <= 1
        @test length(result.solution.σ) == 3
        @test length(result.solution.μ) == 2
        @test length(result.history) == 1
        @test result.history[1].Q_primal == JuMP.MOI.FEASIBLE_POINT
        @test result.history[1].μ_primal == JuMP.MOI.FEASIBLE_POINT
        @test all(isfinite, values(result.timing))
        @test all(>=(0), values(result.timing))
        @test result.history[1].q_time >= 0
        @test result.history[1].μ_time >= 0
        @test result.history[1].alternation_time >=
              result.history[1].q_time + result.history[1].μ_time - 1e-6
        @test isapprox(
            result.timing.total_time,
            result.timing.setup_time + result.timing.alternation_time +
            result.timing.validation_time + result.timing.overhead_time;
            atol=1e-6,
        )
    end

    @testset "Stalled feasible point" begin
        model = StalledFeasibleModel()
        @test termination_status(model) == JuMP.MOI.SLOW_PROGRESS
        @test _has_feasible_point(model)
    end

    @testset "Convergence" begin
        optimizer = CountingCSDPOptimizer(0, 0)
        result = solve_test_system(
            optimizer; max_iterations=3, convergence_tolerance=1.0,
        )

        @test optimizer.calls == 2
        @test result.stop_reason == :converged
        @test length(result.history) == 1
        @test result.status == :numerically_validated
    end

    @testset "Preserve last complete solution" begin
        optimizer = CountingCSDPOptimizer(0, 3)
        result = solve_test_system(
            optimizer; max_iterations=3, convergence_tolerance=0.0,
        )

        @test optimizer.calls == 3
        @test result.stop_reason == :q_step_failed
        @test result.solution !== nothing
        @test result.status == :numerically_validated
        @test length(result.history) == 2
        @test result.history[end].Q_primal != JuMP.MOI.FEASIBLE_POINT
    end

    @testset "No complete solution" begin
        optimizer = CountingCSDPOptimizer(0, 1)
        result = solve_test_system(optimizer; max_iterations=1)

        @test optimizer.calls == 1
        @test result.status == :no_solution
        @test result.stop_reason == :q_step_failed
        @test result.solution === nothing
        @test result.validation === nothing
        @test result.history[1].q_time >= 0
        @test result.history[1].μ_time == 0
        @test result.timing.validation_time == 0
    end

    @testset "Golden-section timing" begin
        solve_at_λ = λ -> (
            solution=(ρ=(λ - 0.5)^2,),
            status=:numerically_validated,
            timing=(
                setup_time=0.0,
                q_time=0.0,
                μ_time=0.0,
                alternation_time=0.0,
                validation_time=0.0,
                optimization_time=0.0,
                overhead_time=0.0,
                total_time=0.0,
            ),
        )
        winner = golden_section_search(solve_at_λ, 0.0, 1.0; tolerance=0.1)

        @test !isempty(winner.evaluations)
        @test length(winner.evaluations) == 7
        @test all(evaluation -> evaluation.timing.total_time == 0, winner.evaluations)
        @test winner.timing.λ_evaluation_time == 0
        @test winner.timing.total_time >= winner.timing.λ_evaluation_time
    end

    @testset "Solution helpers" begin
        solution = SOSSolution(
            [2.0;;], 0.2, 0.5, 2.25, 0.5, 10.0, 1.0, 0.01,
            [(1,)], Any[], Any[],
        )
        @polyvar y

        @test lyapunov_polynomial(solution, [y]) == 2.0y^2
        @test prs_level(solution, 0) == 0.0
        @test prs_level(solution, 3) == 7.875
    end
end
