using DynamicPolynomials
using JuMP
using SumOfSquares

struct SOSSolution{T,E,S,M}
    Q::Matrix{T}
    ρ::T
    λ::T
    β::T
    ε::T
    τ_inf::T
    ζ::T
    γ::T
    basis_exponents::E
    σ::S
    μ::M
end

struct SOSSolveResult{S,H,V}
    status::Symbol
    stop_reason::Symbol
    solution::S
    history::H
    validation::V
end

function _monomial_basis(variables, exponents)
    [prod(variables[i]^α[i] for i in eachindex(variables)) for α in exponents]
end

function lyapunov_polynomial(solution::SOSSolution, e)
    Φ = _monomial_basis(e, solution.basis_exponents)
    sum(solution.Q[i, j] * Φ[i] * Φ[j] for i in eachindex(Φ), j in eachindex(Φ))
end

function prs_level(solution::SOSSolution, k::Integer)
    (1 - solution.λ^k) / (solution.ε * (1 - solution.λ)) * solution.β
end

function _normalized_region_generators(region, e, v)
    variables = [e; v]
    [
        begin
            row = view(region.R, ℓ, :)
            scale = max(one(eltype(region.R)), abs(region.c[ℓ]), maximum(abs, row))
            (region.c[ℓ] - sum(row[k] * variables[k] for k in eachindex(variables))) /
            scale
        end
        for ℓ in axes(region.R, 1)
    ]
end

function _solve_q_step(
        system,
        optimizer,
        silent,
        regions,
        generators,
        e,
        v,
        basis_exponents,
        M_w,
        λ,
        β,
        τ_inf,
        γ,
        σ_half_degree,
        fixed_μ,
    )
    model = SOSModel(optimizer)
    silent && set_silent(model)
    Φ = _monomial_basis(e, basis_exponents)
    σ_basis = monomials([e; v], 0:σ_half_degree)

    @variable(model, Q[1:length(Φ), 1:length(Φ)], PSD)
    @variable(model, 0 <= ρ <= 1)
    V = sum(Q[i, j] * Φ[i] * Φ[j] for i in eachindex(Φ), j in eachindex(Φ))

    positivity_polynomial = V - γ * sum(eᵢ^2 for eᵢ in e)
    positivity_constraint = @constraint(model, positivity_polynomial in SOSCone())
    targets = Any[(
        label="positivity",
        polynomial=positivity_polynomial,
        constraint=positivity_constraint,
    )]
    multipliers = Pair{String,Any}[]
    σ_grams = Vector{Any}[]

    for (j, region) in enumerate(regions)
        grams = @variable(
            model,
            [1:length(generators[j])],
            SOSPoly(σ_basis),
            base_name="σ_$(j)",
        )
        σ = polynomial.(grams)
        Δ = drift_polynomial(Q, λ, β, region, e, v, M_w, basis_exponents)
        polynomial_value = Δ - sum(
            σ[ℓ] * generators[j][ℓ] for ℓ in eachindex(σ)
        )
        constraint = @constraint(model, polynomial_value in SOSCone())
        push!(
            targets,
            (label="drift region $j", polynomial=polynomial_value, constraint=constraint),
        )
        append!(multipliers, ["σ[$j,$ℓ]" => grams[ℓ] for ℓ in eachindex(grams)])
        push!(σ_grams, grams)
    end

    for i in axes(system.H, 1)
        facet = sum(system.H[i, k] * e[k] for k in eachindex(e))
        polynomial_value = ρ * system.h[i] - facet - fixed_μ[i] * (τ_inf - V)
        @constraint(model, polynomial_value in SOSCone())
    end

    @objective(model, Min, ρ)
    optimize!(model)
    return (
        model=model,
        Q=Q,
        ρ=ρ,
        σ_grams=σ_grams,
        targets=targets,
        multipliers=multipliers,
    )
end

function _solve_μ_step(
        system,
        optimizer,
        silent,
        e,
        Q,
        basis_exponents,
        τ_inf,
        μ_half_degree,
    )
    model = SOSModel(optimizer)
    silent && set_silent(model)
    Φ = _monomial_basis(e, basis_exponents)
    μ_basis = monomials(e, 0:μ_half_degree)
    V = sum(Q[i, j] * Φ[i] * Φ[j] for i in eachindex(Φ), j in eachindex(Φ))
    n_facets = size(system.H, 1)

    μ_grams = @variable(model, [1:n_facets], SOSPoly(μ_basis), base_name="μ")
    μ = polynomial.(μ_grams)
    @variable(model, 0 <= ρ <= 1)

    targets = Any[]
    for i in 1:n_facets
        facet = sum(system.H[i, k] * e[k] for k in eachindex(e))
        polynomial_value = ρ * system.h[i] - facet - μ[i] * (τ_inf - V)
        constraint = @constraint(model, polynomial_value in SOSCone())
        push!(
            targets,
            (label="containment facet $i", polynomial=polynomial_value,
             constraint=constraint),
        )
    end

    @objective(model, Min, ρ)
    optimize!(model)
    multipliers = Pair{String,Any}[
        "μ[$i]" => μ_grams[i] for i in eachindex(μ_grams)
    ]
    return (model=model, ρ=ρ, μ_grams=μ_grams, targets=targets, multipliers=multipliers)
end

_has_feasible_point(model) =
    primal_status(model) == JuMP.MOI.FEASIBLE_POINT && has_values(model)

function _solution(Q, ρ, λ, β, ε, τ_inf, ζ, γ, basis_exponents, σ, μ)
    T = promote_type(
        eltype(Q), typeof(ρ), typeof(λ), typeof(β), typeof(ε),
        typeof(τ_inf), typeof(ζ), typeof(γ),
    )
    SOSSolution(
        Matrix{T}(Q), T(ρ), T(λ), T(β), T(ε), T(τ_inf), T(ζ), T(γ),
        basis_exponents, σ, μ,
    )
end

"""
    solve_sos_program(system, optimizer; ...)

Solve the paper's fixed-`λ` alternating SOS program. The multiplier degree
arguments are Gram-basis half-degrees. A returned solution is accepted only
after direct numerical checks of the Gram matrices from its `Q`- and `μ`-steps.
"""
function solve_sos_program(
        system::SaturatedSystem,
        optimizer;
        λ,
        ε,
        τ_inf,
        ζ,
        γ,
        r,
        σ_half_degree,
        μ_half_degree,
        max_iterations=5,
        convergence_tolerance=1e-3,
        initial_μ=1,
        validation_tolerance=1e-7,
        silent=true,
    )
    0 < λ < 1 || throw(ArgumentError("λ must satisfy 0 < λ < 1"))
    0 < ε < 1 || throw(ArgumentError("ε must satisfy 0 < ε < 1"))
    0 <= ζ < τ_inf || throw(ArgumentError("ζ must satisfy 0 ≤ ζ < τ_inf"))
    γ > 0 || throw(ArgumentError("γ must be positive"))

    β = ε * (1 - λ) * (τ_inf - ζ)
    n = size(system.A, 1)
    m = size(system.B, 2)
    basis_exponents = monomial_exponents(n, r)
    noise_exponents = monomial_exponents(n, r, true)
    M_w = moment_matrix(system.noise, noise_exponents)
    regions = generate_regions(system)
    @polyvar e[1:n] v[1:m]
    generators = [_normalized_region_generators(region, e, v) for region in regions]

    fixed_μ = [initial_μ * one(e[1]) for _ in axes(system.H, 1)]
    previous_ρ = 1.0
    history = NamedTuple[]
    last_complete = nothing
    stop_reason = :iteration_limit

    for iteration in 1:max_iterations
        Q_step = _solve_q_step(
            system, optimizer, silent, regions, generators, e, v,
            basis_exponents, M_w, λ, β, τ_inf, γ, σ_half_degree, fixed_μ,
        )
        Q_status = termination_status(Q_step.model)
        Q_primal = primal_status(Q_step.model)
        if !_has_feasible_point(Q_step.model)
            push!(history, (
                iteration=iteration, Q_status=Q_status, Q_primal=Q_primal,
                Q_ρ=nothing, μ_status=nothing, μ_primal=nothing, ρ=nothing,
                decrease=nothing,
            ))
            stop_reason = :q_step_failed
            break
        end

        Q = Matrix(value.(Q_step.Q))
        Q_ρ = value(Q_step.ρ)
        μ_step = _solve_μ_step(
            system, optimizer, silent, e, Q, basis_exponents, τ_inf, μ_half_degree,
        )
        μ_status = termination_status(μ_step.model)
        μ_primal = primal_status(μ_step.model)
        if !_has_feasible_point(μ_step.model)
            push!(history, (
                iteration=iteration, Q_status=Q_status, Q_primal=Q_primal,
                Q_ρ=Q_ρ, μ_status=μ_status, μ_primal=μ_primal, ρ=nothing,
                decrease=nothing,
            ))
            stop_reason = :mu_step_failed
            break
        end

        ρ = value(μ_step.ρ)
        σ = [value.(polynomial.(grams)) for grams in Q_step.σ_grams]
        μ = value.(polynomial.(μ_step.μ_grams))
        decrease = previous_ρ - ρ
        solution = _solution(
            Q, ρ, λ, β, ε, τ_inf, ζ, γ, basis_exponents, σ, μ,
        )
        last_complete = (
            solution=solution,
            targets=[Q_step.targets; μ_step.targets],
            multipliers=[Q_step.multipliers; μ_step.multipliers],
        )
        push!(history, (
            iteration=iteration, Q_status=Q_status, Q_primal=Q_primal,
            Q_ρ=Q_ρ, μ_status=μ_status, μ_primal=μ_primal, ρ=ρ,
            decrease=decrease,
        ))
        fixed_μ = μ

        if decrease <= convergence_tolerance
            stop_reason = :converged
            break
        end
        previous_ρ = ρ
    end

    last_complete === nothing &&
        return SOSSolveResult(:no_solution, stop_reason, nothing, history, nothing)

    validation = _validate_solution(
        last_complete.solution.Q,
        last_complete.targets,
        last_complete.multipliers;
        tolerance=validation_tolerance,
    )
    status = validation.passed ? :numerically_validated : :validation_failed
    return SOSSolveResult(status, stop_reason, last_complete.solution, history, validation)
end
