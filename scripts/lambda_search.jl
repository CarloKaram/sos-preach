function golden_section_search(solve_at_λ, λ_min, λ_max; tolerance=0.002)
    start_time = time()
    golden_ratio = (sqrt(5) - 1) / 2
    evaluations = NamedTuple[]

    function evaluate(λ)
        result = solve_at_λ(λ)
        ρ = result.solution === nothing ? Inf : result.solution.ρ
        push!(evaluations, (λ=λ, ρ=ρ, status=result.status, timing=result.timing))
        return result, ρ
    end

    λ_left = λ_max - golden_ratio * (λ_max - λ_min)
    λ_right = λ_min + golden_ratio * (λ_max - λ_min)

    result_left, ρ_left = evaluate(λ_left)
    result_right, ρ_right = evaluate(λ_right)

    best_λ = nothing
    best_ρ = Inf
    best_result = nothing
    if ρ_left < best_ρ
        best_λ, best_ρ, best_result = λ_left, ρ_left, result_left
    end
    if ρ_right < best_ρ
        best_λ, best_ρ, best_result = λ_right, ρ_right, result_right
    end

    while λ_max - λ_min > tolerance
        if ρ_left <= ρ_right
            λ_max = λ_right
            λ_right, result_right, ρ_right = λ_left, result_left, ρ_left
            λ_left = λ_max - golden_ratio * (λ_max - λ_min)
            result_left, ρ_left = evaluate(λ_left)

            if ρ_left < best_ρ
                best_λ, best_ρ, best_result = λ_left, ρ_left, result_left
            end
        else
            λ_min = λ_left
            λ_left, result_left, ρ_left = λ_right, result_right, ρ_right
            λ_right = λ_min + golden_ratio * (λ_max - λ_min)
            result_right, ρ_right = evaluate(λ_right)

            if ρ_right < best_ρ
                best_λ, best_ρ, best_result = λ_right, ρ_right, result_right
            end
        end
    end

    total_time = time() - start_time
    λ_evaluation_time = sum(evaluation.timing.total_time for evaluation in evaluations)
    return (
        λ=best_λ,
        ρ=best_ρ,
        result=best_result,
        evaluations=evaluations,
        timing=(
            total_time=total_time,
            λ_evaluation_time=λ_evaluation_time,
            overhead_time=total_time - λ_evaluation_time,
        ),
    )
end

function print_timing(winner)
    evaluations = winner.evaluations
    setup_time = sum(evaluation.timing.setup_time for evaluation in evaluations)
    q_time = sum(evaluation.timing.q_time for evaluation in evaluations)
    μ_time = sum(evaluation.timing.μ_time for evaluation in evaluations)
    alternation_time = sum(evaluation.timing.alternation_time for evaluation in evaluations)
    validation_time = sum(evaluation.timing.validation_time for evaluation in evaluations)

    println("  timing total:          ", winner.timing.total_time, " s")
    println("  λ evaluations:         ", length(evaluations))
    println("  setup:                 ", setup_time, " s")
    println("  Q steps:               ", q_time, " s")
    println("  μ steps:               ", μ_time, " s")
    println("  alternations:          ", alternation_time, " s")
    println("  validation:            ", validation_time, " s")
    println("  optimization:          ", winner.timing.total_time - validation_time, " s")
    if winner.result !== nothing
        println("  winning λ total:       ", winner.result.timing.total_time, " s")
        println("  winning λ validation:  ", winner.result.timing.validation_time, " s")
    end
end
