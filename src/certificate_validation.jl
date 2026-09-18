using DynamicPolynomials: coefficients
using JuMP
using LinearAlgebra
using SumOfSquares

struct SOSValidationReport
    passed::Bool
    minimum_psd_margin::Float64
    maximum_coefficient_residual::Float64
    failed_checks::Vector{String}
end

_polynomial_coefficients(p::Number) = [p]
_polynomial_coefficients(p) = coefficients(p)

function _polynomial_value(p)
    any(c -> c isa JuMP.AbstractJuMPScalar, _polynomial_coefficients(p)) ? value(p) : p
end

function _maximum_absolute_coefficient(p)
    c = _polynomial_coefficients(p)
    isempty(c) ? 0.0 : maximum(abs, Float64.(c))
end

function _matrix_check(label, matrix, tolerance)
    numeric = Matrix{Float64}(matrix)
    if !all(isfinite, numeric)
        return (passed=false, margin=-Inf, failed=["$label has nonfinite entries"])
    end

    scale = max(1.0, norm(numeric, Inf))
    asymmetry = norm(numeric - transpose(numeric), Inf) / scale
    margin = asymmetry <= tolerance ? eigmin(Symmetric(numeric)) / scale : -Inf
    failed = String[]
    asymmetry <= tolerance || push!(failed, "$label is not numerically symmetric")
    margin >= -tolerance || push!(failed, "$label is not numerically positive semidefinite")
    return (passed=isempty(failed), margin=margin, failed=failed)
end

function _gram_check(label, target, gram, tolerance)
    matrix_check = _matrix_check(label, gram.Q, tolerance)
    target_value = _polynomial_value(target)
    residual = _maximum_absolute_coefficient(target_value - polynomial(gram)) /
               max(1.0, _maximum_absolute_coefficient(target_value))
    failed = copy(matrix_check.failed)
    isfinite(residual) && residual <= tolerance ||
        push!(failed, "$label has an excessive coefficient residual")
    return (
        passed=isempty(failed),
        margin=matrix_check.margin,
        residual=residual,
        failed=failed,
    )
end

function _validate_solution(Q, targets, multipliers; tolerance)
    Q_check = _matrix_check("Q", Q, tolerance)
    minimum_margin = Q_check.margin
    maximum_residual = 0.0
    failed = copy(Q_check.failed)

    for target in targets
        try
            check = _gram_check(
                target.label,
                target.polynomial,
                gram_matrix(target.constraint),
                tolerance,
            )
            minimum_margin = min(minimum_margin, check.margin)
            maximum_residual = max(maximum_residual, check.residual)
            append!(failed, check.failed)
        catch error
            minimum_margin = -Inf
            maximum_residual = Inf
            push!(failed, "$(target.label) could not be checked: $(sprint(showerror, error))")
        end
    end

    for (label, multiplier) in multipliers
        try
            check = _matrix_check(label, value.(multiplier.Q), tolerance)
            minimum_margin = min(minimum_margin, check.margin)
            append!(failed, check.failed)
        catch error
            minimum_margin = -Inf
            push!(failed, "$label could not be checked: $(sprint(showerror, error))")
        end
    end

    return SOSValidationReport(
        isempty(failed),
        minimum_margin,
        maximum_residual,
        failed,
    )
end
