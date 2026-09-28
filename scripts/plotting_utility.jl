using CairoMakie
using Distributions
using Random

function eval_V(e, Q, exponents)
    Φ = [prod(e[k]^α[k] for k in eachindex(e)) for α in exponents]
    return sum(Q[i, j] * Φ[i] * Φ[j] for i in eachindex(Φ), j in eachindex(Φ))
end

function error_samples(system; t=100, n_samples=1_000, v=zeros(size(system.B, 2)))
    n = size(system.A, 1)
    length(v) == size(system.B, 2) || throw(DimensionMismatch(
        "v must have length $(size(system.B, 2))",
    ))
    all((system.v_min .<= v) .& (v .<= system.v_max)) || throw(ArgumentError(
        "v must lie within the nominal input bounds",
    ))
    samples = zeros(n, n_samples)

    for sample in axes(samples, 2)
        e = zeros(n)
        for _ in 1:t
            w = if system.noise isa AbstractVector
                rand.(system.noise)
            elseif system.noise isa UnivariateDistribution
                [rand(system.noise)]
            else
                rand(system.noise)
            end
            e .= saturated_error_dynamics(system, e, v) + w
        end
        samples[:, sample] = e
    end

    return samples
end

function plot_pub(
        result;
        R=12.0,
        grid_n=301,
        system=nothing,
        t=100,
        n_samples=1_000,
        v=nothing,
        sample_at_bounds=false,
    )
    results = result isa AbstractArray ? result : [result]
    e1_axis = range(-R, R; length=grid_n)
    e2_axis = range(-R, R; length=grid_n)

    fig = Figure(size=(760, 660), fontsize=18)
    ax = Axis(
        fig[1, 1];
        xlabel="e₁",
        ylabel="e₂",
        aspect=DataAspect(),
        title="PUB contour",
    )
    for result in results
        solution = isnothing(result) ? nothing :
                   (hasproperty(result, :solution) ? result.solution : result)
        if isnothing(solution)
            continue
        end
        values = [
            eval_V((e1, e2), solution.Q, solution.basis_exponents) / solution.τ_inf
            for e1 in e1_axis, e2 in e2_axis
        ]
        contour!(ax, e1_axis, e2_axis, values; levels=[1.0], linewidth=3)
    end
    if system !== nothing
        nominal_inputs = sample_at_bounds ?
                         [zeros(size(system.B, 2)), system.v_max, system.v_min] :
                         [isnothing(v) ? zeros(size(system.B, 2)) : v]
        colors = sample_at_bounds ? [:black, :magenta3, :cyan3] : [:dodgerblue]
        labels = sample_at_bounds ? ["v = 0", "v = v̄", "v = v̲"] : [nothing]
        for (nominal_input, color, label) in zip(nominal_inputs, colors, labels)
            samples = error_samples(system; t, n_samples, v=nominal_input)
            scatter!(
                ax,
                samples[1, :],
                samples[2, :];
                color=(color, 0.16),
                markersize=4,
                label,
            )
        end
        sample_at_bounds && axislegend(ax; position=:rt)
    end
    xlims!(ax, -R, R)
    ylims!(ax, -R, R)
    return fig
end
