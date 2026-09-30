This repository implements the sum-of-squares programs from
[*Beyond Ellipsoids: Semi-Algebraic Tightening for Chance Constraints Under
Actuator Saturation*](https://arxiv.org/abs/2607.19639).

The code constructs polynomial Lyapunov certificates for stochastic linear systems subject to unbounded additive disturbances and actuator saturation. It computes finite-time probabilistic reachable sets and probabilistic ultimate bounds (PUBs) from sublevel sets of the Lyapunov function.

## Setup

The checked-in manifest was generated with Julia 1.12. From the repository
root, instantiate the environment with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The paper scripts use [MOSEK](https://www.mosek.com/) through `MosekTools.jl`.
Instantiating the Julia environment installs the solver bindings and runtime,
but a valid MOSEK license must be configured separately before running these
scripts. Academic licenses are available from MOSEK. The test suite uses CSDP
and does not require a MOSEK license.

All remaining Julia dependencies, including JuMP, SumOfSquares,
DynamicPolynomials, Distributions, and CairoMakie, are installed by
`Pkg.instantiate()`.

## Reproducing the examples

Run the three main examples from the repository root:

```bash
julia --project=. scripts/gaussian.jl
julia --project=. scripts/gamma.jl
julia --project=. scripts/asymmetric.jl
```

Each script solves the problem for polynomial half-degrees `r = 1, 2, 4`,
performs the outer search over `λ`, prints the winning `λ` and `ρ`, checks the
numerical SOS certificate, and displays the PUB contours with simulated sample
clouds. The examples can take some time because each value of `λ` requires an
alternating sequence of semidefinite programs.

The `results/` directory contains the reference figures used for the Gaussian,
shifted-Gamma, and asymmetric-input examples:

- `gaussian_pub_paper.png`
- `gamma_pub_paper.png`
- `asymmetric_pub_paper.png`

For a headless run, replace the final `display(fig)` in an example with, for
example:

```julia
save(joinpath(@__DIR__, "..", "results", "gaussian_pub.png"), fig)
```

## Solving a custom problem

A problem is defined by:

- system matrices `A` and `B`;
- a feedback gain `K`, using the convention `u = K * e`;
- state constraints `H * e <= h`;
- actuator bounds `u_min` and `u_max`;
- optional nominal-input bounds `v_min` and `v_max`, with $u_{\min} \leq v_{\min} \leq v_{\max} \leq u_{\max}$; currently `v_min=u_min` and `v_max=u_max` by default.
- a supported disturbance distribution from `Distributions.jl` with the
  required finite moments; multivariate disturbances are specified by stacking
  univariate component distributions, corresponding to their independent
  product measure. Genuine multivariate distributions are not currently supported.

The following is a minimal two-state example:

```julia
using Distributions
using LinearAlgebra
using MosekTools

include("src/system.jl")
include("src/saturation.jl")
include("src/monomials.jl")
include("src/moments.jl")
include("src/drift.jl")
include("src/certificate_validation.jl")
include("src/sos_program.jl")

A = [0.89 0.10; 0.10 0.89]
B = [0.0; 1.0;;]
K = [-0.282 -0.8415]
H = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
h = fill(10.0, 4)
noise = MvNormal(zeros(2), Diagonal(fill(0.1, 2)))

system = SaturatedSystem(
    A, B, K, H, h, noise;
    u_min=[-1.0],
    u_max=[1.0],
    # v_min and v_max default to the actuator bounds.
)

result = solve_sos_program(
    system,
    MosekTools.Optimizer;
    λ=0.98,
    ε=0.2,
    τ_inf=1.0,
    ζ=1e-5,
    γ_floor=1e-5,
    r=2,
    σ_half_degree=2,
    μ_half_degree=1,
    initial_μ=5.0,
    max_iterations=5,
    validation_tolerance=2e-7,
    saturation_formulation=:semialgebraic,    # optional (default setting)
)

if result.solution === nothing
    println("No complete alternating solution: ", result.stop_reason)
else
    println("status = ", result.status)
    println("ρ = ", result.solution.ρ)
    println("γ = ", result.solution.γ)
    println("validated = ", result.validation.passed)
end
```

Here `r` is the half-degree of the Lyapunov polynomial, while
`σ_half_degree` and `μ_half_degree` control the SOS multiplier bases. The
supplied `γ_floor` is a lower bound on the optimized positivity coefficient.
The default `saturation_formulation=:semialgebraic` uses one saturation domain
in `(e, v, u)`; `:pwa` selects the regions-of-saturation piecewise-affine formulation.

`solve_sos_program` solves for a fixed `λ`. To optimize over `λ`, include
`scripts/lambda_search.jl` and follow the `solve_at_λ` pattern used by the main
example scripts.

## Repository layout

- `src/`: system definition, saturation domains, moment and drift calculations,
  SOS program, and certificate validation.
- `scripts/`: reproducible examples, plotting helpers, and $\lambda$ golden-section search.
- `results/`: reference plots and generated experiment data.
- `test/`: unit and solver-integration tests used for development

Because the tests are currently individual Julia files, run all of them with:

```bash
for file in test/*.jl; do julia --project=. "$file" || exit 1; done
```
