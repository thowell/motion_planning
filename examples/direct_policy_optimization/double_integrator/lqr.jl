include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(pwd(), "models/double_integrator.jl"))

# Horizon
T = 51

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T, 0.0, 0.0)

# Initial and final states
x0 = zeros(model.n)
xl, xu = state_bounds(model, T, x0, x0)

# Problem
prob_nom = trajectory_optimization(
			model,
			EmptyObjective(),
			T,
			xl = xl,
			xu = xu,
			ul = ul,
			uu = uu,
			)

# DPO
N = 2 * model.n
D = 2 * model.d

β = 1.0
δ = 10.0

x1 = resample(ones(model.n), Diagonal(ones(model.n)), β)

# Mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false)

# Sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false,
				xl = state_bounds(model, T, x1 = x1[i])[1],
				xu = state_bounds(model, T, x1 = x1[i])[2]) for i = 1:N]

# Sample objective
Q = [Diagonal(ones(model.n)) for t = 1:T]
R = [Diagonal(ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m)
dist = disturbances([Diagonal(δ * ones(model.d)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

z0 = ones(prob_dpo.num_var)

# Solve
optimize = true

if optimize
	z = solve(prob_dpo, copy(z0),
		tol = 1.0e-8, c_tol = 1.0e-8)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
	@load joinpath(@__DIR__, "sol_dpo.jld2") z
end

# TVLQR policy
A, B = get_dynamics(model)
K, P = tvlqr(
	[A for t = 1:T-1],
	[B for t = 1:T-1],
	Q, R)

# DPO policy
θ = get_policy(z, prob_dpo)

# Policy difference
policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
println("policy difference (inf. norm): $(norm(policy_diff, Inf))")
