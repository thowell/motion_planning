include_dpo()

# Model
include_model("double_integrator")

# Horizon
T = 51

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T, 0.0, 0.0)

# Initial and final states
xl, xu = state_bounds(model, T, zeros(model.n), zeros(model.n))

# Problem
prob_nom = trajectory_optimization(
			model,
			EmptyObjective(),
			T,
			xl = xl,
			xu = xu,
			ul = ul,
			uu = uu)

# DPO
β = 1.0
δ = 1.0

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
				xu = state_bounds(model, T, x1 = x1[i])[2]) for i = 1:2 * model.n]

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

z0 = zeros(prob_dpo.num_var)
include_snopt()
z, info = solve(prob_dpo, copy(z0),
	tol = 1.0e-7, c_tol = 1.0e-7,
	nlp = :SNOPT7)

# Solve
if true
	include_snopt()
	z, info = solve(prob_dpo, copy(z0),
		tol = 1.0e-7, c_tol = 1.0e-7,
		nlp = :SNOPT7)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
	@load joinpath(@__DIR__, "sol_dpo.jld2") z
end

# DPO policy
θ = get_policy(z, prob_dpo)

# TVLQR policy
A, B = get_dynamics(model)
K, P = tvlqr(
	[A for t = 1:T-1],
	[B for t = 1:T-1],
	Q, R)

# Policy difference
policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
println("policy difference (inf. norm): $(norm(policy_diff, Inf))")

# Monte Carlo
using Random, Distributions
Random.seed!(1)

function monte_carlo(;M = 5)
	pd = []
	uni_dist = Distributions.Uniform(-1.0, 1.0)
	z_failure = []
	for i = 1:M
		z0 = rand(uni_dist, prob_dpo.num_var)
		z, info = solve(prob_dpo, copy(z0),
			tol = 1.0e-7, c_tol = 1.0e-7,
			nlp = :SNOPT7,
			time_limit = 60)
			# mapl = 5, mipl = 0)

		θ = get_policy(z, prob_dpo)
		policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
		pdi = norm(policy_diff, Inf)

		# if SNOPT fails, try Ipopt
		if pdi > 1.0e-4
			z, info = solve(prob_dpo, copy(z0),
				tol = 1.0e-9, c_tol = 1.0e-9,
				nlp = :ipopt,
				max_iter = 250)
			θ = get_policy(z, prob_dpo)
			policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
			pdi = norm(policy_diff, Inf)

			if pdi > 1.0e-4
				push!(z_failure, (z0, pdi))
			end
		end

		push!(pd, pdi)

	end

	return pd, z_failure
end

M = 1000
pd, z_failure = monte_carlo(M = M)
println("Monte Carlo: M = $M")
println("failures: $(count(pd .> 1.0e-4))")
println("maximum diff.: $(maximum(pd[pd .<= 1.0e-4]))")
println("mean diff.: $(mean(pd[pd .<= 1.0e-4]))")
println("std diff.: $(std(pd[pd .<= 1.0e-4]))")
#
# norm(z_failure[1][1])
# z, info = solve(prob_dpo, copy(z_failure[1][1]),
# 	tol = 1.0e-9, c_tol = 1.0e-9,
# 	nlp = :ipopt,
# 	time_limit = 20)
# θ = get_policy(z, prob_dpo)
# policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
# pdi = norm(policy_diff, Inf)
