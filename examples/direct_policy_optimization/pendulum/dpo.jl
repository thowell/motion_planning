include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "pendulum_minimum_time.jl"))

# Nominal solution
X̄, Ū = unpack(Z̄, prob)
prob_nom = prob.prob

# DPO
N = 2 * model_ft.n
D = 2 * model_ft.d

β = 1.0
δ = 5.0e-3

# initial samples
x1_sample = resample(x1, Diagonal([1.0; 1.0]), 1.0e-3)

# mean problem
prob_mean = trajectory_optimization(
				model_ft,
				EmptyObjective(),
				T,
				ul = control_bounds(model_ft, T, [Inf; 0.0], [Inf; 0.0])[1],
				uu = control_bounds(model_ft, T, [Inf; 0.0], [Inf; 0.0])[2],
				dynamics = false
				)

# sample problems
prob_sample = [trajectory_optimization(
				model_ft,
				EmptyObjective(),
				T,
				xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_free_time,
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal(10.0 * ones(model_ft.n))
	: Diagonal(100.0 * ones(model_ft.n))) for t = 1:T]
R = [Diagonal(1.0e-1 * [1.0; 10.0]) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model_ft.n, model_ft.m - 1)
dist = disturbances([Diagonal(δ * ones(model_ft.d)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K = tvlqr(model_ft, X̄, Ū, Q, R, 0.0)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄, Ū, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄, Ū, prob_nom)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄, Ū, prob_nom)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
end

include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

# Solve
Z = solve(prob_dpo, copy(z0_dpo),
	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3, #max_iter = 1000,
	time_limit = 180,
	mapl = 5)

# NOTE: doesn't work as well with Ipopt
