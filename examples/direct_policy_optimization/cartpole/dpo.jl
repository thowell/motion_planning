include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "cartpole_friction.jl"))

# Additive noise model
model_friction = CartpoleFriction(4, 7, 4, 1.0, 0.2, 0.5, 9.81, μ0)

function fd(model::CartpoleFriction, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

# Nominal solution
X̄, Ū = unpack(Z̄_friction, prob_friction)
prob_nom = prob_friction.prob

# DPO
N = 2 * model_friction.n
D = 2 * model_friction.d

β = 1.0
δ = 1.0e-2

# initial samples
x1_sample = resample(x1, Diagonal(ones(model_friction.n)), 1.0)

# mean problem
prob_mean = trajectory_optimization(
				model_friction,
				EmptyObjective(),
				T,
				h = h,
				dynamics = false
				)

# sample problems
prob_sample = [trajectory_optimization(
				model_friction,
				EmptyObjective(),
				T,
				h = h,
				xl = state_bounds(model_friction, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model_friction, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_friction
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal(10.0 * ones(model_friction.n))
    : Diagonal(100.0 * ones(model_friction.n))) for t = 1:T]
R = [Diagonal([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, 1)
dist = disturbances([Diagonal(δ * ones(model_friction.n)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K = tvlqr(model, X̄, [Ū[t][1:1] for t = 1:T-1],
 	Q, [R[t][1:1, 1:1] for t = 1:T-1], h)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄, Ū, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄, Ū, prob_nom)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄, Ū, prob_nom)
end
for j = 1:(N + D)
	z0_dpo[prob_dpo.prob.idx.slack[j]] = vcat(X̄[2:end]...)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
end

include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

# Solve
Z = solve(prob_dpo, copy(z0_dpo),
	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3,
	# max_iter = 1000)
	time_limit = 600)
