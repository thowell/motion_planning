include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "cartpole_friction.jl"))

# Additive noise model
model_friction = additive_noise_model(model_friction)

function fd(model::CartpoleFriction, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

# Nominal solution
x̄, ū = unpack(z̄_friction, prob_friction)
prob_nom = prob_friction.prob

# DPO
N = 2 * model_friction.n
D = 2 * model_friction.d

β = 1.0
δ = 1.0e-3

# Initial samples
x1_sample = resample(x1, Diagonal(ones(model_friction.n)), 1.0)

# Mean problem
prob_mean = trajectory_optimization(
				model_friction,
				EmptyObjective(),
				T,
				h = h,
				dynamics = false)

# Sample problems
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
				con = con_friction) for i = 1:N]

# Sample objective
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
K, P = tvlqr(model,
	x̄, [ū[t][1:1] for t = 1:T-1],
 	Q, [R[t][1:1, 1:1] for t = 1:T-1],
	h)

z0 = pack(z̄_friction, K, prob_dpo)


# Solve
optimize = true

if optimize
	include_snopt()
	z , info = solve(prob_dpo, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-3, c_tol = 1.0e-3,
		time_limit = 60 * 10)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
	@load joinpath(@__DIR__, "sol_dpo.jld2") z
end
