include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "car_obstacles.jl"))

# Additive noise model
model = Car(3, 2, 3)

function fd(model::Car, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

# Nominal solution
X̄, Ū = unpack(Z̄, prob)
prob_nom = prob.prob

# DPO
N = 2 * model.n
D = 2 * model.d

α = 1.0e-5
β = 1.0 / (N + D)
γ = 1.0 / (N + D)
δ = 1.0e-5

# initial samples
x1_sample = resample(x1, Diagonal([1.0; 1.0; 0.1]), 1.0e-5)

# mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				dynamics = false
				)

# sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				# xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				# xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = control_bounds(model, T, -3.0, 3.0)[1],
				uu = control_bounds(model, T, -3.0, 3.0)[2],
				dynamics = false,
				con = con_obstacles
				) for i = 1:N]

# sample objective
# Q = [(t < T ? Diagonal([10.0; 10.0; 1.0])
	# : Diagonal(100.0 * ones(model.n))) for t = 1:T]
Q = [Diagonal([10.0; 10.0; 1.0]) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(0, 0)#model.n, model.m)
dist = disturbances([Diagonal(δ * [1.0; 1.0; 0.1]) for t = 1:T-1])
sample = sample_params(α, β, γ, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
K = tvlqr(model, X̄, Ū, Q, R)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄, Ū, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄, Ū, prob_nom)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄, Ū, prob_nom)
end
for j = 1:(N + D)
	z0_dpo[prob_dpo.prob.idx.slack[j]] = vcat(X̄[2:end]...)
end
# for t = 1:T-1
# 	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
# end

# include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

# Solve
z_sol_dpo = solve(prob_dpo, copy(z0_dpo),
	nlp = :ipopt,
	tol = 1.0e-2, c_tol = 1.0e-2, max_iter = 1000,
	time_limit = 180,
	mapl = 5)
