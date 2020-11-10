include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(@__DIR__, "rocket_nominal.jl"))
include(joinpath(@__DIR__, "rocket_slosh.jl"))

# DPO
N = 2 * model_sl.n
D = 2 * model_sl.d

β = 1.0
δ = 1.0e-3

# initial samples
x1_sample = resample(x1_slosh, Diagonal(ones(model_sl.n)), 1.0e-3)

prob_nom = prob_nominal.prob

# mean problem
prob_mean = trajectory_optimization(
				model_sl,
				EmptyObjective(),
				T,
				dynamics = false,
				ul = control_bounds(model, T, [Inf * ones(2); 0.0], [Inf * ones(2); 0.0])[1],
				uu = control_bounds(model, T, [Inf * ones(2); 0.0], [Inf * ones(2); 0.0])[2],
				)

# sample problems
prob_sample = [trajectory_optimization(
				model_sl,
				EmptyObjective(),
				T,
				xl = state_bounds(model_sl, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model_sl, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_free_time
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal(100.0 * ones(model_nom.n))
	: Diagonal(1000.0 * ones(model_nom.n))) for t = 1:T]
R = [Diagonal([1.0; 1.0; 100.0]) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(6, 2,
	idx_input = collect([1, 2, 3, 5, 6, 7]),
	idx_input_nom = (1:6),
	idx_output = (1:2))
dist = disturbances([Diagonal(δ * ones(model_sl.n)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

# TVLQR policy
X̄_nom, Ū_nom = unpack(Z̄, prob_nominal)
X̄_slosh, Ū_slosh = unpack(Z̄_slosh, prob_slosh)
K = tvlqr(model_nom, X̄_nom, Ū_nom, Q, R, 0.0)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = copy(Z̄) #+ 0.001 * rand(prob_nominal.num_var)
z0_dpo[prob_dpo.prob.idx.mean] = copy(Z̄_slosh) # + 0.001 * rand(prob_slosh.num_var)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = copy(Z̄_slosh) #+ 0.001 * rand(prob_slosh.num_var)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))# + 0.001 * rand(length(vec(K[t])))
end

include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

# Solve
Z = solve(prob_dpo, copy(z0_dpo),
	nlp = :SNOPT7,
	tol = 5.0e-2, c_tol = 5.0e-2,
	time_limit = 60 * 60)
