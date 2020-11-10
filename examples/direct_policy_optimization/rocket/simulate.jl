include(joinpath(@__DIR__, "dpo.jl"))
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
X̄, Ū = unpack(Z̄, prob_nominal)
X, U = unpack(Z, prob_nominal)
Θ = [reshape(Z[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]],
	model_sl.m - 1, model_sl.n - 2) for t = 1:T-1]

# Simulation setup
model_sim = model_sl
x1_sim = copy(x1_slosh)
T_sim = 10 * T

W = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-5 * ones(model_sim.n)))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-5 * ones(model_sim.n)))
w0 = rand(W0, 1)

z0_sim = vec(copy(x1_sim) + w0)

tf_nom = sum([Ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([U[t][end] for t = 1:T-1])
t_dpo = range(0, stop = tf_dpo, length = T)
t_sim_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = _simulate(
	model_sim,
	policy, K,
    X̄, Ū,
	Q, R,
	T_sim, Ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul[1], uu = uu[2],
	u_idx = (1:model_sl.m - 1))

z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = _simulate(
	model_sim,
	policy, Θ,
    X, U,
	Q, R,
	T_sim, U[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul[1], uu = uu[1],
	u_idx = (1:model_sl.m - 1))

# state tracking
Jx_tvlqr
Jx_dpo

# control tracking
Ju_tvlqr
Ju_dpo

# objective value
J_tvlqr
J_dpo
