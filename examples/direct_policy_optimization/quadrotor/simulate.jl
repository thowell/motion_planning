include(joinpath(@__DIR__, "dpo.jl"))
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
x̄, ū = unpack(z̄, prob)
x, u = unpack(z, prob)
θ = get_policy(z, prob_dpo)

# Simulation setup
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * T

W = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-5 * ones(model_sim.n)))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(1.0e-5 * ones(model_sim.n)))
w0 = rand(W0, 1)

z0_sim = vec(copy(x1_sim) + w0)

tf_nom = sum([ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([u[t][end] for t = 1:T-1])
t_nom_dpo = range(0, stop = tf_dpo, length = T)
t_sim_nom_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
z_tvlqr1, u_tvlqr1, J_tvlqr1, Jx_tvlqr1, Ju_tvlqr1 = _simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom, uu = uu1,
	u_idx = (1:model.m - 1))

z_tvlqr2, u_tvlqr2, J_tvlqr2, Jx_tvlqr2, Ju_tvlqr2 = _simulate(
	model_sim,
	policy,
	K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom,
	uu = uu2,
	u_idx = (1:model.m - 1))

z_tvlqr3, u_tvlqr3, J_tvlqr3, Jx_tvlqr3, Ju_tvlqr3 = _simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul_nom, uu = uu3,
	u_idx = (1:model.m - 1))

z_tvlqr4, u_tvlqr4, J_tvlqr4, Jx_tvlqr4, Ju_tvlqr4 = _simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu4,
	u_idx = (1:model.m - 1))

z_dpo1, u_dpo1, J_dpo1, Jx_dpo1, Ju_dpo1 = _simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu1,
	u_idx = (1:model.m - 1))

z_dpo2, u_dpo2, J_dpo2, Jx_dpo2, Ju_dpo2 = _simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu2,
	u_idx = (1:model.m - 1))

z_dpo3, u_dpo3, J_dpo3, Jx_dpo3, Ju_dpo3 = _simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu3,
	u_idx = (1:model.m - 1))

z_dpo4, u_dpo4, J_dpo4, Jx_dpo4, Ju_dpo4 = _simulate(
	model_sim,
	policy, θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul_nom, uu = uu4,
	u_idx = (1:model.m - 1))

# state tracking
(Jx_tvlqr1 + Jx_tvlqr2 + Jx_tvlqr3 + Jx_tvlqr4) / 4.0
(Jx_dpo1 + Jx_dpo2 + Jx_dpo3 + Jx_dpo4) / 4.0

# control tracking
(Ju_tvlqr1 + Ju_tvlqr2 + Ju_tvlqr3 + Ju_tvlqr4) / 4.0
(Ju_dpo1 + Ju_dpo2 + Ju_dpo3 + Ju_dpo4) / 4.0

# objective value
(J_tvlqr1 + J_tvlqr2 + J_tvlqr3 + J_tvlqr4) / 4.0
(J_dpo1 + J_dpo2 + J_dpo3 + J_dpo4) / 4.0
