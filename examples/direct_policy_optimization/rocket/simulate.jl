include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
x̄, ū = unpack(z̄_nom, prob_nominal)
x, u = unpack(z, prob_dpo.prob.prob.nom)
Θ = get_policy(z, prob_dpo)

# Simulation setup
model_sim = model_slosh = RocketSlosh{RK3, FixedTime}(n_slosh, m, n_slosh, g, m1 - m2, l1, J, m2, l2, l3)

x1_sim = copy(x1_slosh)
T_sim = 10 * T

using Random
Random.seed!(1)

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
t_dpo = range(0, stop = tf_dpo, length = T)
t_sim_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
z_lqr, u_lqr, J_lqr, Jx_lqr, Ju_lqr = simulate(
	model_sim,
	policy, K,
    x̄, ū,
	Q, R,
	T_sim, ū[1][end],
	z0_sim,
	w,
	_norm = 2,
	ul = ul[1], uu = uu[2],
	u_idx = (1:model_sl.m - 1))

z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = simulate(
	model_sim,
	policy, Θ,
    x, u,
	Q, R,
	T_sim, u[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul[1], uu = uu[1],
	u_idx = (1:model_sl.m - 1))

# state tracking
Jx_lqr
Jx_dpo

# control tracking
Ju_lqr
Ju_dpo

# objective value
J_lqr
J_dpo
