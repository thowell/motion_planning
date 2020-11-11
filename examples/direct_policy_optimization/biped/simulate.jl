include(joinpath(@__DIR__, "dpo.jl"))
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
X̄, Ū = unpack(Z̄, prob)
X, U = unpack(Z, prob)
Θ = [reshape(Z[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]],
	model.m - 1, model.n) for t = 1:T-1]

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

tf_nom = sum([Ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([U[t][end] for t = 1:T-1])
t_nom_dpo = range(0, stop = tf_dpo, length = T)
t_sim_nom_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

# Simulate
z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = _simulate(
	model_sim,
	policy, K,
    X̄, Ū,
	Q, R,
	T_sim, Ū[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul[1], uu = uu[1],
	u_idx = (1:model.m - 1))

z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = _simulate(
	model_sim,
	policy, Θ,
    X, U,
	Q, R,
	T_sim, U[1][end],
	z0_sim, w,
	_norm = 2,
	ul = ul[1], uu = uu[1],
	u_idx = (1:model.m - 1))

# state tracking
Jx_tvlqr
Jx_dpo

# control tracking
Ju_tvlqr
Ju_dpo

# objective value
J_tvlqr
J_dpo

plot(t_nom[1:end-1], hcat(Ū...)[1:4,:]', label = "",
	color = :purple, width = 2.0)
plot!(t_sim_nom[1:end-1], hcat(u_tvlqr...)', label = "", color = :black)

plot(t_nom_dpo[1:end-1], hcat(U...)[1:4,:]', label = "",
	color = :orange, width = 2.0)
plot!(t_sim_nom_dpo[1:end-1], hcat(u_dpo...)[1:4,:]', label = "", color = :black)
