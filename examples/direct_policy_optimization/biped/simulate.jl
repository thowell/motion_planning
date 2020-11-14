using Plots
include(joinpath(@__DIR__, "dpo.jl"))
include(joinpath(pwd(), "examples/direct_policy_optimization/simulate.jl"))

# Unpack trajectories
x̄, ū = unpack(z̄, prob)
x, u = unpack(z, prob_dpo.prob.prob.nom)
Θ = get_policy(z, prob_dpo)

# Simulation setup
model_sim = model
T_sim = 10 * T

W = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(5.0e-4 * ones(model_sim.n)))
w = rand(W, T_sim)

W0 = Distributions.MvNormal(zeros(model_sim.n),
	Diagonal(5.0e-4 * ones(model_sim.n)))
w0 = rand(W0, 1)

tf_nom = sum([ū[t][end] for t = 1:T-1])
t_nom = range(0, stop = tf_nom, length = T)
t_sim_nom = range(0, stop = tf_nom, length = T_sim)

tf_dpo = sum([u[t][end] for t = 1:T-1])
t_nom_dpo = range(0, stop = tf_dpo, length = T)
t_sim_nom_dpo = range(0, stop = tf_dpo, length = T_sim)

dt_sim_nom = tf_nom / (T_sim - 1)
dt_sim_dpo = tf_dpo / (T_sim - 1)

z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = _simulate(
	model_sim,
	policy, Θ,
	x, u,
	Q, R,
	T_sim, u[1][end],
	vec(x[1] + w0), w,
	_norm = 2,
	ul = ul[1], uu = uu[1],
	u_idx = (1:model.m - 1))

jt = []
jtx = []
jtu = []

jd = []
jdx = []
jdu = []

n_trials = 100
for i = 1:n_trials
	W = Distributions.MvNormal(zeros(model_sim.n),
		Diagonal(δ * ones(model_sim.n)))
	w = rand(W, T_sim)

	W0 = Distributions.MvNormal(zeros(model_sim.n),
		Diagonal(δ0 * ones(model_sim.n)))
	w0 = rand(W0, 1)

	# Simulate
	z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = _simulate(
		model_sim,
		policy, K,
	    x̄, ū,
		Q, R,
		T_sim, ū[1][end],
		vec(x̄[1] + w0), w,
		_norm = 2,
		ul = ul[1], uu = uu[1],
		u_idx = (1:model.m - 1))

	z_dpo, u_dpo, J_dpo, Jx_dpo, Ju_dpo = _simulate(
		model_sim,
		policy, Θ,
	    x, u,
		Q, R,
		T_sim, u[1][end],
		vec(x[1] + w0), w,
		_norm = 2,
		ul = ul[1], uu = uu[1],
		u_idx = (1:model.m - 1))

	push!(jt, J_tvlqr)
	push!(jtx, Jx_tvlqr)
	push!(jtu, Ju_tvlqr)

	push!(jd, J_dpo)
	push!(jdx, Jx_dpo)
	push!(jdu, Ju_dpo)
end

# average tracking performance

mean(jt)
mean(jd)

std(jt)
std(jd)

mean(jtx)
mean(jdx)

std(jtx)
std(jdx)

mean(jtu)
mean(jdu)

std(jtu)
std(jdu)

using RDataSets, StatsPlots
boxplot(["lqr" "dpo"], hcat(jt, jd), title="Tracking error (n = $n_trials)", label=["lqr" "dpo"], color = [:purple :orange])
boxplot(["lqr" "dpo"], hcat(jtx, jdx), title="State tracking error (n = $n_trials)", label=["lqr" "dpo"], color = [:purple :orange])
boxplot(["lqr" "dpo"], hcat(jtu, jdu), title="Control tracking error (n = $n_trials)", label=["lqr" "dpo"], color = [:purple :orange])

# plot(t_nom[1:end-1], hcat(ū...)[1:4,:]', label = "",
# 	color = :purple, width = 2.0)
# plot!(t_sim_nom[1:end-1], hcat(u_tvlqr...)', label = "", color = :black)
#
# plot(t_nom_dpo[1:end-1], hcat(u...)[1:4,:]', label = "",
# 	color = :orange, width = 2.0)
# plot!(t_sim_nom_dpo[1:end-1], hcat(u_dpo...)[1:4,:]', label = "", color = :black)

# Visualize
include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
render(vis)

urdf = joinpath(pwd(), "src/models/biped/urdf/biped_left_pinned.urdf")
mechanism = parse_urdf(urdf, floating=false)
mvis = MechanismVisualizer(mechanism,
    URDFVisuals(urdf, package_path=[dirname(dirname(urdf))]), vis)

visualize!(mvis, model, x, Δt = u[1][end])
visualize!(mvis, model, z_dpo, Δt = dt_sim_dpo)

# visualize!(mvis, model, z_tvlqr, Δt = dt_sim_nom)
# visualize!(mvis, model, z_dpo, Δt = dt_sim_dpo)
