using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/cartpole/model.jl"))

# build implicit dynamics
h = 0.05
T = 51

# friction model
data = dynamics_data(model, h, T,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
		idx_soc = idx_soc,
		z_subset_init = z_subset_init,
        diff_idx = 3,
        # θ_params = [1.0; 1.0], # fails to reach constraint tolerance
        # θ_params = [0.5; 0.5], # fails to reach constraint tolerance
        θ_params = [0.35; 0.35],
        # θ_params = [0.25; 0.25],
        # θ_params = [0.1; 0.1],
        # θ_params = [0.01; 0.01],
        opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 1.0,
						diff_sol = false))

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

# no friction model
# data = dynamics_data(model, h,
#         r_no_friction_func, rz_no_friction_func, rθ_no_friction_func,
# 		rz_no_friction_array, rθ_no_friction_array,
#         θ_params = [0.0; 0.0],
#         dyn_opts =  InteriorPointOptions{Float64}(
# 						r_tol = 1.0e-8,
# 						κ_tol = 0.1,
# 						κ_init = 0.1,
# 						diff_sol = true),
# 		jac_opts =  InteriorPointOptions{Float64}(
# 						r_tol = 1.0e-8,
# 						κ_tol = 0.1,
# 						κ_init = 0.1,
# 						diff_sol = true))
#
# model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

n = model_implicit.n
nq = model_implicit.dynamics.m.dim.q
m = model_implicit.m

q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
qT = [0.0; π]
q_ref = [0.0; π]

x1 = [q1; q1]
xT = [qT; qT]

# Objective
V = 1.0 * Diagonal(ones(nq))
Q_velocity = [V -V; -V V] ./ h^2.0
Q_track = 1.0 * Diagonal(ones(2 * nq))

Q = [t < T ? Q_velocity + Q_track : Q_velocity + 1.0 * Q_track for t = 1:T]
q = [-2.0 * (t == T ? 1.0 : 1.0) * Q_track * xT for t = 1:T]
R = [Diagonal(1.0 * ones(m)) for t = 1:T-1]
r = [zeros(m) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
ul = [-10.0]
uu = [10.0]
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		# c .= [ul - u; u - uu]
	else
		c .= x - cons.con[T].info[:xT]
	end
end

# ū = [(t == 1 ? -1.0 : 0.0) * ones(m) for t = 1:T-1]
ū = [(t == 1 ? 1.0 : 0.0) * ones(m) for t = 1:T-1] # initialize no friciton model this direction (not sure why it goes the oppostive direction...)
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)

q̄ = state_to_configuration(x̄)
# visualize!(vis, s.model, q̄, Δt = h)

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time stats = constrained_ddp_solve!(prob,
	max_iter = 1000,
    max_al_iter = 10,
	ρ_init = 1.0,
    ρ_scale = 10.0,
    grad_tol = 1.0e-3,
	con_tol = 0.001)

@show ilqr_iterations(stats)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)
b = [data.z_cache[t][collect([model.dim.q + 2, model.dim.q + nc + 2])] for t = 1:T-1]

# x_01, u_01, b_01 = x, u, b
# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_01.jld2") x_01 u_01 b_01
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_01.jld2") x_01 u_01 b_01

# x_1, u_1, b_1 = x, u, b
# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_1.jld2") x_1 u_1 b_1
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_1.jld2") x_1 u_1 b_1

# x_25, u_25, b_25 = x, u, b
# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_25.jld2") x_25 u_25 b_25
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_25.jld2") x_25 u_25 b_25

# x_35, u_35, b_35 = x, u, b
# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_35.jld2") x_35 u_35 b_35
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/cartpole_35.jld2") x_35 u_35 b_35

# using Plots
# t = range(0, stop = h * (T - 1), length = T)
# plot(t, hcat(b_01..., b_01[end])[1, :], linetype = :steppost, color = :magenta, width = 2.0, labels = "")
# plot!(t, hcat(b_1..., b_1[end])[1, :], linetype = :steppost, color = :orange, width = 2.0, labels = "")
# plot!(t, hcat(b_25..., b_25[end])[1, :], linetype = :steppost, color = :green, width = 2.0, labels = "")
# plot!(t, hcat(b_35..., b_35[end])[1, :], linetype = :steppost, color = :cyan, width = 2.0, labels = "")
#
# plot(t, hcat(b_01..., b_01[end])[2, :], linetype = :steppost, color = :magenta, width = 2.0, labels = "")
# plot!(t, hcat(b_1..., b_1[end])[2, :], linetype = :steppost, color = :orange, width = 2.0, labels = "")
# plot!(t, hcat(b_25..., b_25[end])[2, :], linetype = :steppost, color = :green, width = 2.0, labels = "")
# plot!(t, hcat(b_35..., b_35[end])[2, :], linetype = :steppost, color = :cyan, width = 2.0, labels = "")
#
# using PGFPlots
# const PGF = PGFPlots
#
# plt_b_01_s = PGF.Plots.Linear(t, hcat(b_01..., b_01[end])[1, :],
# 	mark="none",style="const plot, color=magenta, line width = 2pt",legendentry="0.01")
#
# plt_b_1_s = PGF.Plots.Linear(t, hcat(b_1..., b_1[end])[1, :],
# 	mark="none",style="const plot, color=orange, line width = 2pt",legendentry="0.1")
#
# plt_b_25_s = PGF.Plots.Linear(t, hcat(b_25..., b_25[end])[1, :],
# 	mark="none",style="const plot, color=green, line width = 2pt",legendentry="0.25")
#
# plt_b_35_s = PGF.Plots.Linear(t, hcat(b_35..., b_35[end])[1, :],
# 	mark="none",style="const plot, color=cyan, line width = 2pt",legendentry="0.35")
#
# plt_b_01_a = PGF.Plots.Linear(t, hcat(b_01..., b_01[end])[2, :],
# 	mark="none",style="const plot, color=magenta, line width = 2pt",legendentry="0.01")
#
# plt_b_1_a = PGF.Plots.Linear(t, hcat(b_1..., b_1[end])[2, :],
# 	mark="none",style="const plot, color=orange, line width = 2pt",legendentry="0.1")
#
# plt_b_25_a = PGF.Plots.Linear(t, hcat(b_25..., b_25[end])[2, :],
# 	mark="none",style="const plot, color=green, line width = 2pt",legendentry="0.25")
#
# plt_b_35_a = PGF.Plots.Linear(t, hcat(b_35..., b_35[end])[2, :],
# 	mark="none",style="const plot, color=cyan, line width = 2pt",legendentry="0.35")
#
# a_s = Axis([plt_b_01_s, plt_b_1_s, plt_b_25_s, plt_b_35_s],
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="friction",
# 	xlabel="time (s)",
# 	# xlim=(0.0, 5.0),
# 	legendStyle="{at={(0.01,0.99)},anchor=north west}")
#
# a_a = Axis([plt_b_01_a, plt_b_1_a, plt_b_25_a, plt_b_35_a],
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="configuration",
# 	xlabel="time (s)",
# 	# xlim=(0.0, 5.0),
# 	legendStyle="{at={(0.01,0.99)},anchor=north west}")
#
# PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_slider.tikz", a_s, include_preamble=false)
# PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_angle.tikz", a_a, include_preamble=false)


plot(hcat(b..., b[end])', linetype = :steppost)
data.θ_params[1] * (model.mp + model.mc) * model.g * h[1]
data.θ_params[1] * (model.mp * model.g * model.l) * h[1]

q̄ = state_to_configuration(x̄)
v̄ = [(q̄[t+1] - q̄[t]) ./ h for t = 1:length(q̄)-1]


# t = range(0, stop = h * (length(q̄) - 1), length = length(q̄))
# plt = plot();
# plt = plot!(t, hcat(q̄...)', width = 2.0,
# 	color = [:magenta :orange],
# 	labels = ["q1" "q2"],
# 	legend = :topleft,
# 	xlabel = "time (s)",
# 	ylabel = "configuration",
# 	# title = "cartpole (w / o friction)")
# 	title = "cartpole (w/ friction)")
#
# plt = plot();
# plt = plot!(t, hcat(v̄..., v̄[end])', width = 2.0,
# 	color = [:magenta :orange],
# 	labels = ["q1" "q2"],
# 	legend = :topleft,
# 	xlabel = "time (s)",
# 	ylabel = "velocity",
# 	linetype = :steppost,
# 	# title = "cartpole (w / o friction)")
# 	title = "cartpole (w/ friction)")
# 	# title = "acrobot (w/ joint limits)")

# show(plt)
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction.png")
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_no_friction.png")

# plot(hcat(ū..., ū[end])', linetype = :steppost)
# #
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, q̄, Δt = h)
open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
setvisible!(vis["/Grid"], false)
#
# # q̄ = state_to_configuration(x̄)
# # q_anim = [[q̄[1] for t = 1:20]..., q̄..., [q̄[end] for t = 1:20]...]
# # visualize!(vis, model, q_anim, Δt = h)
#
# # ghost
# limit_color = [0.0, 0.0, 0.0]
# # limit_color = [0.0, 1.0, 0.0]
#
# t = 1
# id = t
# tl = 0.05
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 5
# id = t
# tl = 0.15
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 10
# id = t
# tl = 0.25
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 15
# id = t
# tl = 0.35
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 20
# id = t
# tl = 0.45
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 25
# id = t
# tl = 0.55
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 30
# id = t
# tl = 0.65
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 35
# id = t
# tl = 0.75
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 40
# id = t
# tl = 0.85
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# t = 45
# id = t
# tl = 0.95
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
#
# t = 51
# id = t
# tl = 1.0
# _create_cartpole!(vis, model;
#         tl = tl,
#         color = RGBA(limit_color..., tl),
#         i = id)
# _set_cartpole!(vis, model, x[t], i = id)
#
# line_mat = LineBasicMaterial(color=color=RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=10.0)
# # line_mat = LineBasicMaterial(color=color=RGBA(51.0 / 255.0, 1.0, 1.0, 1.0), linewidth=10.0)
#
# points = Vector{Point{3,Float64}}()
# for (i, xt) in enumerate(x̄)
#     k = kinematics(model, xt)
# 	push!(points, Point(k[1], 0.0, k[2]))
#
#     setobject!(vis["ee_vertex_$i"], Sphere(Point3f0(0),
#         convert(Float32, 0.001)),
#         MeshPhongMaterial(color = RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0)))
#         settransform!(vis["ee_vertex_$i"], Translation(points[i]))
# end
# setobject!(vis[:ee_traj], MeshCat.Line(points, line_mat))
#
#
# # using PGFPlots
# # const PGF = PGFPlots
# #
# # plt_q1_smooth = PGF.Plots.Linear(t, hcat(q̄_smooth...)[1,:],
# # 	mark="none",style="color=cyan, line width = 2pt, dashed",legendentry="q1")
# #
# # plt_q2_smooth = PGF.Plots.Linear(t, hcat(q̄_smooth...)[2,:],
# # 	mark="none",style="color=orange, line width = 2pt, dashed",legendentry="q2")
# #
# # plt_qd1_smooth = PGF.Plots.Linear(t, hcat(v̄_smooth..., v̄_smooth[end])[1,:],
# # 	mark="none",style="const plot, color=magenta, line width = 2pt, dashed",legendentry="q1")
# #
# # plt_qd2_smooth = PGF.Plots.Linear(t, hcat(v̄_smooth..., v̄_smooth[end])[2,:],
# # 	mark="none",style="const plot, color=green, line width = 2pt, dashed",legendentry="q2")
# #
# # plt_q1_friction = PGF.Plots.Linear(t, hcat(q̄_friction...)[1,:],
# # 	mark="none",style="color=cyan, line width = 2pt",legendentry="q1 (friction)")
# #
# # plt_q2_friction = PGF.Plots.Linear(t, hcat(q̄_friction...)[2,:],
# # 	mark="none",style="color=orange, line width = 2pt",legendentry="q2 (friction)")
# #
# # plt_qd1_friction = PGF.Plots.Linear(t, hcat(v̄_friction..., v̄_friction[end])[1,:],
# # 	mark="none",style="const plot, color=magenta, line width = 2pt",legendentry="q1 (friction)")
# #
# # plt_qd2_friction = PGF.Plots.Linear(t, hcat(v̄_friction..., v̄_friction[end])[2,:],
# # 	mark="none",style="const plot, color=green, line width = 2pt",legendentry="q2 (friction)")
# #
# # aq = Axis([plt_q1_friction; plt_q2_friction; plt_q1_smooth; plt_q2_smooth],#; plt_qd1_smooth; plt_qd1_friction; plt_qd2_smooth; plt_qd2_friction],
# #     axisEqualImage=false,
# #     hideAxis=false,
# # 	ylabel="configuration",
# # 	xlabel="time (s)",
# # 	xlims=(0.0, 2.5),
# # 	legendStyle="{at={(0.01,0.99)},anchor=north west}")
# #
# # av = Axis([plt_qd1_friction; plt_qd2_friction; plt_qd1_smooth; plt_qd2_smooth],
# #     axisEqualImage=false,
# #     hideAxis=false,
# # 	ylabel="velocity",
# # 	xlabel="time (s)",
# # 	legendStyle="{at={(0.01,0.99)},anchor=north west}")
# #
# # # Save to tikz format
# # PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_configuration.tikz", aq, include_preamble=false)
# # PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_velocity.tikz", av, include_preamble=false)
