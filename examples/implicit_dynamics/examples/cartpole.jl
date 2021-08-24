using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/cartpole/model.jl"))

# build implicit dynamics
h = 0.1

# friction model
data = dynamics_data(model, h,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
		idx_soc = idx_soc,
		z_subset_init = z_subset_init,
        # θ_params = [0.5; 0.5],
        θ_params = [0.35; 0.35],
        # θ_params = [0.25; 0.25],
        # θ_params = [0.1; 0.1],
        # θ_params = [0.01; 0.01],
        # θ_params = [0.001; 0.001],
        dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-1,
						κ_init = 0.1,
						diff_sol = true))

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

# no friction model
data = dynamics_data(model, h,
        r_no_friction_func, rz_no_friction_func, rθ_no_friction_func,
		rz_no_friction_array, rθ_no_friction_array,
        θ_params = [0.0; 0.0],
        dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 0.1,
						κ_init = 0.1,
						diff_sol = true),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 0.1,
						κ_init = 0.1,
						diff_sol = true))

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

T = 26

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
		c .= [ul - u; u - uu]
	else
		c .= x - cons.con[T].info[:xT]
	end
end

ū = [(t == 1 ? -1.0 : 0.0) * ones(m) for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)

q̄ = state_to_configuration(x̄)
# visualize!(vis, s.model, q̄, Δt = h)

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.001)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
v̄ = [(q̄[t+1] - q̄[t]) ./ h for t = 1:length(q̄)-1]

# q̄_friction = q̄
# v̄_friction = v̄
# ū_friction = ū
# q̄_smooth = q̄
# v̄_smooth = v̄
# ū_smooth = ū
# @save "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/cartpole_friction.jld2" q̄_friction v̄_friction ū_friction
# @load "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/cartpole_friction.jld2"
# @save "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/cartpole_smooth.jld2" q̄_smooth v̄_smooth ū_smooth
# @load "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/cartpole_smooth.jld2"

t = range(0, stop = h * (length(q̄) - 1), length = length(q̄))
plt = plot();
plt = plot!(t, hcat(q̄...)', width = 2.0,
	color = [:magenta :orange],
	labels = ["q1" "q2"],
	legend = :topleft,
	xlabel = "time (s)",
	ylabel = "configuration",
	# title = "cartpole (w / o friction)")
	title = "cartpole (w/ friction)")

plt = plot();
plt = plot!(t, hcat(v̄..., v̄[end])', width = 2.0,
	color = [:magenta :orange],
	labels = ["q1" "q2"],
	legend = :topleft,
	xlabel = "time (s)",
	ylabel = "velocity",
	linetype = :steppost,
	# title = "cartpole (w / o friction)")
	title = "cartpole (w/ friction)")
	# title = "acrobot (w/ joint limits)")

# show(plt)
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction.png")
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_no_friction.png")

plot(hcat(ū..., ū[end])', linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 1)
q̄ = state_to_configuration(x̄)
q_anim = [[q̄[1] for t = 1:20]..., q̄..., [q̄[end] for t = 1:20]...]
visualize!(vis, model, q_anim, Δt = h)

using PGFPlots
const PGF = PGFPlots

plt_q1_smooth = PGF.Plots.Linear(t, hcat(q̄_smooth...)[1,:],
	mark="none",style="color=cyan, line width = 2pt, dashed",legendentry="q1")

plt_q2_smooth = PGF.Plots.Linear(t, hcat(q̄_smooth...)[2,:],
	mark="none",style="color=orange, line width = 2pt, dashed",legendentry="q2")

plt_qd1_smooth = PGF.Plots.Linear(t, hcat(v̄_smooth..., v̄_smooth[end])[1,:],
	mark="none",style="const plot, color=magenta, line width = 2pt, dashed",legendentry="q1")

plt_qd2_smooth = PGF.Plots.Linear(t, hcat(v̄_smooth..., v̄_smooth[end])[2,:],
	mark="none",style="const plot, color=green, line width = 2pt, dashed",legendentry="q2")

plt_q1_friction = PGF.Plots.Linear(t, hcat(q̄_friction...)[1,:],
	mark="none",style="color=cyan, line width = 2pt",legendentry="q1 (friction)")

plt_q2_friction = PGF.Plots.Linear(t, hcat(q̄_friction...)[2,:],
	mark="none",style="color=orange, line width = 2pt",legendentry="q2 (friction)")

plt_qd1_friction = PGF.Plots.Linear(t, hcat(v̄_friction..., v̄_friction[end])[1,:],
	mark="none",style="const plot, color=magenta, line width = 2pt",legendentry="q1 (friction)")

plt_qd2_friction = PGF.Plots.Linear(t, hcat(v̄_friction..., v̄_friction[end])[2,:],
	mark="none",style="const plot, color=green, line width = 2pt",legendentry="q2 (friction)")

aq = Axis([plt_q1_friction; plt_q2_friction; plt_q1_smooth; plt_q2_smooth],#; plt_qd1_smooth; plt_qd1_friction; plt_qd2_smooth; plt_qd2_friction],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="configuration",
	xlabel="time (s)",
	xlims=(0.0, 2.5),
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

av = Axis([plt_qd1_friction; plt_qd2_friction; plt_qd1_smooth; plt_qd2_smooth],
    axisEqualImage=false,
    hideAxis=false,
	ylabel="velocity",
	xlabel="time (s)",
	legendStyle="{at={(0.01,0.99)},anchor=north west}")

# Save to tikz format
PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_configuration.tikz", aq, include_preamble=false)
PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/cartpole_friction_velocity.tikz", av, include_preamble=false)
