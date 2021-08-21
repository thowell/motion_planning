using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/double_pendulum/model.jl"))

# build implicit dynamics
h = 0.1

# impact model
data = dynamics_data(model, h,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
        idx_ineq = idx_ineq)

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

# no impact model
# data = dynamics_data(model_no_impact, h,
#         r_no_impact_func, rz_no_impact_func, rθ_no_impact_func,
# 		rz_no_impact_array, rθ_no_impact_array;
#         idx_ineq = collect(1:0),
# 		z_subset_init = zeros(0))
#
# model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model_no_impact.dim.q, model_no_impact.dim.u, 0, data)

# problem setup
T = 51

q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
qT = [π; 0.0]
q_ref = [π; 0.0]

x1 = [q0; q1]
xT = [qT; qT]

n = model_implicit.n
nq = model_implicit.dynamics.m.dim.q
m = model_implicit.m

ū = [1.0e-2 * randn(model_implicit.m) for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)

# Objective
V = Diagonal(ones(nq))
_Q = [V -V; -V V] ./ h^2.0
Q = [t < T ? _Q : _Q for t = 1:T]
q = [-2.0 * Q[t] * zeros(n) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(m)) for t = 1:T-1]
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
ul = [-2.0]
uu = [2.0]
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

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.005)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
q2l = -0.5 * π * ones(length(q̄))
q2u = 0.5 * π * ones(length(q̄))
t = range(0, stop = h * (length(q̄) - 1), length = length(q̄))
plt = plot();
plt = plot!(t, q2l, color = :black, width = 2.0, label = "q2 limit lower")
plt = plot!(t, q2u, color = :black, width = 2.0, label = "q2 limit upper")
plt = plot!(t, hcat(q̄...)', width = 2.0,
	color = [:magenta :orange],
	labels = ["q1" "q2"],
	legend = :topleft,
	xlabel = "time (s)",
	ylabel = "configuration",
	title = "acrobot (w/o joint limits)")

	# title = "acrobot (w/ joint limits)")

# show(plt)
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/acrobot_joint_limits.png")
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/acrobot_no_joint_limits.png")

plot(hcat(ū..., ū[end])', linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 30)

visualize_elbow!(vis, model, q̄, Δt = h)
