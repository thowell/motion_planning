using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/double_pendulum/model.jl"))

# build implicit dynamics
h = 0.05

# impact model
data = dynamics_data(model, h,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
        idx_ineq = idx_ineq,
        z_subset_init = 0.1 * ones(4),
        dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = false),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-3,
						κ_init = 0.1,
						diff_sol = true))

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

# no impact model
# data = dynamics_data(model_no_impact, h,
#         r_no_impact_func, rz_no_impact_func, rθ_no_impact_func,
# 		rz_no_impact_array, rθ_no_impact_array;
#         idx_ineq = collect(1:0),
# 		z_subset_init = zeros(0),
#         dyn_opts =  InteriorPointOptions{Float64}(
# 						r_tol = 1.0e-8,
# 						κ_tol = 0.1,
# 						κ_init = 0.1,
# 						diff_sol = false),
# 		jac_opts =  InteriorPointOptions{Float64}(
# 						r_tol = 1.0e-8,
# 						κ_tol = 0.1,
# 						κ_init = 0.1,
# 						diff_sol = true))
#
# model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model_no_impact.dim.q, model_no_impact.dim.u, 0, data)

# problem setup
T = 101

q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
qT = [π; 0.0]
q_ref = [π; 0.0]

x1 = [q0; q1]
xT = [qT; qT]

n = model_implicit.n
nq = model_implicit.dynamics.m.dim.q
m = model_implicit.m

ū = [1.0e-3 * randn(model_implicit.m) for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)

# Objective
V = 0.1 * Diagonal(ones(nq))
_Q = [V -V; -V V] ./ h^2.0
Q = [t < T ? _Q : _Q for t = 1:T]
q = [-2.0 * Q[t] * zeros(n) for t = 1:T]
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
ul = [-1.0]
uu = [1.0]
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
include(joinpath(pwd(), "examples/implicit_dynamics/models/double_pendulum/visuals.jl"))
vis = Visualizer()
render(vis)
open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 30)
setvisible!(vis["/Grid"], false)

# visualize_elbow!(vis, model, q̄, Δt = h)

# ghost
limit_color = [1.0, 0.0, 0.0]
# limit_color = [0.0, 1.0, 0.0]

t = 1
id = t
tl = 0.05
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 10
id = t
tl = 0.15
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 20
tl = 0.25
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 30
id = t
tl = 0.35
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 40
id = t
tl = 0.45
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 50
id = t
tl = 0.55
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 60
id = t
tl = 0.65
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 70
id = t
tl = 0.75
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 80
id = t
tl = 0.85
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 90
id = t
tl = 0.95
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

t = 101
id = t
tl = 1.0
_create_acrobot!(vis, model;
        tl = tl,
        limit_color = RGBA(limit_color..., tl),
        i = id)
_set_acrobot!(vis, model, x[t], i = id)

# line_mat = LineBasicMaterial(color=color=RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=10.0)
line_mat = LineBasicMaterial(color=color=RGBA(51.0 / 255.0, 1.0, 1.0, 1.0), linewidth=10.0)

points = Vector{Point{3,Float64}}()
for (i, xt) in enumerate(x̄)
    k = kinematics(model, xt)
	push!(points, Point(k[1], 0.0, k[2]))

    setobject!(vis["ee_vertex_$i"], Sphere(Point3f0(0),
        convert(Float32, 0.001)),
        MeshPhongMaterial(color = RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0)))
        settransform!(vis["ee_vertex_$i"], Translation(points[i]))
end
setobject!(vis[:ee_traj], MeshCat.Line(points, line_mat))
