using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/planar_push/model.jl"))

# visualize
include(joinpath(pwd(), "models/visualize.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/planar_push/visuals.jl"))

vis = Visualizer()
render(vis)

# Implicit dynamics
h = 0.1
data = dynamics_data(model, h,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
        idx_ineq = idx_ineq,
		z_subset_init = z_subset_init,
        dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-2,
						κ_init = 0.1,
						diff_sol = true))

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

n = model_implicit.n
m = model_implicit.m

# Problem setup
T = 26

# Initial and final states
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
x1 = [q1; q1]

# x_goal = 0.25
# y_goal = 0.25
x_goal = 0.5
y_goal = 0.5
θ_goal = 0.5 * π
# qT = q1
qT = [x_goal, y_goal, θ_goal, x_goal-r_dim, y_goal-r_dim]
xT = [qT; qT]

# Objective
V = 1.0 * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1])
Q_velocity = [V -V; -V V] ./ h^2.0
_Q_track = [1.0, 1.0, 1.0, 0.1, 0.1]
Q_track = 1.0 * Diagonal([_Q_track; _Q_track])

Q = [t < T ? Q_velocity + Q_track : Q_velocity + 1.0 * Q_track for t = 1:T]
q = [-2.0 * (t == T ? 1.0 : 1.0) * Q_track * xT for t = 1:T]
R = [Diagonal(1.0e-2 * ones(m)) for t = 1:T-1]
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
ul = -5.0 * ones(m)
uu = 5.0 * ones(m)
p = [t < T ? 2 * m : n - 2 * 2 for t = 1:T]
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
		c .= (x - cons.con[T].info[:xT])[collect([(1:3)..., (6:8)...])]
	end
end

q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
x1 = [q0; q1]
ū = [t < 5 ? [0.5; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

x̄ = rollout(model_implicit, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)

# visualize!(vis, model, q̄, ū, Δt = h, r = r_dim)

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
# plot(hcat(ū..., ū[end])', linetype = :steppost)

vis = Visualizer()
render(vis)
open(vis)
visualize!(vis, model, q̄, Δt = h, r = r_dim, r_pusher = 0.25 * r_dim)
default_background!(vis)
settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)

t = 1
id = t
tl = 0.5
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

t = 5
id = t
tl = 0.6
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

t = 10
id = t
tl = 0.7
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

t = 15
id = t
tl = 0.8
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

t = 20
id = t
tl = 0.9
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

t = 26
id = t
tl = 1.0
_create_planar_push!(vis, model,
        r = r_dim,
        r_pusher = 0.25 * r_dim,
        tl = tl,
        i = id)
_set_planar_push!(vis, model, q̄[t], i = id)

box_line_mat = LineBasicMaterial(color=color=RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=10.0)
pusher_line_mat = LineBasicMaterial(color=color=RGBA(51.0 / 255.0, 1.0, 1.0, 1.0), linewidth=10.0)

points_box = Vector{Point{3,Float64}}()
points_pusher = Vector{Point{3,Float64}}()

for (i, xt) in enumerate(x̄)
	push!(points_box, Point(xt[1], xt[2], 0.0))
    push!(points_pusher, Point(xt[4], xt[5], 0.0))
end
setobject!(vis[:box_traj], MeshCat.Line(points_box, box_line_mat))
setobject!(vis[:pusher_traj], MeshCat.Line(points_pusher, pusher_line_mat))
