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
        # idx_soc = idx_soc,
		z_subset_init = z_subset_init)

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

n = model_implicit.n
m = model_implicit.m

# Problem setup
T = 26

# Initial and final states
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
x1 = [q1; q1]

x_goal = 1.0
y_goal = 0.0
θ_goal = 0.0 * π
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
		c .= [ul - u; u - uu]
	else
		c .= (x - cons.con[T].info[:xT])[collect([(1:3)..., (6:8)...])]
	end
end

q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
x1 = [q0; q1]
ū = [t < 5 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

x̄ = rollout(model_implicit, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)

visualize!(vis, model, q̄, ū, Δt = h, r = r_dim)

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 5,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.01)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
v̄ = [(q̄[t+1] - q̄[t]) ./ h for t = 1:length(q̄)-1]

# vis = Visualizer()
# render(vis)
open(vis)
visualize!(vis, model, q̄, ū, Δt = h, r = r_dim)

q̄[end][1]

plot(hcat(ū..., ū[end])', linetype = :steppost)
