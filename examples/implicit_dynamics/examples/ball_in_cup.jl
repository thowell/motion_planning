using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/ball_in_cup/model.jl"))

include(joinpath(pwd(), "models/visualize.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/ball_in_cup/visuals.jl"))

vis = Visualizer()
mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
render(vis)
default_background!(vis)

# Implicit dynamics
h = 0.1
data = dynamics_data(model, h,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
        idx_ineq = idx_ineq,
		z_subset_init = z_subset_init)

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data)

n = model_implicit.n
nq = model.dim.q
m = model.dim.u

# problem setup
T = 21

q0 = zeros(nq)
q0[1] = 0
q0[3] = 0
q0[4] = -pi/2
q0[5] = 0.0
p_pos0 = Array(kinematics_ee(model, q0))
p_pos0[1] += 0.0
p_pos0[3] -= 0.5
q0[8:10] = p_pos0

q1 = copy(q0)

qN = zeros(nq)
qN[1] = 0
qN[3] = 0
qN[4] = -pi/2
qN[5] = 0.

ee_posN = Array(kinematics_ee(model, qN))
p_posN = ee_posN
p_posN[1] += 0.0
p_posN[3] += 0.1
qN[8:10] = p_posN

qD = zeros(nq)
qD[1] = 0
qD[3] = 0
qD[4] = -pi/2
qD[5] = 0.

ee_posD = Array(kinematics_ee(model, qD))
p_posD = ee_posD
# p_posD[1] += 0.5
qD[8:10] = p_posD

x1 = [q1; q1]
xT = [qN; qN]
ū = [0.0 * randn(m) for t = 1:T-1]#[t == 1 ? [0.0, 10.0, 0.0, 0.0, 0.0, -0.0, 0.0] : t == 2 ? [0.0, -10.0, 0.0, -0.0, 0.0, 0.0, 0.0] : [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0] for t = 1:T-1] #u_hist #[-1.0 * gravity_compensation(_model, h, q1, q1) for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)

visualize!(mvis, model, q̄, Δt = h)

# Objective
V = Diagonal(ones(nq))
Q_velocity = 1.0e-3 * [V -V; -V V] ./ h^2.0
Q_track = 1.0 * Diagonal(ones(n))
Q_track_extend = copy(Q_track)
Q_track_extend[8, 8] = 10.0
Q_track_extend[nq + 8, nq + 8] = 10.0
x_track = [qD; qD]
x_track_extend = copy(x_track)
x_track_extend[8] += 0.5
x_track_extend[nq + 8] += 0.5
Q = [t < T ? Q_velocity + (t == 11 ? Q_track_extend : 1.0 * Q_track) : Q_velocity + 1.0 * Q_track for t = 1:T]
q = [t < T ? -2.0 * (t == 11 ? Q_track_extend * x_track_extend : 1.0 * Q_track * x_track) : -2.0 * 1.0 * Q_track * x_track for t = 1:T]
R = [Diagonal(1.0e-3 * ones(m)) for t = 1:T-1]
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
ul = -100.0 * ones(m)
uu = 100.0 * ones(m)
p = [t < T ? (t == 11 ? 2 * m + 0 * nq : 2 * m) : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_m = Dict(:qD => x_track_extend[1:nq], :ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 11 ? info_m : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c[1:2 * m] .= [ul - u; u - uu]
        # if t == 11
        #     c[2 * m .+ (1:nq)] = x[1:nq] - cons.con[t].info[:qD]
        # end
	else
		c[1:n] .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

ū = [t == 1 ? [0.0, 6.5, 0.0, 0.0, 0.0, -0.0, 0.0] : t == 2 ? [0.0, -6.5, 0.0, -0.0, 0.0, 0.0, 0.0] : [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0] for t = 1:T-1] #u_hist #[-1.0 * gravity_compensation(model, h, q1, q1) for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)
visualize!(mvis, model, q̄, Δt = h)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.005)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
visualize!(mvis, model, q̄, Δt = h)
