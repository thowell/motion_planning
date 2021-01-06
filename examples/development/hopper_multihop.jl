# Model
include_model("hopper")

# Horizon
T = 31

# Time step
tf = 3.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
z_h = 0.5
q1 = [0.0, 0.5, 0.0, 0.5]
v1 = [0.0, 0.0, 0.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q1]
qT = [3.0, 0.5, 0.0, 0.5]

xl, xu = state_bounds(model, T, [model.qL; model.qL], [model.qU; model.qU],
    x1 = x1, xT = [Inf*ones(model.nq); qT])

# Objective
include_objective("velocity")
obj_velocity = velocity_objective([Diagonal(ones(model.nq)) for t = 1:T],
	model.nq, idx_angle = (3:3), h = h)
obj_tracking = quadratic_tracking_objective(
    [Diagonal(zeros(model.n)) for t = 1:T],
    [Diagonal([1.0e-1, 1.0e-1, zeros(model.m - model.nu)...]) for t = 1:T-1],
    [zeros(model.n) for t = 1:T],
    [zeros(model.m) for t = 1:T-1])
obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])

# Constraints
include_constraints("contact")
con_contact = contact_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact)

# Trajectory initialization
x0 = configuration_to_state(linear_interpolation(q1, qT, T)) # linear interpolation on state
u0 = [1.0e-3 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()
@time z̄ , info = solve(prob, copy(z0),
 	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
