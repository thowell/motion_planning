# Model
include_model("simple_manipulator")

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [pi / 2.5, -2.0 * pi / 2.5, 0.5, 0.0]
qT = [pi / 2.5, -2.0 * pi / 2.5, 1.5, 0.0]

x1 = [q1; q1]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
q_ref = linear_interpolation(q1, qT, T)
x_ref = configuration_to_state(q_ref)

Qq = Diagonal([1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, Diagonal([10.0, 10.0, 100.0, 1.0]), dims = (1, 2))
R = Diagonal([1.0e-3, 1.0e-3, zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e3, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

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
x0 = deepcopy(x_ref) # linear interpolation on state
u0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄ , info = solve(prob, copy(z0))

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration([[x̄[1] for i = 1:5]...,x̄..., [x̄[end] for i = 1:5]...]),
    Δt = h)
open(vis)
