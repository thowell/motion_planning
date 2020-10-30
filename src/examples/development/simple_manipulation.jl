include(joinpath(pwd(), "src/models/simple_manipulator.jl"))
include(joinpath(pwd(), "src/constraints/constraints_contact.jl"))

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
q1 = [pi / 2.5, -2.0 * pi / 2.5, 0.75, 0.0]
qT = [pi / 2.5, -2.0 * pi / 2.5, 1.5, 0.0]

x1 = [q1; q1]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
q_ref = linear_interp(q1, qT, T)
x_ref = configuration_to_state(q_ref)

Qq = Diagonal([1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, Diagonal([10.0, 10.0, 100.0, 1.0]), dims = (1, 2))
R = Diagonal([1.0e-1, 1.0e-1, zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [zeros(model.m) for t = 1:T]
    )
obj_penalty = PenaltyObjective(100.0, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
con_contact = contact_constraints(model, T)

# Problem
prob = problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact
               )

# Trajectory initialization
X0 = deepcopy(x_ref) # linear interpolation on state
U0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z̄ = solve(prob, copy(Z0))

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model, state_to_configuration(X̄), Δt = h)
