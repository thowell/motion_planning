include(joinpath(pwd(), "src/models/hopper.jl"))
include(joinpath(pwd(), "src/constraints/constraints_contact.jl"))

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
Qq = Diagonal([1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1, 1.0e-3, zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [[qT; qT] for t = 1:T],
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
X0 = [x1 for t = 1:T] #linear_interp(x1, x1, T) # linear interpolation on state
U0 = [1.0e-2 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time Z̄ = solve(prob, copy(Z0), tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model, state_to_configuration(X̄), Δt = h)
