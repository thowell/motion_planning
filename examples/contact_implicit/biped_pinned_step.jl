# Model
include_model("biped")

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)

θ = pi / 10.0
# q1 = initial_configuration(model, θ)
q1, qT = loop_configurations(model, θ)
visualize!(vis, model, [q1])

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T-1)

# Bounds

# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= model.uU
_ul = zeros(model.m)
_ul[model.idx_u] .= model.uL
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T, x1 = [q1; Inf * ones(model.nq)], xT = [Inf * ones(model.nq); qT])

# Objective
include_objective(["velocity", "nonlinear_stage"])
q_ref = linear_interpolation(q1, qT, T)
x0 = configuration_to_state(q_ref)

obj_penalty = PenaltyObjective(1.0e5, model.m)

Qq = Diagonal(ones(model.nq))
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 10.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [x0[t] for t = 1:T],
    [zeros(model.m) for t = 1:T]
    )
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq)

l_stage_fh(x, u, t) = 1.0 * (kinematics_2(model, view(x, 8:14), body = :leg_2, mode = :ee)[2] - 0.025)^2.0
l_terminal_fh(x) = 0.0
obj_fh = nonlinear_stage_objective(l_stage_fh, l_terminal_fh)
l_stage_fh(rand(model.n), rand(model.m), 1)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])#, obj_fh])

# Constraints
include_constraints(["contact", "pinned_foot"])
con_contact = contact_constraints(model, T)
con_pinned = pinned_foot_constraint(model, q1, T)
con = multiple_constraints([con_contact, con_pinned])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# trajectory initialization
q_ref = linear_interpolation(q1, qT, T)
x0 = configuration_to_state(q_ref) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

@time z̄ = solve(prob, copy(z0),
    nlp = :ipopt,
    tol = 1.0e-3, c_tol = 1.0e-3)

check_slack(z̄, prob)

x̄, ū = unpack(z̄, prob)

# Visualize
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
