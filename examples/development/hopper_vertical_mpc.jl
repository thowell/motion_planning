include(joinpath(pwd(), "examples/contact_implicit/hopper_vertical.jl"))

using TrajectoryOptimization
using Altro
using RobotDynamics
using StaticArrays
using LinearAlgebra
using ForwardDiff
const TO = TrajectoryOptimization

n, m = size(model)

# Horizon
T = 101

# Time step
tf = 1.0
h = tf / (T - 1)

# Objective
include("nonlinear_objective.jl")
z_h = 0.0
q1 = [0.0, 0.5 + z_h, 0.0, 0.25]
x1 = [q1; q1; 0.0]
qT = [0.0, 0.5 + z_h, 0.0, 0.5]
xT = [qT; qT; 0.0]
X0 = linear_interpolation(x1, xT, T)

include("initial_torque.jl")
u1 = initial_torque(model, q1, h)
U0 = [copy(u1) + 0.0 * rand(m) for t = 1:T-1]
u_ref = u1 #[u1[1:model.nu]; zeros(m - model.nu)]

function obj_stage_1(x, u)
    J = 0.0
    J += (u - u_ref)' * Diagonal([1.0e-1 * ones(model.nu); 1.0e-1 * ones(m - model.nu)]) * (u - u_ref)
    # J += 10.0 * (kinematics(model, view(x, model.nq .+ (1:model.nq)))[2] - 0.5)^2
    # J += 1.0 * sum((view(x, model.nq .+ (1:model.nq)) - view(x, 1:model.nq)).^2) / h
    return J
end

function obj_stage_t(x, u)
    J = 0.0
    J += (u - u_ref)' * Diagonal([1.0e-1 * ones(model.nu); 1.0e-1 * ones(m - model.nu)]) * (u - u_ref)
    J += 100.0 * (kinematics(model, view(x, model.nq .+ (1:model.nq)))[2] - 0.0)^2
    # J += 1.0 * sum((view(x, model.nq .+ (1:model.nq)) - view(x, 1:model.nq)).^2) / h
    # J += 10.0 * (view(x, model.nq .+ (1:model.nq))[2] - 1.0)^2
    return J
end

function obj_T(x)
    (x - xT)' * (x - xT) * 0.0
end

nl_stage_1 = NonlinearCostFunction{n, m}(obj_stage_1)
nl_stage_t = NonlinearCostFunction{n, m}(obj_stage_t)
nl_T = NonlinearCostFunction{n, m}(obj_T)
obj = Objective([t != 1 ? nl_stage_t : nl_stage_1 for t = 1:T-1], nl_terminal)

# Constraints
include(joinpath(pwd(), "contact_constraints.jl"))
cons = ConstraintList(n, m, T)

add_constraint!(cons, GoalConstraint(xT, (model.nq .+ (1:model.nq))), T)
# add_constraint!(cons, GoalConstraint(xT, (1:2*model.nq)), T)

uu = Inf * ones(m)
ul = zeros(m)
ul[model.idx_u] .= -Inf
add_constraint!(cons, BoundConstraint(n, m,
    x_min = [model.qL; model.qL; 0.0], x_max = [model.qU; model.qU; Inf],
    u_min = ul, u_max = uu), 1:T-1)

con_sd = SD(n, model.nc, model)
con_ic = IC(n, model.nc, model)
con_fc = FC(m, model.nc, model)
con_ns = NS(n, model.nc, model, h)
add_constraint!(cons, con_sd, 1:T)
add_constraint!(cons, con_ic, 2:T)
add_constraint!(cons, con_fc, 1:T-1)
add_constraint!(cons, con_ns, 2:T)

# x0 = @SVector ones(n)
# u0 = @SVector ones(m)
# TO.evaluate(con_sd, x0)
# TO.evaluate(con_ic, x0)
# TO.evaluate(con_fc, u0)
# TO.evaluate(con_fc, x0)

# Create and solve problem
prob = Problem(model, obj, xT, tf,
    U0 = U0,
    x0 = x1, constraints = cons, integration = PassThrough)
rollout!(prob)

solver = ALTROSolver(prob, projected_newton = false,
    constraint_tolerance = 1.0e-2, verbose = 2,
    penalty_initial = 10.0, penalty_scaling = 10.0)
cost(solver)           # initial cost
@time solve!(solver)         # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)

# using Plots
# plot(hcat(X...)')
# plot(hcat(U...)', linetype=:steppost)

# vis = Visualizer()
# render(vis)
visualize!(vis, model, state_to_configuration(X), Δt = h)
