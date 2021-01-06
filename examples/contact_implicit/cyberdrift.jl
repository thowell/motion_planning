# Model
include_model("cybertruck")

# Horizon
T = 26

# Time step
tf = 1.0
h = tf / (T-1)

_uu = Inf * ones(model.m)
_uu[model.idx_u] = [Inf; 1.0]
_ul = zeros(model.m)
_ul[model.idx_u] .= [0.0; -1.0]

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 1.0, 0.0, -1.0 * pi / 2.0]
qT = [3.0, 0.0, 0.0, pi / 2.0]

x1 = [q1; q1]
xT = [qT; qT]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
Qq = Diagonal(ones(model.nq))
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(10000.0 * Qq, 10000.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-3 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
include_constraints(["obstacles", "control_complementarity", "contact"])
p_car1 = [3.0, 0.65]
p_car2 = [3.0, -0.65]

function obstacles!(c, x)
    c[1] = circle_obs(x[1], x[2], p_car1[1], p_car1[2], 0.5)
    c[2] = circle_obs(x[1], x[2], p_car2[1], p_car2[2], 0.5)
    nothing
end

n_obs_stage = 2
n_obs_con = n_obs_stage * T
con_obstacles = ObstacleConstraints(n_obs_con, (1:n_obs_con), n_obs_stage)

n_cc_stage = 2
n_cc_con = n_cc_stage * (T - 1)
con_ctrl_comp = ControlComplementarity(n_cc_con, (1:n_cc_con), n_cc_stage)

con_contact = contact_constraints(model, T)

con = multiple_constraints([con_contact, con_ctrl_comp, con_obstacles])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
q_ref = linear_interpolation(q1, qT, T)
x_ref = configuration_to_state(q_ref)
x0 = deepcopy(x_ref)
u0 = [1.0e-4 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()
@time z̄ , info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(z̄, prob)

x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model,
    state_to_configuration([[x̄[1] for i = 1:10]...,x̄..., [x̄[end] for i = 1:10]...]),
    Δt = h)
