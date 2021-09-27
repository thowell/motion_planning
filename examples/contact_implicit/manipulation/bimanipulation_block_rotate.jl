using Plots

# Model
include_model("bimanipulation_block_wall")

# Horizon
T = 26

# Time step
tf = 2.5 # 2.5
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u[1:4]] .= [Inf; 0.0; 0.0; 0.0]
# _uu[model.idx_u[1]] = 0.5
_ul = zeros(model.m)
_ul[model.idx_u[1:4]] .= [-Inf; -0.0; 0.0; 0.0]
# _ul[model.idx_u[1]] = 0.5
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, r, 0.0, -r, r + 0.05, -r, r + 0.05]
x1 = [q1; q1]
# qT = [0.0, r, 0.0, -2.0 * r, r, -2.0 * r, r]

# x1 = [q1; q1]
qT = [r, r, 0.0, -0, r, -0, r]
qT = [r, r, 0.5 * π, -0, r, -0, r]
ϕ_func(model, qT)
xT = [qT; qT]

# Trajectory initialization
q0 = [q1, linear_interpolation(q1, qT, T)...]
x0 = configuration_to_state(q0)

u0 = [[0.001; 0.001 * rand(model.m-1)] for t = 1:T-1] # random controls

xl, xu = state_bounds(model, T, x1 = x1, xT = [xT[1:3]; Inf * ones(4); xT[8:10]; Inf * ones(4)])
for t = 1:T
    xl[t][5] = 0.25 * r
    xl[t][12] = 0.25 * r
end
# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(1.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

obj_tracking = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu);
        zeros(model.nc);
        zeros(model.nb);
        zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_ctrl_vel = control_velocity_objective(Diagonal([1.0e-2 * ones(model.nu); 1.0 * ones(model.nc + model.nb); zeros(model.m - model.nu - model.nc - model.nb)]))

obj_penalty = PenaltyObjective(1.0e5, model.m)

obj = MultiObjective([obj_penalty, obj_velocity, obj_tracking, obj_ctrl_vel])

# Constraints
include_constraints(["contact"])
con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])

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

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-2, c_tol = 1.0e-2, max_iter=2500)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = h

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model,
    # q0, u0,
	q, u,
    Δt = h)


plot(hcat(q...)[1:2, :]', width = 1.0)
plot(hcat([u..., u[end]]...)[model.idx_u[1:4], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")

model.idx_u
