using Plots

# Model
include_model("planar_push_block_v3")

# Horizon
T = 26

# Time step
h = 0.1

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:2]] .= 5.0
_uu[model.idx_λ[1:4]] .= μ_surface * gravity * h * 0.25
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:2]] .= -5.0
_ul[model.idx_λ[1:4]] .= μ_surface * gravity * h * 0.25

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 0.0, -r - 1.0e-8, 0.0] # straight
x_goal = 1.0
y_goal = 0.0
θ_goal = 0.0 * π
qT = [x_goal, y_goal, θ_goal, x_goal-r, y_goal-r]

q1 = [0.0, 0.0, 0.0, -r - 1.0e-8, 0.0] # straight
x_goal = 1.0
y_goal = 0.0
θ_goal = 0.0 * π
qT = [x_goal, y_goal, θ_goal, x_goal-r, y_goal-r]

x1 = [q1; q1]
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)
xl[T][4:5] .= -Inf
xl[T][9:10] .= -Inf
xu[T][4:5] .= Inf
xu[T][9:10] .= Inf

# Objective
include_objective("velocity")
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(1.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

_Q_track = [1.0, 1.0, 1.0, 0.1, 0.1]
Q_track = 1.0 * Diagonal([_Q_track; _Q_track])
obj_tracking = quadratic_tracking_objective(
    [Q_track for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([0.1 * ones(model.nu);
		zeros(model.nc);
		ones(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

obj_penalty = PenaltyObjective(1.0e4, model.m)
obj = MultiObjective([obj_tracking, obj_penalty, obj_velocity])

# Constraints
include_constraints(["contact", "stage"])
t_idx = vcat([t for t = 1:T-1])
con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])#, con_ctrl_comp, con_ctrl_lim])

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
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = [0.1 * randn(model.m) for t = 1:T-1] # random controls
for t = 1:T-1
    if t <= 5
        u0[t][1:2] = [1.0; 0.0]
    else
        u0[t][1:2] .= 0.0
    end
end

# Pack trajectoriesff into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0),
	tol = 1.0e-3, c_tol = 1.0e-3, max_iter = 2500)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
h̄ = h

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
#open(vis)
visualize!(vis, model,
    q, u,
    Δt = h,
	r = model.block_dim[1] + model.block_rnd)

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:2], :]', linetype = :steppost, color = :black, width = 1.0, labels = "")

# q_array = hcat(q...)
# u_array = hcat(u...)

# traj = Dict("q" => q_array, "u" => u_array)

# using NPZ
# i = 2
# file_path = joinpath(pwd(), "examples/contact_implicit/manipulation/", "traj$i.npz")
# npzwrite(file_path, traj)
