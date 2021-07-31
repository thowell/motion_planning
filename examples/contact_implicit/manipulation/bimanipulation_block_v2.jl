using Plots

# Model
include_model("bimanipulation_block_v2")

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T-1)


# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[model.idx_u[1:4]] .= 10.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_u[1:4]] .= -10.0

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
ϵ = 1.0e-8
q1 = [0.0, r, 0.0, -r - ϵ, r, r + ϵ, r]
x1 = [q1; q1]
ϕ_func(model, q1)
x_shift = 0.0
z_shift = 0.25
qT = [x_shift, r + z_shift, 0.0, -r + x_shift, r + z_shift, r + x_shift, r + z_shift]
ϕ_func(model, qT)
xT = [qT; qT]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

ϕ_func(model, zeros(7))
P_func(model, zeros(7))

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])
obj_velocity = velocity_objective(
    [t > T / 2 ? Diagonal(1.0 * ones(model.nq)) : Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3]))

obj_tracking = quadratic_tracking_objective(
    [Diagonal(0.0 * ones(model.n)) for t = 1:T],
    # [Diagonal(0.1 * ones(model.m)) for t = 1:T-1],
	[Diagonal([0.0 * ones(model.nu);
		zeros(model.nc);
		ones(model.nb);
		zeros(model.m - model.nu - model.nc - model.nb)]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model.m) for t = 1:T])

function l_stage(x, u, t)
	J = 0.0
	p_box = view(x, 7 .+ (1:2))
	p_pusher1 = view(x, 7 .+ (4:5))
	p_pusher2 = view(x, 7 .+ (6:7))

	ϕ = ϕ_func(model, view(x, 7 .+ (1:7)))
	if true
		J += 1.0 * sum(ϕ[5:6].^2.0)
		# J += 100.0 * sum((p_box - p_pusher1).^2.0)
		# J += 100.0 * sum((p_box - p_pusher2).^2.0)
	end

	return J
end

obj_stage = nonlinear_stage_objective(l_stage, l_stage)

obj_penalty = PenaltyObjective(1.0e5, model.m)

obj_ctrl_velocity = control_velocity_objective(Diagonal([1.0 * ones(model.nu)..., 0.0e-3 * ones(model.nc + model.nb)..., zeros(model.m - model.nu - model.nc - model.nb)...]))

obj = MultiObjective([obj_tracking,
	obj_penalty, obj_velocity, obj_stage, obj_ctrl_velocity])

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

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄, info = solve(prob, copy(z0), tol = 1.0e-3, c_tol = 1.0e-3, max_iter = 2000)

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
	r = model.block_dim[1])

plot(hcat(q̄...)', color = :red, width = 1.0, labels = "")
plot(hcat([ū..., ū[end]]...)[model.idx_u[1:2], :]', linetype = :steppost, width = 1.0, labels = ["x" "z"])
plot(hcat([ū..., ū[end]]...)[model.idx_u[3:4], :]', linetype = :steppost, width = 1.0, labels = ["x" "z"])
