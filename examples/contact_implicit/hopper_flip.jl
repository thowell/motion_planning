# Model
include_model("hopper")

# Free-time model
model_ft = free_time_model(model)

function fd(model::Hopper, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
	- M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	+ transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b)
	- h * G_func(model, q2⁺))]
end

function maximum_dissipation(model::Hopper, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nb)
	η = u[model.idx_η]
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

# Horizon
T = 21

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 25.0
_uu[end] = 5.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -25.0
_ul[end] = 0.1 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.5 , 0.0, 0.5]
q_right = [-0.25, 0.5 + 0.25, pi / 2.0, 0.25]
q_top = [-0.5, 0.5 + 0.5, pi, 0.25]
q_left = [-0.75, 0.5 + 0.25, 3.0 * pi / 2.0, 0.25]
qT = [-1.0, 0.5,  2.0 * pi, 0.5]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; q1],
		xT = [Inf * ones(model_ft.nq); qT])

q_ref = [linear_interp(q1, q_right, 6)...,
         linear_interp(q_right, q_top, 6)[2:end]...,
         linear_interp(q_top, q_left, 6)[2:end]...,
         linear_interp(q_left, qT, 6)[2:end]...]

x_ref = configuration_to_state(q_ref)

# Objective
include_objective(["velocity"])
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(zeros(model_ft.n)) for t = 1:T],
    [Diagonal([1.0e-1, 1.0e-1, zeros(model_ft.m - model_ft.nu)...]) for t = 1:T-1],
    [zeros(model_ft.n) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T-1],
    1.0)

obj_contact_penalty = PenaltyObjective(1.0e3, model_ft.m - 1)
obj_velocity = velocity_objective([Diagonal(ones(model_ft.nq)) for t = 1:T],
	model_ft.nq)
obj = MultiObjective([obj_tracking, obj_velocity, obj_contact_penalty])

# Constraints
include_constraints(["contact", "free_time"])
con_contact = contact_constraints(model_ft, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_free_time, con_contact])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# Trajectory initialization
x0 = deepcopy(x_ref) # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()

@time z̄ = solve(prob, copy(z0),
	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

x̄, ū = unpack(z̄, prob)

@show ū[4][end]
@show check_slack(z̄, prob)

using Plots
tf, t, h = get_time(ū)
plot(t[1:end-1], hcat(ū...)[1:2,:]', linetype=:steppost,
	xlabel="time (s)", ylabel = "control",
	label = ["angle" "length"],
	width = 2.0, legend = :top)
plot(t[1:end-1], h, linetype=:steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model_ft, state_to_configuration(x̄), Δt = h[1])
