# Model
include_model("hopper")

# Free-time model
model_ft = free_time_model(model)

# Horizon
T = 49
Tm = 25

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 250.0 #model_ft.uU
_uu[end] = 5.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -250.0 #model_ft.uL
_ul[end] = 0.5 * h
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
		xT = [qT; qT])
xl[Tm][nq .+ (1:nq)] = q_top
xu[Tm][nq .+ (1:nq)] = q_top

q_ref = [linear_interpolation(q1, q_right, 13)...,
         linear_interpolation(q_right, q_top, 13)[2:end]...,
         linear_interpolation(q_top, q_left, 13)[2:end]...,
         linear_interpolation(q_left, qT, 13)[2:end]...]

x_ref = configuration_to_state(q_ref)

# Objective
include_objective(["velocity"])
obj_tracking = quadratic_time_tracking_objective(
    [100.0 * Diagonal([1.0; 10.0; 0.1; 0.1; 1.0; 10.0; 0.1; 0.1]) for t = 1:T],
    [Diagonal([1.0, 1.0,
		zeros(model_ft.m - model_ft.nu)...]) for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T-1],
    1.0)

obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective([Diagonal(ones(model_ft.nq)) for t = 1:T],
	model_ft.nq, idx_angle = (3:3))
obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity])

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
               con = con)

# Trajectory initialization
x0 = deepcopy(x_ref) # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# include_snopt()

@time z̄ , info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

x̄, ū = unpack(z̄, prob)

@show ū[4][end]
@show check_slack(z̄, prob)

using Plots
tf, t, h̄ = get_time(ū)
plot(t[1:end-1], hcat(ū...)[1:2,:]', linetype=:steppost,
	xlabel="time (s)", ylabel = "control",
	label = ["angle" "length"],
	width = 2.0, legend = :top)
plot(t[1:end-1], h̄, linetype=:steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft,
	state_to_configuration([[x̄[1] for i = 1:50]...,x̄..., [x̄[end] for i = 1:50]...]),
	Δt = h̄[1],
	scenario = :flip)
