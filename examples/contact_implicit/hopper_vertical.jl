# Model
include_model("hopper")
model_ft = free_time_model(model)

# Horizon
T = 101

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 10.0
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -10.0
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.25
q1 = [0.0, 0.5 + z_h, 0.0, 0.25]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; Inf * ones(model.nq)])

# Objective
include_objective("velocity")
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(zeros(model_ft.n)) for t = 1:T],
    [Diagonal([1.0, 1.0, zeros(model_ft.m - model_ft.nu - 1)..., 0.0]) for t = 1:T-1],
    [zeros(model_ft.n) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)
obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective(
    [Diagonal(10.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))
obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity])

# Constraints
include_constraints(["free_time", "contact", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)
con_loop = loop_constraints(model, (1:model.n), 1, T)
con = multiple_constraints([con_free_time, con_contact, con_loop])

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
x0 = [[q1; q1] for t = 1:T] # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

optimize = true
if optimize
	# include_snopt()
	@time z̄ = solve(prob, copy(z0),
		nlp = :ipopt,
		tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
		time_limit = 60)
	@show check_slack(z̄, prob)
	x̄, ū = unpack(z̄, prob)
	tf, t, h̄ = get_time(ū)

	# projection
	Q = [Diagonal(ones(model_ft.n)) for t = 1:T]
	R = [Diagonal(0.1 * ones(model_ft.m)) for t = 1:T-1]
	x_proj, u_proj = lqr_projection(model_ft, x̄, ū, h̄[1], Q, R)

	@show tf
	@show h̄[1]
	@save joinpath(@__DIR__, "hopper_vertical_gait.jld2") x̄ ū h̄ x_proj u_proj
else
	@load joinpath(@__DIR__, "hopper_vertical_gait.jld2") x̄ ū h̄ x_proj u_proj
end

using Plots
plot(hcat(ū...)[1:2, :]',
    linetype = :steppost,
    label = "",
    color = :red,
    width = 2.0)

plot!(hcat(u_proj...)[1:2, :]',
    linetype = :steppost,
    label = "", color = :black)

plot(hcat(state_to_configuration(x̄)...)',
    color = :red,
    width = 2.0,
	label = "")

plot!(hcat(state_to_configuration(x_proj)...)',
    color = :black,
	label = "")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x_proj), Δt = h̄[1])

# Reference trajectory
x_track = deepcopy(x_proj)
u_track = deepcopy(u_proj)

for i = 1:10
	x_track = [x_track..., x_track[2:T]...]
	u_track = [u_track..., u_track...]
end
T_track = length(x_track)

plot(hcat(state_to_configuration(x_track)...)',
    color = :black,
	label = "")

plot(hcat(u_track...)',
	color = :black,
	label = "")


K, P = tvlqr(model_ft, x_track, u_track, h̄[1],
	[Diagonal(ones(model_ft.n)) for t = 1:T_track],
	[Diagonal(ones(model_ft.m)) for t = 1:T_track - 1])

K_vec = [vec(K[t]) for t = 1:T_track-1]
P_vec = [vec(P[t]) for t = 1:T_track-1]

plot(hcat(K_vec...)', label = "")
plot(hcat(P_vec...)', label = "")

include(joinpath(pwd(), "src/simulate_contact.jl"))
model_sim = model

T_sim = 1 * T_track
tf_track = h̄[1] * (T_track - 1)
t_sim = range(0, stop = tf, length = T_sim)
t_ctrl = range(0, stop = tf, length = T_track)
h_sim = tf_track / (T_sim - 1)
x_track_stack = hcat(x_track...)

x_sim = [copy(x_proj[1])]
u_sim = []
T_horizon = T_sim - T

for tt = 1:T_horizon-1
	t = t_sim[tt]
	i = searchsortedlast(t_ctrl, t)
	println("t: $t")
	println("	i: $i")

	x_cubic = zeros(model_sim.n)
	for i = 1:model_sim.n
		interp_cubic = CubicSplineInterpolation(t_ctrl, x_track_stack[i, :])
		x_cubic[i] = interp_cubic(t)
	end

	push!(u_sim, u_track[i][1:end-1] - K[i] * (x_sim[end] - x_cubic))
	w0 = (tt == 101 ? 20.0 * [1.0; 0.0; 0.0; 0.0] .* randn(model.nq) : 1.0e-5 * randn(model.nq))
	push!(x_sim,
		step_contact(model_sim,
			x_sim[end],
			u_sim[end][1:model.nu],
			w0,
			h_sim))
end

plot(hcat(state_to_configuration(x_track[1:1:T_horizon])...)',
    labels = "", legend = :bottomleft,
    width = 2.0, color = ["red" "green" "blue" "orange"], linestyle = :dash)

plot!(hcat(state_to_configuration(x_sim[1:1:T_horizon])...)',
    labels = "", legend = :bottom,
    width = 1.0, color = ["red" "green" "blue" "orange"])

vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x_sim), Δt = h̄[1])
