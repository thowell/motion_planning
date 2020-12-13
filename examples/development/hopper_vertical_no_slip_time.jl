# Model
include_model("hopper")
model = Hopper{Discrete, FixedTime}(n, m, nq,
			   1.0, 0.1, 0.25, 0.025,
			   1.0, g,
			   qL, qU,
			   nq,
		       nu,
		       nc,
		       nf,
		       nb,
		   	   ns,
		       idx_u,
		       idx_λ,
		       idx_b,
		       idx_ψ,
		       idx_η,
		       idx_s)

model_ft = free_time_model(no_slip_model(model))

function fd(model::Hopper{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    [q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
    - M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	- h * G_func(model, q2⁺)
    + h * (transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{1}(λ)
    + transpose(P_func(model, q3)) * SVector{2}(b))
    + h * w)]
end

function fd(model::Hopper{Discrete, FreeTime}, x⁺, x, u, w, h, t)
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
	- h * G_func(model, q2⁺)
	+ h * (transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b))
	+ h * w)]
end

function B_func(model::Hopper, q)
	m1 = model.mb + model.ml
	J1 = model.Jb + model.Jl
	@SMatrix [0.0 0.0 1.0 0.0;
              -sin(q[3]) cos(q[3]) 0.0 1.0]
end

B_func(model, q1)

# Horizon
T = 101

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 100.0
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -100.0
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
    [Diagonal([1.0e-1, 1.0e-1, zeros(model_ft.m - model_ft.nu - 1)..., 0.0]) for t = 1:T-1],
    [zeros(model_ft.n) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)
obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))
obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity])

# Constraints
include_constraints(["free_time", "contact_no_slip", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_no_slip_constraints(model_ft, T)
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
u0 = [[1.0e-5 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem

optimize = true
if optimize
	include_snopt()
	@time z̄ = solve(prob, copy(z0),
		nlp = :ipopt,
		tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
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
	@save joinpath(@__DIR__, "hopper_vertical_gait_no_slip_time.jld2") x̄ ū h̄ x_proj u_proj
else
	@load joinpath(@__DIR__, "hopper_vertical_gait_no_slip_time.jld2") x̄ ū h̄ x_proj u_proj
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

for i = 1:5
	x_track = [x_track..., x_track[2:T]...]
	u_track = [u_track..., u_track...]
end
u_track = [u_track..., u_track[end]]
T_track = length(x_track)

plot(hcat(state_to_configuration(x_track)...)',
    color = :black,
	label = "")

plot(hcat(u_track...)',
	color = :black,
	label = "")

K, P = tvlqr(model_ft, x_track, u_track, h̄[1],
	[Diagonal(10.0 * ones(model_ft.n)) for t = 1:T_track],
	[Diagonal(ones(model_ft.m)) for t = 1:T_track - 1])

K_vec = [vec(K[t]) for t = 1:T_track-1]
P_vec = [vec(P[t]) for t = 1:T_track-1]

plot(hcat(K_vec...)', label = "")
plot(hcat(P_vec...)', label = "")

include(joinpath(pwd(), "src/simulate_contact.jl"))
model_sim = model

S = 1
T_sim = S * T_track
tf_track = h̄[1] * (T_track - 1)
t_sim = range(0, stop = tf_track, length = T_sim)
t_ctrl = range(0, stop = tf_track, length = T_track)
h_sim = tf_track / (T_sim - 1)
x_track_stack = hcat(x_track...)
u_track_stack = hcat(u_track...)

x_sim = [copy(x_proj[1])]
u_sim = []
T_horizon = T_sim - T
# kk = max(1, searchsortedlast(t_sim, 0.0 - h_sim))

# x_cubic = zeros(model_sim.n)
# for p = 1:model_sim.n
# 	interp_cubic = LinearInterpolation(t_ctrl, x_track_stack[p, :])
# 	x_cubic[p] = interp_cubic(t)
# end

# u_linear = zeros(model_sim.m)
# for p = 1:model_sim.m
# 	interp_linear = LinearInterpolation(t_ctrl, u_track_stack[p, :])
# 	u_linear[p] = interp_linear(t)
# end

for tt = 1:T_horizon-1
	t = t_sim[tt]
	i = searchsortedlast(t_ctrl, t)
	tm = t - h_sim + 1.0e-8
	ii = max(1, searchsortedlast(t_sim, tm))
	println("t: $t")
	println("	i: $i")
	println("t-h: $(t_sim[ii])")
	println("	ii: $ii")

	x_cubic = zeros(model_sim.n)
	for p = 1:model_sim.n
		interp_cubic = CubicSplineInterpolation(t_ctrl, x_track_stack[p, :])
		x_cubic[p] = interp_cubic(t)
	end

	if S == 1
		z = [x_sim[ii][model.nq .+ (1:model.nq)];
			 x_sim[end][model.nq .+ (1:model.nq)]]
	else
		z = [x_sim[ii][model.nq .+ (1:model.nq)];
			 x_sim[end][model.nq .+ (1:model.nq)]]
		# z = [x_sim[ii][1:model.nq];
		# 	 x_sim[end][model.nq .+ (1:model.nq)]]
	end
	# z = x_sim[end]

	# @show norm(x_sim[ii][model.nq .+ (1:model.nq)] - x_sim[end][(1:model.nq)])

	push!(u_sim, u_track[i][1:end-1] - (t >= h_sim ? 1 : 0 ) * K[i] * (z - x_track[i]))
	w0 = (tt == 101 ? 0.0 * [1.0; 0.0; 0.0; 0.0] : 0.0 * randn(model.nq))
	push!(x_sim,
		step_contact(model_sim,
			x_sim[end],
			u_sim[end][1:model.nu],
			# max.(min.(u_sim[end][1:model.nu], 100.0), -100.0),
			w0,
			h_sim))
end

_x_track = hcat(state_to_configuration(x_track)[2:end]...)
_x_sim = hcat(state_to_configuration(x_sim)[2:end]...)

plt = plot();
colors = ["red" "green" "blue" "orange"]
for i = 1:model.nq
	plt = plot!(t_ctrl, _x_track[i, :],
	    labels = "", legend = :bottomleft,
	    width = 2.0, color = colors[i], linestyle = :dash)

	plt = plot!(t_sim[1:T_horizon], _x_sim[i, :],
	    labels = "", legend = :bottom,
	    width = 1.0, color = colors[i])
end
display(plt)

# plot(hcat(u_sim...)[1:2, :]', linetype = :steppost)

vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x_sim), Δt = h_sim)
