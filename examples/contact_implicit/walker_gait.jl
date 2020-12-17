# Model
include_model("walker")

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 26
Tm = 13

# Time step
tf = 2.5
h = tf / (T - 1)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to thigh 1)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to thigh 2)
# 8: foot 1 (rel. to calf 1)
# 9: foot 2 (rel. to calf 2)

q1 = zeros(model.nq)
q1[3] = pi
q1[4] = -pi
q1[6] = -pi
q1[8] = pi / 2.0
q1[9] = pi / 2.0
q1[2] = model.l_thigh1 + model.l_calf1

qT = copy(q1)
qT[1] = 1.0
q_ref = linear_interpolation(q1, qT, T)
visualize!(vis, model, q_ref, Δt = h)

# zh = 0.15
# foot_2_x = [range(foot_2_1[1], stop = foot_2_M[1], length = Tm)...,
#     [foot_2_M[1] for t = 1:Tm-1]...]
# foot_2_z = sqrt.((zh^2.0) * (1.0 .- ((foot_2_x).^2.0)
#     ./ abs(foot_2_1[1])^2.0) .+ 1.0e-8)
#
# foot_1_x = [[foot_1_M[1] for t = 1:Tm-1]...,
#     range(foot_1_M[1], stop = foot_1_T[1], length = Tm)...]
# foot_1_z = sqrt.((zh^2.0) * (1.0 .- ((foot_1_x .- abs(foot_1_T[1] / 2.0)).^2.0)
#     ./ abs(foot_1_T[1] / 2.0)^2.0) .+ 1.0e-8)
#
# using Plots
# plot(foot_2_x, foot_2_z)
# plot!(foot_1_x, foot_1_z, aspect_ratio = :equal)
#
# t = range(0.0, stop = tf, length = T)
# plot(t, foot_2_x)
# plot!(t, foot_1_x)
#
# plot(t, foot_2_z)
# plot!(t, foot_1_z)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
u1 = initial_torque(model, q1, h)[model.idx_u]
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# u1 = initial_torque(model, q1, h)[model.idx_u]

xl, xu = state_bounds(model, T,
    -Inf * ones(model.n), Inf * ones(model.n),
    x1 = [q1; q1],
    xT = [qT; qT])

# plot(hcat(q_ref...)')

# Objective
include_objective(["velocity", "nonlinear_stage"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [Diagonal(1.0 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu)..., 0.0 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x0[end] for t = 1:T],
    [[u1; zeros(model.m - model.nu)] for t = 1:T-1])

# quadratic velocity penalty
# Σ v' Q v
q_v = 10.0 * ones(model.nq)
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
function l_stage_torso_h(x, u, t)
    return 100.0 * (kinematics_1(model,
            view(x, 9 .+ (1:9)), body = :torso, mode = :com)[2] - t_h)^2.0
end

l_terminal_torso_h(x) = 0.0
obj_th = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
function l_stage_torso_lat(x, u, t)
    return 0.0 * (kinematics_1(model,
        view(x, 9 .+ (1:9)), body = :torso, mode = :com)[1]
		- kinematics_1(model,
	        view(x0[t], 9 .+ (1:9)), body = :torso, mode = :com)[1])
end
l_terminal_torso_lat(x) = 0.0
obj_tl = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
function l_stage_fh1(x, u, t)
    return 100.0 * (kinematics_3(model,
        view(x, 9 .+ (1:9)), body = :foot_1, mode = :com)[2] - 0.25)^2.0
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# # foot 1 lateral
# function l_stage_fl1(x, u, t)
#     return 10.0 * (kinematics_3(model,
#         view(x, 9 .+ (1:9)), body = :foot_1, mode = :com)[1] - foot_1_x[t])^2.0
# end
# l_terminal_fl1(x) = 0.0
# obj_fl1 = nonlinear_stage_objective(l_stage_fl1, l_terminal_fl1)
#
# foot 2 height
function l_stage_fh2(x, u, t)
    return 100.0 * (kinematics_3(model,
        view(x, 9 .+ (1:9)), body = :foot_2, mode = :com)[2] - 0.25)^2.0
end
l_terminal_fh2(x) = 0.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# # foot 2 lateral
# function l_stage_fl2(x, u, t)
#     return 10.0 * (kinematics_3(model,
#         view(x, 9 .+ (1:9)), body = :foot_2, mode = :com)[2] - foot_2_x[t])^2.0
# end
# l_terminal_fl2(x) = 0.0
# obj_fl2 = nonlinear_stage_objective(l_stage_fl2, l_terminal_fl2)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_th,
                      obj_tl,
                      obj_fh1,
                      obj_fh2])
                      # obj_fl1,
                      # obj_fl2])

# Constraints
include_constraints(["contact"])#, "loop", "free_time"])
# con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)
con_contact = contact_constraints(model, T)
# con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact])#, con_free_time, con_loop])

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

# trajectory initialization
u0 = [[u1; 1.0e-5 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
optimize = true
include_snopt()
@time z̄ = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
    time_limit = 60 * 3)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

visualize!(vis, model, state_to_configuration(x̄), Δt = h)

# plot(hcat(state_to_configuration(x̄)...)')
# if optimize
#     include_snopt()
#
# 	@time z̄ = solve(prob, copy(z0),
# 		nlp = :SNOPT7,
# 		tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
# 		time_limit = 60 * 3)
# 	@show check_slack(z̄, prob)
# 	x̄, ū = unpack(z̄, prob)
#     tfc, tc, h̄ = get_time(ū)
#
# 	# projection
# 	# Q = [Diagonal(ones(model.n)) for t = 1:T]
# 	# R = [Diagonal(0.1 * ones(model.m)) for t = 1:T-1]
# 	# x_proj, u_proj = lqr_projection(model, x̄, ū, h̄[1], Q, R)
#     #
# 	# @show tfc
# 	# @show h̄[1]
# 	# @save joinpath(@__DIR__, "biped_gait_no_slip.jld2") x̄ ū h̄ x_proj u_proj
# else
# 	# @load joinpath(@__DIR__, "biped_gait_no_slip.jld2") x̄ ū h̄ x_proj u_proj
# end

# Visualize
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h̄[1])

fh1 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :calf_1, mode = :ee)[2] for t = 1:T]
fh2 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :calf_2, mode = :ee)[2] for t = 1:T]

plot(fh1, linetype = :steppost, label = "foot 1")
plot!(fh2, linetype = :steppost, label = "foot 2")

# plot(hcat(ū...)[1:4, :]',
#     linetype = :steppost,
#     label = "",
#     color = :red,
#     width = 2.0)
#
# plot!(hcat(u_proj...)[1:4, :]',
#     linetype = :steppost,
#     label = "",
#     color = :black)
#
# plot(hcat(u_proj...)[5:6, :]',
#     linetype = :steppost,
#     label = "",
#     width = 2.0)
#
# plot(hcat(state_to_configuration(x̄)...)',
#     color = :red,
#     width = 2.0,
#     label = "")
#
# plot!(hcat(state_to_configuration(x_proj)...)',
#     color = :black,
#     label = "")
