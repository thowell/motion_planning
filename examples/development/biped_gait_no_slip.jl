# Model
include_model("biped")
model_fixed_time = model
model = free_time_model(no_slip_model(model))

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 51
Tm = 26

# Time step
tf = 1.0
h = tf / (T - 1)
t = range(0.0, stop = tf, length = T)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to thigh 1)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to thigh 2)


# θ = pi / 10.0
# q1 = initial_configuration(model, θ) # generate initial config from θ
# k1 = kinematics_2(model, q1, body = :leg_1, mode = :ee)[1]
# k2 = kinematics_2(model, q1, body = :leg_2, mode = :ee)[1]
# d1 = abs(k1 - k2)
# qT = copy(q1)
# qT[1] += 2 * d1
# q1[3] -= pi / 30.0
# q1[4] += pi / 20.0
# q1[5] -= pi / 10.0
# q1, qM = loop_configurations(model, θ)
# qT[1] += 1.0

# θ = pi / 5.0
# q1 = zeros(model.nq)
# q1[3] = pi
# q1[4] = θ #+ pi / 20.0
# q1[5] = -2.0 * θ
# q1[6] = θ #- pi / 20.0
# q1[7] = -2.0 * θ
# q1[2] = model.l2 * cos(θ) + model.l3 * cos(θ)
# qT = copy(q1)
# qT[1] = 0.5
# # qT[2] = 1.0
# kinematics_2(model, q1, body = :leg_1, mode = :ee)[2]
# kinematics_2(model, q1, body = :leg_2, mode = :ee)[2]
function initial_configuration_1(model, θ_torso, θ_thigh_1, θ_leg_1)
    q1 = zeros(model.nq)
    q1[3] = θ_torso #pi - pi / 50.0
    q1[4] = θ_thigh_1 #pi / 7.5
    q1[5] = θ_leg_1 #- pi / 10.0
    z1 = model.l2 * cos(q1[4]) + model.l3 * cos(q1[4] + q1[5])

    q1[6] = - pi / 20.0
    q1[7] = -1.0 * acos((z1 - model.l4 * cos(q1[6])) / model.l5) - q1[6] #-pi / 20.0
    q1[2] = z1

    p1 = kinematics_2(model, q1, body = :leg_1, mode = :ee)
    p2 = kinematics_2(model, q1, body = :leg_2, mode = :ee)
    @show stride = abs(p1[1] - p2[1])

    q1[1] = -1.0 * p1[1]

    qM = copy(q1)
    qM[4] = q1[6]
    qM[5] = q1[7]
    qM[6] = q1[4]
    qM[7] = q1[5]
    qM[1] = abs(p2[1])

    pM_1 = kinematics_2(model, qM, body = :leg_1, mode = :ee)
    pM_2 = kinematics_2(model, qM, body = :leg_2, mode = :ee)

    qT = copy(q1)
    qT[1] = 2 * stride

    pT_1 = kinematics_2(model, qT, body = :leg_1, mode = :ee)
    pT_2 = kinematics_2(model, qT, body = :leg_2, mode = :ee)

    return q1, qM, qT
end

q1, qM, qT = initial_configuration_1(model, pi - pi / 50.0, pi / 7.5, -pi / 5.0)

visualize!(vis, model, [q1])

foot_2_1 = kinematics_2(model, q1, body = :leg_2, mode = :ee)
foot_2_M = kinematics_2(model, qM, body = :leg_2, mode = :ee)

foot_1_M = kinematics_2(model, qM, body = :leg_1, mode = :ee)
foot_1_T = kinematics_2(model, qT, body = :leg_1, mode = :ee)


zh = 0.05
foot_2_x = [range(foot_2_1[1], stop = foot_2_M[1], length = Tm)...,
    [foot_2_M[1] for t = 1:Tm-1]...]
foot_2_z = sqrt.((zh^2.0) * (1.0 .- ((foot_2_x).^2.0)
    ./ abs(foot_2_1[1])^2.0) .+ 1.0e-8)

foot_1_x = [[foot_1_M[1] for t = 1:Tm-1]...,
    range(foot_1_M[1], stop = foot_1_T[1], length = Tm)...]
foot_1_z = sqrt.((zh^2.0) * (1.0 .- ((foot_1_x .- abs(foot_1_T[1] / 2.0)).^2.0)
    ./ abs(foot_1_T[1] / 2.0)^2.0) .+ 1.0e-8)

using Plots
plot(foot_2_x, foot_2_z)
plot!(foot_1_x, foot_1_z, aspect_ratio = :equal)

plot(t, foot_2_x)
plot!(t, foot_1_x)

plot(t, foot_2_z)
plot!(t, foot_1_z)

# u1 = initial_torque(model_τ, qM, h)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[end] = 0.01 * h
ul, uu = control_bounds(model, T, _ul, _uu)

foot_lift2 = zeros(model.nq)
foot_lift2[7] = -pi / 50.0
visualize!(vis, model, [q1 + 1.0 * foot_lift2])

foot_lift1 = zeros(model.nq)
foot_lift1[5] = -pi / 50.0
visualize!(vis, model, [qM + 1.0 * foot_lift1])

u1 = initial_torque(model_fixed_time, q1 + 1.0 * foot_lift2, h,
    tol_r = 1.0e-3, tol_d = 1.0e-3)[model.idx_u]
uM = initial_torque(model_fixed_time, qM + 1.0 * foot_lift1, h,
    tol_r = 1.0e-3, tol_d = 1.0e-3)[model.idx_u]

# # q1_mod = copy(q1)
# # q1_mod[3] = Inf
# _xl = -Inf * ones(model.n)
# _xu = Inf * ones(model.n)

xl, xu = state_bounds(model, T,
    x1 = [Inf * ones(model.nq); q1],
    xT = [Inf * ones(model.nq); qT])

xl[Tm][model.nq .+ (1:model.nq)] = copy(qM)
xu[Tm][model.nq .+ (1:model.nq)] = copy(qM)

for t = 1:T
    xl[t][3] = q1[3] - pi / 20.0
    xl[t][10] = q1[3] - pi / 20.0
    xu[t][3] = q1[3] #+ pi / 20.0
    xu[t][10] = q1[3] #+ pi / 20.0
end

# Objective
include_objective(["velocity", "nonlinear_stage"])

# q_ref = linear_interpolation(q1, qM, T)
q_ref = [linear_interpolation(q1, qM, Tm)...,
    linear_interpolation(qM, qT, Tm)[2:end]...]
x0 = configuration_to_state(q_ref)

vis = Visualizer()
render(vis)
visualize!(vis, model, q_ref, Δt = h)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e5, model.m - 1)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_time_tracking_objective(
    [Diagonal(1.0e-5 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-3 * ones(model.nu)..., 0.0e-5 * ones(model.m - model.nu - 1)..., 0.0]) for t = 1:T-1],
    [x0[t] for t = 1:T],
    [zeros(model.m) for t = 1:T-1],
    1.0)

# quadratic velocity penalty
# Σ v' Q v
q_v = 100.0 * ones(model.nq)
# q_v[2] = 0.0
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
function l_stage_torso_h(x, u, t)
    return 1000.0 * (kinematics_1(model,
            view(x, 8:14), body = :torso, mode = :com)[2] - t_h)^2.0
end

l_terminal_torso_h(x) = 0.0
obj_th = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
function l_stage_torso_lat(x, u, t)
    return (0.0 * (kinematics_1(model,
        view(x, 8:14), body = :torso, mode = :com)[1]
        - kinematics_1(model,
        view(x0[t], 8:14), body = :torso, mode = :com)[1])^2.0)
end
l_terminal_torso_lat(x) = 0.0
obj_tl = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
function l_stage_fh1(x, u, t)
    return (10000.0 * (kinematics_2(model,
        view(x, 1:7), body = :leg_1, mode = :ee)[2] - foot_1_z[t])^2.0
        + 10000.0 * (kinematics_2(model,
            view(x, 8:14), body = :leg_1, mode = :ee)[2] - foot_1_z[t])^2.0)
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 1 lateral
function l_stage_fl1(x, u, t)
    return (10.0 * (kinematics_2(model,
        view(x, 1:7), body = :leg_1, mode = :ee)[1] - foot_1_x[t])^2.0
        + 10.0 * (kinematics_2(model,
            view(x, 8:14), body = :leg_1, mode = :ee)[1] - foot_1_x[t])^2.0)
end
l_terminal_fl1(x) = 0.0
obj_fl1 = nonlinear_stage_objective(l_stage_fl1, l_terminal_fl1)

# foot 2 height
function l_stage_fh2(x, u, t)
    return (10000.0 * (kinematics_2(model,
        view(x, 1:7), body = :leg_2, mode = :ee)[2] - foot_2_z[t])^2.0
        + 10000.0 * (kinematics_2(model,
            view(x, 8:14), body = :leg_2, mode = :ee)[2] - foot_2_z[t])^2.0)
end
l_terminal_fh2(x) = 0.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# foot 2 lateral
function l_stage_fl2(x, u, t)
    return (10.0 * (kinematics_2(model,
        view(x, 1:7), body = :leg_2, mode = :ee)[2] - foot_2_x[t])^2.0
        + 10.0 * (kinematics_2(model,
            view(x, 8:14), body = :leg_2, mode = :ee)[2] - foot_2_x[t])^2.0)
end
l_terminal_fl2(x) = 0.0
obj_fl2 = nonlinear_stage_objective(l_stage_fl2, l_terminal_fl2)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_th,
                      obj_tl,
                      obj_fh1,
                      obj_fh2,
                      obj_fl1,
                      obj_fl2])

# Constraints
include_constraints(["contact_no_slip", "loop", "free_time"])
con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)
con_contact = contact_no_slip_constraints(model, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_contact, con_free_time, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               # h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# trajectory initialization
u0 = [[1.0e-4 * rand(model.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
optimize = true

if optimize
    include_snopt()

	@time z̄ , info = solve(prob, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
		time_limit = 60 * 1)
	@show check_slack(z̄, prob)
	x̄, ū = unpack(z̄, prob)
    tf, t, h̄ = get_time(ū)

	# projection
	# Q = [Diagonal(ones(model.n)) for t = 1:T]
	# R = [Diagonal(0.1 * ones(model.m)) for t = 1:T-1]
	# x_proj, u_proj = lqr_projection(model, x̄, ū, h̄[1], Q, R)
    #
	# @show tf
	# @show h̄[1]
	# @save joinpath(@__DIR__, "biped_gait_no_slip.jld2") x̄ ū h̄ x_proj u_proj
else
	# @load joinpath(@__DIR__, "biped_gait_no_slip.jld2") x̄ ū h̄ x_proj u_proj
end

# Visualize
# vis = Visualizer()
# render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h̄[1])

fh1 = [kinematics_2(model,
    state_to_configuration(x_proj)[t], body = :leg_1, mode = :ee)[2] for t = 1:T]
fh2 = [kinematics_2(model,
    state_to_configuration(x_proj)[t], body = :leg_2, mode = :ee)[2] for t = 1:T]

# plot(fh1, linetype = :steppost, label = "foot 1")
# plot!(fh2, linetype = :steppost, label = "foot 2")
#
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
