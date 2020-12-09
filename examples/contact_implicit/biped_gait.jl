# Model
include_model("biped")
model = free_time_model(model)

# model_τ = Biped{Discrete, FixedTime}(n, m, d,
# 			  g, μ,
# 			  l_torso, d_torso, m_torso, J_torso,
# 			  l_thigh, d_thigh, m_thigh, J_thigh,
# 			  l_leg, d_leg, m_leg, J_leg,
# 			  l_thigh, d_thigh, m_thigh, J_thigh,
# 			  l_leg, d_leg, m_leg, J_leg,
# 			  qL, qU,
# 			  uL, uU,
# 			  nq,
# 			  nu,
# 			  nc,
# 			  nf,
# 			  nb,
# 			  ns,
# 			  idx_u,
# 			  idx_λ,
# 			  idx_b,
# 			  idx_ψ,
# 			  idx_η,
# 			  idx_s)

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

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

zh = 0.1
foot_2_x1 = range(foot_2_1[1], stop = foot_2_M[1], length = T)
foot_z1 = sqrt.((zh^2.0) * (1.0 .- ((foot_2_x1).^2.0) ./ abs(foot_2_1[1])^2.0) .+ 1.0e-8)

foot_1_x2 = range(foot_1_M[1], stop = foot_1_T[1], length = 26)
foot_z2 = sqrt.((zh^2.0) * (1.0 .- ((foot_1_x2 .- abs(foot_1_T[1] / 2.0)).^2.0) ./ abs(foot_1_T[1] / 2.0)^2.0) .+ 1.0e-8)

using Plots
plot(foot_2_x1, foot_z1)
plot!(foot_1_x2, foot_z2, aspect_ratio = :equal)

# Horizon
T = 51
Tm = 26

# Time step
tf = 2.0
h = tf / (T - 1)

# u1 = initial_torque(model_τ, qM, h)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= model.uU
_uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= model.uL
_ul[end] = 0.01 * h
ul, uu = control_bounds(model, T, _ul, _uu)

# u1 = initial_torque(model, q1, h)[model.idx_u]

# q1_mod = copy(q1)
# q1_mod[3] = Inf
_xl = -Inf * ones(model.n)
_xu = Inf * ones(model.n)
_xl[3] = q1[3] - pi / 25.0
_xl[10] = q1[3] - pi / 25.0
_xu[3] = q1[3] + pi / 25.0
_xu[10] = q1[3] + pi / 25.0

xl, xu = state_bounds(model, T,
    _xl, _xu,
    x1 = [Inf * ones(model.nq); q1],
    xT = [Inf * ones(model.nq); qT])

xl[Tm][model.nq .+ (1:model.nq)] = copy(qM)
xu[Tm][model.nq .+ (1:model.nq)] = copy(qM)

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
    [Diagonal(0.0 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu)..., 0.0 * ones(model.m - model.nu - 1)..., 0.0]) for t = 1:T-1],
    [0.0 * x0[t] for t = 1:T],
    [zeros(model.m) for t = 1:T-1],
    1.0)

# quadratic velocity penalty
# Σ v' Q v
q_v = 1.0 * ones(model.nq)
q_v[2] = 1000.0
obj_velocity = velocity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
l_stage_torso_h(x, u, t) = 1000.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - t_h)^2.0
l_terminal_torso_h(x) = 0.0 #* (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - t_h)^2.0
obj_torso_h = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
l_stage_torso_lat(x, u, t) = (1.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[t], 8:14), body = :torso, mode = :com)[1])^2.0)
l_terminal_torso_lat(x) = (0.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[T], 8:14), body = :torso, mode = :com)[1])^2.0)
obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
function l_stage_fh1(x, u, t)
    if t > Tm
        # return 0.0
        return (1000.0 * (kinematics_2(model,
            view(x, 1:7), body = :leg_1, mode = :ee)[2] - foot_z2[t - 25])^2.0
            + 1000.0 * (kinematics_2(model,
                view(x, 8:14), body = :leg_1, mode = :ee)[2] - foot_z2[t - 25])^2.0)
    else
        # return 0.0
        return (10000.0 * (kinematics_2(model,
            view(x, 1:7), body = :leg_1, mode = :ee)[2] - 0.0)^2.0
            + 10000.0 * (kinematics_2(model,
                view(x, 8:14), body = :leg_1, mode = :ee)[2] - 0.0)^2.0)
    end
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
function l_stage_fh2(x, u, t)
    if t < Tm
        return (1000.0 * (kinematics_2(model,
            view(x, 1:7), body = :leg_2, mode = :ee)[2] - foot_z1[t])^2.0
            + 1000.0 * (kinematics_2(model,
                view(x, 8:14), body = :leg_2, mode = :ee)[2] - foot_z1[t])^2.0)
    else
        # return 0.0
        return (10000.0 * (kinematics_2(model,
            view(x, 1:7), body = :leg_2, mode = :ee)[2] - 0.0)^2.0
            + 10000.0 * (kinematics_2(model,
                view(x, 8:14), body = :leg_2, mode = :ee)[2] - 0.0)^2.0)
    end
end
l_terminal_fh2(x) = 0.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# initial configuration
# function l_stage_conf(x, u, t)
#     if t == 1
#         return 0.0#(x - [q1; q1])' * Diagonal(1000.0 * ones(model.n)) * (x - [q1; q1])
#     else
#         return 0.0
#     end
# end
# l_terminal_conf(x) = (x - [qT; qT])' * Diagonal(1000.0 * ones(model.n)) * (x - [qT; qT])
# obj_conf = nonlinear_stage_objective(l_stage_conf, l_terminal_conf)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_torso_h,
                      obj_torso_lat,
                      obj_fh1,
                      obj_fh2])

# Constraints
include_constraints(["contact", "loop", "free_time"])
con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)
con_contact = contact_constraints(model, T)
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
u0 = [[1.0e-5 * rand(model.m - 1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
include_snopt()

@time z̄ = solve(prob, copy(z0),
    nlp = :SNOPT7,
    # max_iter = 1000,
    tol = 1.0e-3, c_tol = 1.0e-3,
    time_limit = 60 * 3, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
tfc, tc, hc = get_time(ū)

# Visualize
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = ū[1][end])

fh1 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :leg_1, mode = :ee)[2] for t = 1:T]
fh2 = [kinematics_2(model,
    state_to_configuration(x̄)[t], body = :leg_2, mode = :ee)[2] for t = 1:T]

plot(fh1)
plot!(fh2)
