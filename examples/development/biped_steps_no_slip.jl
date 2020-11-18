# Model
include_model("biped")
include_constraints(["contact_no_slip"])
model = no_slip_model(model)

include(joinpath(pwd(), "src/objectives/velocity.jl"))
include(joinpath(pwd(), "src/objectives/nonlinear_stage.jl"))

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
# θ = pi / 12.5
# q1 = initial_configuration(model, θ) # generate initial config from θ
# qT = copy(q1)
# qT[1] += 1.0
# q1[3] -= pi / 30.0
# q1[4] += pi / 20.0
# q1[5] -= pi / 10.0
# q1, qT = loop_configurations(model, θ)
# qT[1] += 1.0

θ = pi / 7.5
q1 = zeros(model.nq)
q1[3] = pi
q1[4] = θ #+ pi / 20.0
q1[5] = -2.0 * θ
q1[6] = θ #- pi / 20.0
q1[7] = -2.0 * θ
q1[2] = model.l2 * cos(θ) + model.l3 * cos(θ)
qT = copy(q1)
qT[1] = 2.5
# qT[2] = 1.0
kinematics_2(model, q1, body = :leg_1, mode = :ee)[2]
kinematics_2(model, q1, body = :leg_2, mode = :ee)[2]

visualize!(vis, model, [q1])

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T - 1)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] = model.uU

_ul = zeros(model.m)
_ul[model.idx_u] = model.uL
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [q1; q1],
    xT = [qT; qT])

# Objective
q_ref = linear_interp(q1, qT, T)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e3, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [zeros(model.n, model.n) for t = 1:T],
    [Diagonal([1.0 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
    [zeros(model.n) for t = 1:T],
    [zeros(model.m) for t = 1:T]
    )

# quadratic velocity penalty
# Σ v' Q v
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7]))

# torso height
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
l_stage_torso_h(x, u, t) = 10.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - t_h)^2.0
l_terminal_torso_h(x) = 10.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - t_h)^2.0
obj_torso_h = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
l_stage_torso_lat(x, u, t) = (1.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[t], 8:14), body = :torso, mode = :com)[1])^2.0)
l_terminal_torso_lat(x) = (1.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[T], 8:14), body = :torso, mode = :com)[1])^2.0)
obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
l_stage_fh1(x, u, t) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_1, mode = :ee)[2] - 0.025)^2.0
l_terminal_fh1(x) = 1.0 * (kinematics_2(model, view(x, 8:14), body = :leg_1, mode = :ee)[2])^2.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
l_stage_fh2(x, u, t) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_2, mode = :ee)[2] - 0.025)^2.0
l_terminal_fh2(x) = 1.0 * (kinematics_2(model, view(x, 8:14), body = :leg_2, mode = :ee)[2])^2.0
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
                      obj_fh2])#,
                      # obj_conf])

# Constraints
con_contact = contact_no_slip_constraints(model, T)
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
               con = con
               )

# trajectory initialization
u0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
include_snopt()

@time z̄ = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3,
    time_limit = 60 * 3, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

# Visualize
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
