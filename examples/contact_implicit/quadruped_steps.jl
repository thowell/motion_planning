# Model
include_model("quadruped")

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
θ = pi / 3.5

q1 = initial_configuration(model, θ)
qT = copy(q1)
q1[1] = -1.0
qT[1] = 1.0
visualize!(vis, model, [q1])

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T - 1)

# Bounds

# control
u1 = initial_torque(model, q1, h)[model.idx_u]

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
include_objective(["velocity", "nonlinear_stage"])
q_ref = linear_interpolation(q1, qT, T)
render(vis)
visualize!(vis, model, q_ref)
x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [Diagonal(1.0e-1 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-2 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
    [x0[end] for t = 1:T],
    [[u1; zeros(model.m - model.nu)] for t = 1:T])

# quadratic velocity penalty
# Σ v' Q v
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9, 10, 11]))

# torso height
q2_idx = (12:22)
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
function l_stage_torso_h(x, u, t)
    10000.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[2] - t_h)^2.0
end
l_terminal_torso_h(x) = 0.0
obj_torso_h = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
function l_stage_torso_lat(x, u, t)
    1.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[t], q2_idx), body = :torso, mode = :com)[1])^2.0
end
l_terminal_torso_lat(x) = 0.0
obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
function l_stage_fh1(x, u, t)
    100.0 * (kinematics_2(model, view(x, q2_idx), body = :calf_1, mode = :ee)[2] - 0.1)^2.0
end
l_terminal_fh1(x) = 0.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
function l_stage_fh2(x, u, t)
    100.0 * (kinematics_2(model, view(x, q2_idx), body = :calf_2, mode = :ee)[2] - 0.1)^2.0
end
l_terminal_fh2(x) = 0.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# foot 3 height
function l_stage_fh3(x, u, t)
    100.0 * (kinematics_3(model, view(x, q2_idx), body = :calf_3, mode = :ee)[2] - 0.1)^2.0
end
l_terminal_fh3(x) = 0.0
obj_fh3 = nonlinear_stage_objective(l_stage_fh3, l_terminal_fh3)

# foot 4 height
function l_stage_fh4(x, u, t)
    100.0 * (kinematics_3(model, view(x, q2_idx), body = :calf_4, mode = :ee)[2] - 0.1)^2.0
end
l_terminal_fh4(x) = 0.0
obj_fh4 = nonlinear_stage_objective(l_stage_fh4, l_terminal_fh4)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_torso_h,
                      obj_torso_lat,
                      obj_fh1,
                      obj_fh2,
                      obj_fh3,
                      obj_fh4])

# Constraints
include_constraints(["contact"])
con_contact = contact_constraints(model, T)
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
               con = con)

# trajectory initialization
u0 = [[u1; 1.0e-5 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
z0 .+= 0.001 * randn(prob.num_var)

# Solve
include_snopt()

@time z̄ , info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3,
    time_limit = 60 * 20, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

# Visualize
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
