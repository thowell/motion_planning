include(joinpath(pwd(), "src/models/biped.jl"))
include(joinpath(pwd(), "src/objectives/velocity.jl"))
include(joinpath(pwd(), "src/objectives/nonlinear_stage.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to thigh 1)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to thigh 2)
θ = pi / 10.0
# q1 = initial_configuration(model, θ) # generate initial config from θ
# qT = copy(q1)
# qT[1] += 1.0
# q1[3] -= pi / 30.0
# q1[4] += pi / 20.0
# q1[5] -= pi / 10.0
q1, qT = loop_configurations(model, θ)
qT[1] += 1.0
visualize!(vis, model, [q1])

# Horizon
T = 21

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 100.0

_ul = zeros(model.m)
_ul[model.idx_u] .= -100.0
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T)#, x1 = [q1; q1])

# Objective
q_ref = linear_interp(q1, qT, T)
X0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(100.0, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [zeros(model.n, model.n) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
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
kinematics_1(model, q1, body = :torso, mode = :com)[2]
# torso height
l_stage_torso_h(x, u, t) = 1000.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - 0.9)^2.0
l_terminal_torso_h(x) = 1000.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[2] - 0.9)^2.0
obj_torso_h = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
l_stage_torso_lat(x, u, t) = (1.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(X0[t], 8:14), body = :torso, mode = :com)[1])^2.0)
l_terminal_torso_lat(x) = (1.0 * (kinematics_1(model, view(x, 8:14), body = :torso, mode = :com)[1] - kinematics_1(model, view(X0[T], 8:14), body = :torso, mode = :com)[1])^2.0)
obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
l_stage_fh1(x, u, t) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_1, mode = :ee)[2] - 0.05)^2.0
l_terminal_fh1(x) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_1, mode = :ee)[2])^2.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
l_stage_fh2(x, u, t) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_2, mode = :ee)[2] - 0.05)^2.0
l_terminal_fh2(x) = 10.0 * (kinematics_2(model, view(x, 8:14), body = :leg_2, mode = :ee)[2])^2.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# initial configuration
function l_stage_init(x, u, t)
    if t == 1
        return (x - [q1; q1])' * Diagonal(1000.0 * ones(model.n)) * (x - [q1; q1])
    else
        return 0.0
    end
end
l_terminal_init(x) = 0.0
obj_init = nonlinear_stage_objective(l_stage_init, l_terminal_init)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_torso_h,
                      obj_torso_lat,
                      obj_fh1,
                      obj_fh2,
                      obj_init])

# Constraints
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
               con = con
               )

# trajectory initialization
U0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

@time Z̄ = solve(prob, copy(Z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3)#, mapl = 5)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

# Visualize
visualize!(vis, model, state_to_configuration(X̄), Δt = h)

using Plots
plot(hcat(Ū...)[1:4,:]', label= "", linetype = :steppost)
