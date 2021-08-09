using Plots

# Model
include_model("quadruped")
# model = free_time_model(model)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Horizon
T = 2

# Time step
h = 0.01

function initial_configuration(model::Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = pi / 2.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

θ1 = pi / 3.5
θ2 = pi / 3.5
θ3 = pi / 3.5

q1 = initial_configuration(model, θ1, θ2, θ3)
visualize!(vis, model, [q1])

# control
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 33.5 * h
# _uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -33.5 * h
# _ul[end] = 0.75 * h
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T,
    x1 = [q1; q1],
    xT = [q1; q1])

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e4, model.m)

obj = MultiObjective([obj_penalty])

# Constraints
include_constraints(["stage", "contact"])

con_contact = contact_constraints(model, T)

con = multiple_constraints([con_contact])#, con_loop, con_pinned1, con_pinned2])

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
q_ref = linear_interpolation(q1, q1, T+1)
x0 = configuration_to_state(q_ref)
u0 = [1.0e-3 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
    tol = 1.0e-2, c_tol = 1.0e-2,
	max_iter = 2000,
    time_limit = 60 * 2, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
q̄ = state_to_configuration(x̄)

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

u_stand = ū[1][1:model.m]
@save joinpath(@__DIR__, "quadruped_stand_100Hz.jld2") u_stand
