# Model
include_model("hopper")

# modify parameters to match implicit dynamics example
gravity = 9.81 # gravity
μ_world = 0.8 # coefficient of friction
μ_joint = 0.0

mb = 3.0 # body mass
ml = 0.3  # leg mass
Jb = 0.75 # body inertia
Jl = 0.075 # leg inertia

model = Hopper{Discrete, FixedTime}(n, m, d,
			   mb, ml, Jb, Jl,
			   μ_world, gravity,
			   qL, qU,
			   uL, uU,
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

# Horizon
T = 21
# Time step
h = 0.05
tf = h * (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 10.0 #model_ft.uU
_ul = zeros(model.m)
_ul[model.idx_u] .= -10.0 #model_ft.uL
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0; 0.5; 0.0; 0.5]
# qM = [0.5; 0.5; 0.0; 0.5]
qT = [1.0; 0.5; 0.0; 0.5]
q_ref = [0.5; 0.75; 0.0; 0.25]
x_ref = [q_ref; q_ref]
ql = [-Inf; 0; -Inf; 0.0]
qu = [Inf; Inf; Inf; 1.0]
xl, xu = state_bounds(model, T, [ql; ql], [qu; qu])
xl[1][nq .+ (1:nq)] = q1
xu[1][nq .+ (1:nq)] = q1

# Objective

# # gait 1
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [0.1 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

# gait 2
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [1.0 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

# gait 3
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])) for t = 1:T]
q = [2.0 * Q[t] * x_ref for t = 1:T]
R = [1.0e-3 * Diagonal(ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

obj_tracking = quadratic_tracking_objective(Q, R, q, r)

obj_contact_penalty = PenaltyObjective(1.0e4, model.m)

obj = MultiObjective([obj_tracking, obj_contact_penalty])

# Constraints
include_constraints(["contact", "stage", "loop"])

function pinned!(c, x, u, t)
    _q1 = view(x, 1:4)
    _q2 = view(x, 4 .+ (1:4))
    c[1:2] = kinematics(model, _q1) - kinematics(model, q1)
    c[3:4] = kinematics(model, _q2) - kinematics(model, q1)
    nothing
end
n_pinned = 4
con_pinned = stage_constraints(pinned!, n_pinned, (1:0), collect(1:1))

function distance_traveled!(c, x, u, t)
    x_travel = 0.5
    _q1 = view(x, 1:4)
    _q2 = view(x, 4 .+ (1:4))
	c[1] = x[1] - x_travel #TODO: add methods for trajectory-wise constraints in order to change to exactly match implicit dynamics example
    c[2] = x[5] - x_travel
    nothing
end

n_distance = 2
con_distance = stage_constraints(distance_traveled!, n_distance, (1:2), collect(T:T))

con_contact = contact_constraints(model, T)

con_loop = loop_constraints(model, collect([(2:model.nq)...,
	(nq .+ (2:model.nq))...]), 1, T)

con = multiple_constraints([con_contact, con_pinned, con_distance, con_loop])

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

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T]
@load joinpath(pwd(), "examples/implicit_dynamics/examples/comparisons/hopper_stand.jld2") u_stand
u0 = [u_stand[1] for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# include_snopt()

@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

x̄, ū = unpack(z̄, prob)
@show check_slack(z̄, prob)

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1_direct.jld2") x̄ ū
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1_direct.jld2") x̄ ū

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2_direct.jld2") x̄ ū
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2_direct.jld2") x̄ ū

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3_direct.jld2") x̄ ū
@load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3_direct.jld2") x̄ ū

# compute comparable objective
J = 0.0
x = x̄
u = ū
for t = 1:T
    if t < T
        J += x[t]' * Q[t] * x[t] + q[t]' * x[t] + u[t][1:2]' * R[t][1:2, 1:2] * u[t][1:2] + r[t][1:2]' * u[t][1:2]
    elseif t == T
        J += x[t]' * Q[t] * x[t] + q[t]' * x[t]
    else
        J += 0.0
    end
end
@show J

plot(hcat(ū...)[1:2, :]', linetype = :steppost)


using Plots
t = range(0, stop = h * (T - 1), length = T)
plot(t, hcat(ū..., ū[end])[1:2,:]', linetype=:steppost,
	xlabel="time (s)", ylabel = "control",
	label = ["angle" "length"],
	width = 2.0, legend = :top)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model,
    state_to_configuration(x̄),
	# state_to_configuration([[x̄[1] for i = 1:50]...,x̄..., [x̄[end] for i = 1:50]...]),
	Δt = h,
	scenario = :flip)
