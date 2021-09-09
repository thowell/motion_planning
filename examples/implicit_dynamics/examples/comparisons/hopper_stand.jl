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
T = 5
# Time step
h = 0.05
tf = h * (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 1000.0 #model_ft.uU
_ul = zeros(model.m)
_ul[model.idx_u] .= -1000.0 #model_ft.uL
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0; 0.5; 0.0; 0.5]
# qM = [0.5; 0.5; 0.0; 0.5]
qT = q1#[1.0; 0.5; 0.0; 0.5]
q_ref = q1#[0.5; 0.75; 0.0; 0.25]
x_ref = [q_ref; q_ref]

xl, xu = state_bounds(model, T, [q1; q1], [q1; q1])

# Objective

# gait 1
obj_contact_penalty = PenaltyObjective(1.0e5, model.m)

obj = MultiObjective([obj_contact_penalty])

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

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T]
u0 = [[0.0; model.g * (model.mb + model.ml) * 0.5 * h; 0.0 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

fd(model, x0[1], x0[1], u0[1], zeros(model.d), h, 1)

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# include_snopt()

@time z̄, info = solve(prob, copy(z0),
	nlp = :ipopt,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

x̄, ū = unpack(z̄, prob)
u_stand = ū

@show check_slack(z̄, prob)

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

@save joinpath(pwd(), "examples/implicit_dynamics/examples/comparisons/hopper_stand.jld2") u_stand
