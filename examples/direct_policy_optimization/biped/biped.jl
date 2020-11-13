include(joinpath(pwd(), "src/models/biped_pinned.jl"))
include(joinpath(pwd(), "src/objectives/nonlinear_stage.jl"))
include(joinpath(pwd(), "src/constraints/loop_delta.jl"))
include(joinpath(pwd(), "src/constraints/free_time.jl"))
include(joinpath(pwd(), "src/constraints/stage.jl"))

model = free_time_model(additive_noise_model(model))

function fd(model::BipedPinned, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end]) - w
end

# # Visualize
# include(joinpath(pwd(), "src/models/visualize.jl"))
# vis = Visualizer()
# open(vis)
#
# urdf = joinpath(pwd(), "src/models/biped/urdf/biped_left_pinned.urdf")
# mechanism = parse_urdf(urdf, floating=false)
# mvis = MechanismVisualizer(mechanism,
#     URDFVisuals(urdf, package_path=[dirname(dirname(urdf))]), vis)

ϵ = 1.0e-8
θ = 12.5 * pi / 180
h = model.l2 + model.l1 * cos(θ)
ψ = acos(h / (model.l1 + model.l2))
stride = sin(θ) * model.l1 + sin(ψ) * (model.l1 + model.l2)
x1 = [π - θ, π + ψ, θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xT = [π + ψ, π - θ - ϵ, 0.0, θ, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
kinematics(model, x1)[1]
kinematics(model, xT)[1] * 2.0

kinematics(model, x1)[2]
kinematics(model, xT)[2]

q1 = transformation_to_urdf_left_pinned(model, x1[1:5])
# set_configuration!(mvis, q1)

qT = transformation_to_urdf_left_pinned(model, xT[1:5])
# set_configuration!(mvis, qT)

# Horizon
T = 21

tf0 = 2.0
h0 = tf0 / (T-1)

# Bounds
ul, uu = control_bounds(model, T,
	[-10.0 * ones(model.m - 1); 0.0 * h0],
	[10.0 * ones(model.m - 1); h0])

xl, xu = state_bounds(model, T, x1 = [x1[1:5]; Inf * ones(5)])

# Objective
qq = 1.0 * [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Q = [Diagonal(qq) for t = 1:T]
R = [Diagonal([1.0e-1 * ones(model.m - 1); h0]) for t = 1:T-1]

obj_track = quadratic_time_tracking_objective(
		Q,
		R,
    	[xT for t = 1:T],
		[zeros(model.m) for t = 1:T-1],
		1.0)

l_stage_fh(x, u, t) = 100.0 * (kinematics(model, view(x, 1:5))[2] - 0.25)^2.0
l_terminal_fh(x) = 0.0
obj_fh = nonlinear_stage_objective(l_stage_fh, l_terminal_fh)

obj_multi = MultiObjective([obj_track, obj_fh])

# Constraints
con_loop = loop_delta_constraints(model, (1:model.n), 1, T)
con_free_time = free_time_constraints(T)

con = multiple_constraints([con_loop, con_free_time])#, con_pinned_foot])

# Problem
prob = trajectory_optimization_problem(model,
           obj_multi,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu,
		   con = con
           )

# Trajectory initialization
x0 = linear_interp(x1, xT, T) # linear interpolation on state
u0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
optimize = true

if optimize
	include_snopt()
	@time z̄ = solve(prob, copy(z0),
		nlp = :SNOPT7,
		time_limit = 60 * 10)
	@save joinpath(@__DIR__, "sol_to.jld2") z̄
else
	println("Loading solution...")
	@load joinpath(@__DIR__, "sol_to.jld2") z̄
end


# Unpack solutions
x̄, ū = unpack(z̄, prob)
@show tf = sum([ū[t][end] for t = 1:T-1])
# visualize!(mvis, model, x̄, Δt = ū[1][end])
