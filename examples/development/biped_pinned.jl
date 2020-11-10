using MeshCatMechanisms
include(joinpath(pwd(), "src/models/biped_pinned.jl"))
include(joinpath(pwd(), "src/constraints/loop_delta.jl"))


model = free_time_model(additive_noise_model(model))

function fd(model::BipedPinned, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end]) - w
end

# Visualize
include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)

urdf = joinpath(pwd(), "src/models/biped/urdf/biped_left_pinned.urdf")
mechanism = parse_urdf(urdf, floating=false)
mvis = MechanismVisualizer(mechanism,
    URDFVisuals(urdf, package_path=[dirname(dirname(urdf))]), vis)

ϵ = 1.0e-8
θ = 10 * pi / 180
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

set_configuration!(mvis, q1)

qT = transformation_to_urdf_left_pinned(model, xT[1:5])
set_configuration!(mvis, qT)

ζ = 11
xM = [π, π - ζ * pi / 180.0, 0.0, 2.0 * ζ * pi / 180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
qM = transformation_to_urdf_left_pinned(model, xM[1:5])
set_configuration!(mvis, qM)
kinematics(model, xM)[1]
kinematics(model, xM)[2]

x1_foot_des = kinematics(model, x1)[1]
xT_foot_des = kinematics(model, xT)[1]
xc = 0.5 * (x1_foot_des + xT_foot_des)

x1_foot_des * -1.0 + xT_foot_des
# r1 = x1_foot_des - xc
r1 = xT_foot_des - xc
r2 = 0.1

zM_foot_des = r2

function z_foot_traj(x)
    sqrt((1.0 - ((x - xc)^2.0) / (r1^2.0)) * (r2^2.0))
end

foot_x_ref = range(x1_foot_des, stop = xT_foot_des, length = T)
foot_z_ref = z_foot_traj.(foot_x_ref)

@assert norm(Δ(xT)[1:5] - x1[1:5]) < 1.0e-5

# Horizon
T = 21

tf0 = 1.0
h0 = tf0 / (T-1)

# Bounds
ul, uu = control_bounds(model, T,
	[-20.0 * ones(model.m - 1); 0.1 * h0],
	[20.0 * ones(model.m - 1); 2.0 * h0])
xl, xu = state_bounds(model, T, x1 = [x1[1:5]; zeros(5)])

# Objective
qq = 0.1 * [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal([1.0e-1 * ones(model.m - 1); h0]) for t = 1:T-1]

obj = quadratic_time_tracking_objective(
		Q,
		R,
    	[xT for t = 1:T],
		[zeros(model.m) for t = 1:T-1],
		1.0)

# Constraints
con_loop_1 = loop_delta_constraints(model, (1:5), 1, T)
# con_loop_2 = loop_delta_constraints(model, (1:model.n), 11, T)

con = multiple_constraints([con_loop_1])#, con_loop_2])

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

# Trajectory initialization
X0 = linear_interp(x1, xT, T) # linear interpolation on state
U0 = random_controls(model, T, 0.001) # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

# Solve
@time Z̄ = solve(prob, copy(Z0))

# Unpack solutions
X̄, Ū = unpack(Z̄, prob)
tf = sum([Ū[t][end] for t = 1:T-1])
t = range(0, stop = tf, length = T)

anim = Animation(mvis,
	range(0, stop = tf, length = T),
	[transformation_to_urdf_left_pinned(model, X̄[t]) for t = 1:T])

setanimation!(mvis, anim)
set_configuration!(mvis, transformation_to_urdf_left_pinned(model, X̄[1]))

Δ(X̄[T]) - X̄[1]
