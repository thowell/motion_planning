# Model
include_model("hopper")

# Dimensions
nq = 4 # configuration dimension
nu = 2 # control dimension
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone
nb = nc * nf
ns = nq

# Parameters
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 10.0 # body mass
ml = 1.0  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

n = 2 * nq
m = nu + nc + nb + nc + nb + ns
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

model = Hopper(n, m, d,
			   mb, ml, Jb, Jl,
			   μ, g,
			   qL, qU,
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

function fd(model::Hopper, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	s = view(u, model.idx_s)

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
	- M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	+ transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b)
	- h * G_func(model, q2⁺)) + s]
end

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_s] .= 0.0
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
_ul[model.idx_s] .= 0.0
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
z_h = 0.5
q1 = [0.0, 0.5, 0.0, 0.5]
v1 = [0.0, 0.0, 0.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q1]
qT = [1.0, 0.5, 0.0, 0.5]

xl, xu = state_bounds(model, T, [model.qL; model.qL], [model.qU; model.qU],
    x1 = x1, xT = [Inf*ones(model.nq); qT])

# Objective
include_objective(["velocity", "nonlinear_stage"])
obj_velocity = velocity_objective([Diagonal(ones(model.nq)) for t = 1:T],
	model.nq, idx_angle = (3:3), h = h)
obj_tracking = quadratic_tracking_objective(
    [Diagonal(zeros(model.n)) for t = 1:T],
    [Diagonal([1.0, 1.0, zeros(model.m - model.nu)...]) for t = 1:T-1],
    [zeros(model.n) for t = 1:T],
    [zeros(model.m) for t = 1:T-1])
function l_stage_fh1(x, u, t)
	if t > 1
		return 10.0 * (kinematics(model, view(x, 5:8))[2] - 0.25)^2.0
	else
		return 0.0
	end
end
l_terminal_fh1(x) = 0.0 #* (kinematics(model, view(x, 5:8)) - 0.025)^2.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

function l_stage_b(x, u, t)
	if t > 1
		return 10.0 * (x[6] - 0.5)^2.0
	else
		return 0.0
	end
end
l_terminal_b(x) = 0.0
obj_b = nonlinear_stage_objective(l_stage_b, l_terminal_b)

# obj_penalty = PenaltyObjective(1.0e5, model.m)
obj = MultiObjective([obj_tracking, obj_velocity, obj_fh1, obj_b])#, obj_penalty])

# Constraints
include_constraints("contact_al")
con_contact = contact_al_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact)

# Trajectory initialization
x0 = configuration_to_state(linear_interpolation(q1, qT, T)) # linear interpolation on state
u0 = [1.0e-3 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
# include_snopt()
# @time z̄ , info = solve(prob, copy(z0),
#  	nlp = :SNOPT7,
# 	tol = 1.0e-3, c_tol = 1.0e-3)
#
# check_slack(z̄, prob)
# x̄, ū = unpack(z̄, prob)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
