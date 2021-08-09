using Plots

# Model
include_model("flamingo")
model = Flamingo{Discrete, FixedTime}(n, m, d,
			  g_world, 0.5,
			  l_torso, d_torso, m_torso, J_torso,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
			  l_thigh, d_thigh, m_thigh, J_thigh,
			  l_calf, d_calf, m_calf, J_calf,
			  l_foot, d_foot, m_foot, J_foot,
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
			  idx_s,
			  0.0 * joint_friction)# model = free_time_model(model)

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
open(vis)
"""
Useful methods

joint-space inertia matrix:
	M_func(model, q)

position (px, pz) of foot 1 heel:
	kinematics_2(model,
		get_q⁺(x), body = :calf_1, mode = :ee)

position (px, pz) of foot 2 heel:
	kinematics_2(model,
		get_q⁺(x), body = :calf_2, mode = :ee)

position (px, pz) of foot 1 toe:
		kinematics_3(model,
	        get_q⁺(x), body = :foot_1, mode = :ee)

position (px, pz) of foot 2 toe:
		kinematics_3(model,
	        get_q⁺(x), body = :foot_2, mode = :ee)

jacobian to foot 1 heel
	jacobian_2(model, get_q⁺(x), body = :calf_1, mode = :ee)

jacobian to foot 2 heel
	jacobian_2(model, get_q⁺(x), body = :calf_2, mode = :ee)

jacobian to foot 1 toe
	jacobian_3(model, get_q⁺(x), body = :foot_1, mode = :ee)

jacobian to foot 2 toe
	jacobian_3(model, get_q⁺(x), body = :foot_2, mode = :ee)

gravity compensating torques
	initial_torque(model, q, h)[model.idx_u] # returns torques for system
	initial_torque(model, q, h) # returns torques for system and contact forces

	-NOTE: this method is iterative and is not differentiable (yet)
		-TODO: implicit function theorem to compute derivatives

contact models
	- maximum-dissipation principle:
		contact_constraints(model, T)
	- no slip:
		contact_no_slip_constraints(T)

loop constraint
	- constrain variables at two indices to be the same:
		loop_constraints(model, x_idx, t1_idx, t2_idx)
"""

function get_q⁺(x)
	view(x, model.nq .+ (1:model.nq))
end

function ellipse_traj(x_start, x_goal, z, T)
	dist = x_goal - x_start
	a = 0.5 * dist
	b = z

	z̄ = 0.0

	x = circular_projection_range(x_start, stop = x_goal, length = T)
	# x = range(x_start, stop = x_goal, length = T)

	z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))

	return x, z
end

function circular_projection_range(start; stop=1.0, length=10)
	dist = stop - start
	θr = range(π, stop=0, length=length)
	r = start .+ dist * ((1 .+ cos.(θr))./2)
	return r
end


# Horizon
T = 2
h = 0.01

# Permutation matrix
perm = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
				 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
				 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to downward vertical)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to downward vertical)
# 8: foot 1 (rel. to downward vertical)
# 9: foot 2 (rel. to downward vertical)
function initial_configuration(model, θ_torso, θ_thigh_1, θ_leg_1, θ_thigh_2)
    q1 = zeros(model.nq)
    q1[3] = θ_torso
    q1[4] = θ_thigh_1
    q1[5] = θ_leg_1
    z1 = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])
    q1[6] = θ_thigh_2
    q1[7] = 1.0 * acos((z1 - model.l_thigh2 * cos(q1[6])) / model.l_calf2)
	q1[2] = z1
	q1[8] = pi / 2.0
	q1[9] = pi / 2.0

    return q1
end

# q1 = initial_configuration(model, -π / 50.0, -π / 6.5, π / 10.0 , -π / 10.0)
# q1 = initial_configuration(model, -0.00*π / 50.0, -π / 5.6, π / 25.0 , -π / 7.0)
q1 = initial_configuration(model, -0.00*π / 50.0, -π / 6.0, π / 12.0 , -π / 6.0)
pf1 = kinematics_3(model, q1, body = :foot_1, mode = :com)
pf2 = kinematics_3(model, q1, body = :foot_2, mode = :com)

ph1 = kinematics_3(model, q1, body = :foot_1, mode = :heel)
ph2 = kinematics_3(model, q1, body = :foot_2, mode = :heel)[1]
ph2 = kinematics_2(model, q1, body = :calf_2, mode = :ee)[1]
q1[1]

strd = 2 * (pf2 - pf1)[1]

qT = Array(perm) * copy(q1)
qT[1] += 0.5 * strd
q_ref = linear_interpolation(q1, qT, T+1)
x_ref = configuration_to_state(q_ref)
visualize!(vis, model, q_ref, Δt = h)

# Control

# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
# _uu[end] = 2.0 * h
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
# _ul[end] = 0.25 * h
ul, uu = control_bounds(model, T, _ul, _uu)

qL = [-Inf; -Inf; q1[3] - pi / 750.0; q1[4:end] .- pi / 6.0; -Inf; -Inf; q1[3] - pi / 50.0; q1[4:end] .- pi / 6.0]
qU = [Inf; q1[2] + 0.001; q1[3] + pi / 750.0; q1[4:end] .+ pi / 6.0; Inf; Inf; q1[3:end] .+ pi / 6.0]
qL[8] = q1[8] - pi / 10.0
qL[9] = q1[9] - pi / 10.0
qL[9 + 8] = q1[8] - pi / 10.0
qL[9 + 9] = q1[9] - pi / 10.0

qU[8] = q1[8] + pi / 10.0
qU[9] = q1[9] + pi / 10.0
qU[9 + 8] = q1[8] + pi / 10.0
qU[9 + 9] = q1[9] + pi / 10.0

xl, xu = state_bounds(model, T,
    qL, qU,
    x1 = [q1; q1],
	xT = [q1; q1])

# Objective
include_objective(["velocity", "nonlinear_stage", "control_velocity"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e5, model.m)

obj = MultiObjective([obj_penalty])

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
u0 = [[1.0e-2 * randn(model.nu); 0.01 * randn(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
# NOTE: run multiple times to get good trajectory
@time z̄, info = solve(prob, copy(z0),
    nlp = :ipopt,
	max_iter = 2000,
    tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
    time_limit = 60 * 3)

@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
# _tf, _t, h̄ = get_time(ū)

q = state_to_configuration(x̄)
u = [u[model.idx_u] for u in ū]
γ = [u[model.idx_λ] for u in ū]
b = [u[model.idx_b] for u in ū]
ψ = [u[model.idx_ψ] for u in ū]
η = [u[model.idx_η] for u in ū]

plot(u, linetype = :steppost)

μ = model.μ
@save joinpath(@__DIR__, "flamingo_stand_100hz.jld2") q u γ b ψ η μ h

vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, q, Δt = h)
