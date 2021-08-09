using Plots
using Random
Random.seed!(1)

include_ddp()

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

function visualize!(vis, model, q;
      r = 0.025, Δt = 0.1)

	default_background!(vis)

	torso = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l_torso),
		convert(Float32, 0.035))
	setobject!(vis["torso"], torso,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh1),
		convert(Float32, 0.0175))
	setobject!(vis["thigh1"], thigh_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_1 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf1),
		convert(Float32, 0.0125))
	setobject!(vis["leg1"], calf_1,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh2),
		convert(Float32, 0.0175))
	setobject!(vis["thigh2"], thigh_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf2),
		convert(Float32, 0.0125))
	setobject!(vis["leg2"], calf_2,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh3),
		convert(Float32, 0.0175))
	setobject!(vis["thigh3"], thigh_3,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_3 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf3),
		convert(Float32, 0.0125))
	setobject!(vis["leg3"], calf_3,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	thigh_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_thigh4),
		convert(Float32, 0.0175))
	setobject!(vis["thigh4"], thigh_4,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	calf_4 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l_calf4),
		convert(Float32, 0.0125))
	setobject!(vis["leg4"], calf_4,
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	hip1 = setobject!(vis["hip1"], Sphere(Point3f0(0),
        convert(Float32, 0.035)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	hip2 = setobject!(vis["hip2"], Sphere(Point3f0(0),
        convert(Float32, 0.035)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee1 = setobject!(vis["knee1"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee2 = setobject!(vis["knee2"], Sphere(Point3f0(0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee3 = setobject!(vis["knee3"], Sphere(Point3f0(0),
		convert(Float32, 0.025)),
		MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	knee4 = setobject!(vis["knee4"], Sphere(Point3f0(0),
        convert(Float32, 0.025)),
        MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))

	feet1 = setobject!(vis["feet1"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet2 = setobject!(vis["feet2"], Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet3 = setobject!(vis["feet3"], Sphere(Point3f0(0),
		convert(Float32, r)),
		MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	feet4 = setobject!(vis["feet4"], Sphere(Point3f0(0),
        convert(Float32, r)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

	T = length(q)
	p_shift = [0.0, 0.0, r]
	for t = 1:T
		MeshCat.atframe(anim, t) do
			p = [q[t][1]; 0.0; q[t][2]] + p_shift

			k_torso = kinematics_1(model, q[t], body = :torso, mode = :ee)
			p_torso = [k_torso[1], 0.0, k_torso[2]] + p_shift

			k_thigh_1 = kinematics_1(model, q[t], body = :thigh_1, mode = :ee)
			p_thigh_1 = [k_thigh_1[1], 0.0, k_thigh_1[2]] + p_shift

			k_calf_1 = kinematics_2(model, q[t], body = :calf_1, mode = :ee)
			p_calf_1 = [k_calf_1[1], 0.0, k_calf_1[2]] + p_shift

			k_thigh_2 = kinematics_1(model, q[t], body = :thigh_2, mode = :ee)
			p_thigh_2 = [k_thigh_2[1], 0.0, k_thigh_2[2]] + p_shift

			k_calf_2 = kinematics_2(model, q[t], body = :calf_2, mode = :ee)
			p_calf_2 = [k_calf_2[1], 0.0, k_calf_2[2]] + p_shift


			k_thigh_3 = kinematics_2(model, q[t], body = :thigh_3, mode = :ee)
			p_thigh_3 = [k_thigh_3[1], 0.0, k_thigh_3[2]] + p_shift

			k_calf_3 = kinematics_3(model, q[t], body = :calf_3, mode = :ee)
			p_calf_3 = [k_calf_3[1], 0.0, k_calf_3[2]] + p_shift

			k_thigh_4 = kinematics_2(model, q[t], body = :thigh_4, mode = :ee)
			p_thigh_4 = [k_thigh_4[1], 0.0, k_thigh_4[2]] + p_shift

			k_calf_4 = kinematics_3(model, q[t], body = :calf_4, mode = :ee)
			p_calf_4 = [k_calf_4[1], 0.0, k_calf_4[2]] + p_shift

			settransform!(vis["thigh1"], cable_transform(p, p_thigh_1))
			settransform!(vis["leg1"], cable_transform(p_thigh_1, p_calf_1))
			settransform!(vis["thigh2"], cable_transform(p, p_thigh_2))
			settransform!(vis["leg2"], cable_transform(p_thigh_2, p_calf_2))
			settransform!(vis["thigh3"], cable_transform(p_torso, p_thigh_3))
			settransform!(vis["leg3"], cable_transform(p_thigh_3, p_calf_3))
			settransform!(vis["thigh4"], cable_transform(p_torso, p_thigh_4))
			settransform!(vis["leg4"], cable_transform(p_thigh_4, p_calf_4))
			settransform!(vis["torso"], cable_transform(p, p_torso))
			settransform!(vis["hip1"], Translation(p))
			settransform!(vis["hip2"], Translation(p_torso))
			settransform!(vis["knee1"], Translation(p_thigh_1))
			settransform!(vis["knee2"], Translation(p_thigh_2))
			settransform!(vis["knee3"], Translation(p_thigh_3))
			settransform!(vis["knee4"], Translation(p_thigh_4))
			settransform!(vis["feet1"], Translation(p_calf_1))
			settransform!(vis["feet2"], Translation(p_calf_2))
			settransform!(vis["feet3"], Translation(p_calf_3))
			settransform!(vis["feet4"], Translation(p_calf_4))
		end
	end

	settransform!(vis["/Cameras/default"],
	    compose(Translation(0.0, 0.0, -1.0), LinearMap(RotZ(-pi / 2.0))))

	MeshCat.setanimation!(vis, anim)
end

contact_control_path = "/home/taylor/Research/ContactControl.jl/src"

using Parameters
# Utilities
include(joinpath(contact_control_path, "utils.jl"))

# Solver
include(joinpath(contact_control_path, "solver/cones.jl"))
include(joinpath(contact_control_path, "solver/interior_point.jl"))
include(joinpath(contact_control_path, "solver/lu.jl"))

# Environment
include(joinpath(contact_control_path, "simulator/environment.jl"))

# Dynamics
include(joinpath(contact_control_path, "dynamics/model.jl"))

# Simulator
include(joinpath(contact_control_path, "simulation/contact_methods.jl"))
include(joinpath(contact_control_path, "simulation/simulation.jl"))
include(joinpath(contact_control_path, "simulator/trajectory.jl"))

include(joinpath(contact_control_path, "dynamics/code_gen_dynamics.jl"))
include(joinpath(contact_control_path, "dynamics/fast_methods_dynamics.jl"))

# Models
include(joinpath(contact_control_path, "dynamics/quaternions.jl"))
include(joinpath(contact_control_path, "dynamics/mrp.jl"))
include(joinpath(contact_control_path, "dynamics/euler.jl"))

# include("dynamics/particle_2D/model.jl")
# include("dynamics/particle/model.jl")
# include(joinpath(contact_control_path, "dynamics/hopper_2D/model.jl"))
# include("dynamics/hopper_3D/model.jl")
# include("dynamics/hopper_3D_quaternion/model.jl")
include(joinpath(contact_control_path, "dynamics/quadruped/model.jl"))
# include("dynamics/quadruped_simple/model.jl")
# include("dynamics/biped/model.jl")
# include(joinpath(contact_control_path, "dynamics/flamingo/model.jl"))
# include("dynamics/pushbot/model.jl")
# include("dynamics/planarpush/model.jl")
# include("dynamics/planarpush_2D/model.jl")
# include("dynamics/rigidbody/model.jl")
# include("dynamics/box/model.jl")

# Simulation
include(joinpath(contact_control_path, "simulation/environments/flat.jl"))
# include("simulation/environments/piecewise.jl")
# include("simulation/environments/quadratic.jl")
# include("simulation/environments/slope.jl")
# include("simulation/environments/sinusoidal.jl")
# include("simulation/environments/stairs.jl")

include(joinpath(contact_control_path, "simulation/residual_approx.jl"))
include(joinpath(contact_control_path, "simulation/code_gen_simulation.jl"))

# Visuals
using MeshCatMechanisms
include(joinpath(contact_control_path, "dynamics/visuals.jl"))
include(joinpath(contact_control_path, "dynamics/visual_utils.jl"))

s = get_simulation("quadruped", "flat_2D_lc", "flat")

# @load joinpath(pwd(), "examples/contact_implicit", "flamingo_stand_100hz.jld2") q u γ b ψ η μ h
@load joinpath(pwd(), "examples/contact_implicit", "quadruped_gait_100Hz.jld2") qm um γm bm ψm ηm μm hm
visualize!(vis, s.model, qm, Δt = h)

nq = s.model.dim.q
m = s.model.dim.u

function initial_configuration(model::Quadruped, θ1, θ2, θ3)
    q1 = zeros(nq)
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

q1 = initial_configuration(s.model, θ1, θ2, θ3)
q0 = copy(q1)
visualize!(vis, s.model, [q1])

strd_diff = kinematics_2(s.model, qm[1], body = :calf_2, mode = :ee)[1] - kinematics_2(s.model, q1, body = :calf_2, mode = :ee)[1]

q0[1] += strd_diff
q1[1] += strd_diff

T = 5
h = 0.01

qT1 = qm[1]
qT = qm[2]

x1 = [q0; q1]
# xT = [q0; q1]#
xT = [qT1; qT]
visualize!(vis, s.model, [q0, q1], Δt = h)

visualize!(vis, s.model, [q1, qT], Δt = 1.0)

visualize!(vis, s.model, [qT1, qT], Δt = h)

struct Dynamics{T}
	s::Simulation
	ip_dyn::InteriorPoint
	ip_jac::InteriorPoint
	h::T
end

function gen_dynamics(s::Simulation, h;
		dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true))

	z = zeros(num_var(s.model, s.env))
	θ = zeros(num_data(s.model))

	ip_dyn = interior_point(z, θ,
		idx_ineq = inequality_indices(s.model, s.env),
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = dyn_opts)

	ip_dyn.opts.diff_sol = false

	ip_jac = interior_point(z, θ,
		idx_ineq = inequality_indices(s.model, s.env),
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = jac_opts)

	ip_jac.opts.diff_sol = true

	Dynamics(s, ip_dyn, ip_jac, h)
end

d = gen_dynamics(s, h,
	dyn_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-5, κ_init = 0.1),
	jac_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-5, κ_init = 0.1))

function f!(d::Dynamics, q0, q1, u1, mode = :dynamics)
	s = d.s
	ip = (mode == :dynamics ? d.ip_dyn : d.ip_jac)
	h = d.h

	z_initialize!(ip.z, s.model, s.env, copy(q1))
	θ_initialize!(ip.θ, s.model, copy(q0), copy(q1), copy(u1), zeros(s.model.dim.w), s.model.μ_world, h)

	status = interior_point_solve!(ip)

	!status && (@warn "dynamics failure")
end

function f(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :dynamics)
	return copy(d.ip_dyn.z[1:d.s.model.dim.q])
end

f(d, q0, q1, zeros(m))

function fq0(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 1:d.s.model.dim.q])
end

fq0(d, q0, q1, zeros(m))

function fq1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, d.s.model.dim.q .+ (1:d.s.model.dim.q)])
end

fq1(d, q0, q1, zeros(m))

function fx1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 1:(2 * d.s.model.dim.q)])
end

fx1(d, q0, q1, zeros(m))

function fu1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 2 * d.s.model.dim.q .+ (1:d.s.model.dim.u)])
end

fu1(d, q0, q1, zeros(m))

struct QuadrupedCI{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::Dynamics
end

model = QuadrupedCI{Midpoint, FixedTime}(2 * s.model.dim.q, s.model.dim.u, 0, d)
u1 = zeros(model.m)

function fd(model::QuadrupedCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]

	q2 = f(model.dynamics, q0, q1, u)

	return [q1; q2]
end

fd(model, x1, u1, zeros(0), h, 1)

function fdx(model::QuadrupedCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2dx1 = fx1(model.dynamics, q0, q1, u)

	return [zeros(nq, nq) I; dq2dx1]
end

fdx(model, x1, u1, zeros(0), h, 1)


function fdu(model::QuadrupedCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2du1 = fu1(model.dynamics, q0, q1, u)
	return [zeros(nq, model.m); dq2du1]
end

fdu(model, x1, u1, zeros(0), h, 1)

n = model.n
m = model.m

T = 20
# ū = [u[1] + 0.0 * randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
# @save joinpath(pwd(), "examples/contact_implicit", "flamingo_stand_100hz_v2.jld2") u_stand
@load joinpath(pwd(), "examples/contact_implicit", "quadruped_stand_100Hz.jld2") u_stand
u_stand = u_stand[1:m]
ū = [u_stand + 0.00 * randn(model.m) for t = 1:T-1]

x̄ = rollout(model, x1, ū, w, h, T)

q̄ = state_to_configuration(x̄)
visualize!(vis, s.model, q̄, Δt = h)

# Objective
Q = [t < T ? 1.0e-1 * Diagonal(ones(model.n)) : 10.0 * Diagonal(ones(model.n)) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]
R = [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
r = [-2.0 * R[t] * u_stand for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
ul = -1.0 * ones(model.m)
uu = 1.0 * ones(model.m)
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	else
		c .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 100, max_al_iter = 2,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-2)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)

vis = Visualizer()
render(vis)
visualize!(vis, s.model, q̄, Δt = h)

plot(hcat(ū...)', linetype = :steppost)

norm(x̄[end] - xT, Inf)
