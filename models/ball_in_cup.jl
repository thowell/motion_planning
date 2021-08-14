using MeshCat, MeshCatMechanisms, RigidBodyDynamics
using FileIO, MeshIO, GeometryBasics, CoordinateTransformations, Rotations, Meshing
using Plots

include_ddp()

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

# Simulation
include(joinpath(contact_control_path, "simulation/environments/flat.jl"))

include(joinpath(contact_control_path, "simulation/residual_approx.jl"))
include(joinpath(contact_control_path, "simulation/code_gen_simulation.jl"))

# Visuals
using MeshCatMechanisms
include(joinpath(contact_control_path, "dynamics/visuals.jl"))
include(joinpath(contact_control_path, "dynamics/visual_utils.jl"))


include(joinpath(pwd(),"models/kuka/kuka_utils.jl"))

urdf_path = joinpath(pwd(), "models/kuka/temp/kuka.urdf")

kuka = MeshCatMechanisms.parse_urdf(urdf_path, remove_fixed_tree_joints = true)
kuka_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(kuka)
state_cache = StateCache(kuka)
result = DynamicsResult(kuka)
result_cache = DynamicsResultCache(kuka)

vis = Visualizer()
mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
render(vis)
default_background!(vis)

nq = num_positions(kuka)
q_init = zeros(nq)
q_init[1] = 0
q_init[2] = pi/4

q_init[4] = -pi/2
q_init[5] = 0
q_init[6] = -pi/4
q_init[7] = 0
set_configuration!(state, q_init)
set_configuration!(mvis, q_init)
ee = findbody(kuka, "iiwa_link_7")
ee_point = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
setelement!(mvis, ee_point, 0.05)
ee_jacobian_frame = ee_point.frame
ee_jacobian_path = path(kuka, root_body(kuka), ee)

world = root_frame(kuka)
ee_in_world = transform(state, ee_point, world).v
desired = Point3D(world, 0.5, 0.0, 0.0)
q_res1 = jacobian_transpose_ik!(state, ee, ee_point, desired,
    visualize = true, mvis = mvis)
q_res1 = Array(q_res1)
set_configuration!(mvis, q_res1)
set_configuration!(state, q_res1)
ee_in_world = transform(state, ee_point, world).v

desired = Point3D(world, 0.75, 0.0, 0.0)
q_res2 = jacobian_transpose_ik!(state, ee, ee_point, desired,
    visualize = true, mvis = mvis)
q_res2 = Array(q_res2)
set_configuration!(mvis, q_res2)
set_configuration!(state, q_res2)
ee_in_world = transform(state, ee_point, world).v

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
# + 3D particle
struct KukaParticle <: ContactModel
	dim::Dimensions

	# particle
	mp

    state_cache1
    state_cache2
    state_cache3

    result_cache1
    result_cache2
    result_cache3

    world

	ee
    ee_point
    ee_jacobian_frame
    ee_jacobian_path
end

# Dimensions
nq = 7 + 3 # configuration dim
nu = 7 # control dim
nc = 1 # number of contact points
nf = 0 # number of faces for friction cone
nb = 0
ns = 0

n = 2 * nq
m_contact = 0 #nc + nb + nc + nb + 1
m = nu #nu + m_contact
d = 0

mp = 0.01

results_cache1 = DynamicsResultCache(kuka)
results_cache2 = DynamicsResultCache(kuka)
results_cache3 = DynamicsResultCache(kuka)

state_cache1 = StateCache(kuka)
state_cache2 = StateCache(kuka)
state_cache3 = StateCache(kuka)

function kuka_q(q)
	view(q, 1:7)
end

function particle_q(q)
	view(q, 8:10)
end

# Methods
function M_func(m::KukaParticle, q::AbstractVector{T}) where T
    # return Diagonal(m.mp * ones(3))
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state, kuka_q(q))
    mass_matrix!(result.massmatrix, state)
    return SMatrix{10, 10}(cat(result.massmatrix, Diagonal(m.mp * ones(3)), dims = (1, 2)))
end

function C_func(m::KukaParticle, q, q̇)
    # return SVector{3}([0.0, 0.0, m.mp * 9.81])
    if eltype(q) <: ForwardDiff.Dual
		return SVector{10}([dynamics_bias_q(m, kuka_q(q), kuka_q(q̇))..., 0.0, 0.0, m.mp * 9.81])
	elseif eltype(q̇) <: ForwardDiff.Dual
		return SVector{10}([dynamics_bias_q̇(m, kuka_q(q), kuka_q(q̇))..., 0.0 ,0.0, m.mp * 9.81])
	else
		return SVector{10}([_dynamics_bias(m, kuka_q(q), kuka_q(q̇))..., 0.0, 0.0, m.mp * 9.81])
	end
end

function _dynamics_bias(m::KukaParticle, q, q̇)
	T = eltype(m.mp)
	state = m.state_cache1[T]
	result = m.result_cache1[T]
	set_configuration!(state, q)
	set_velocity!(state, q̇)
	dynamics_bias!(result, state)
    return result.dynamicsbias
end

function dynamics_bias_q(m::KukaParticle, q::AbstractVector{T}, q̇) where T
    state = m.state_cache2[T]
    result = m.result_cache2[T]
    set_configuration!(state, q)
    set_velocity!(state,q̇)
    dynamics_bias!(result, state)
    return result.dynamicsbias
end

function dynamics_bias_q̇(m::KukaParticle, q, q̇::AbstractVector{T}) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state, q)
    set_velocity!(state, q̇)
    dynamics_bias!(result, state)
    return result.dynamicsbias
end

function B_func(m::KukaParticle, q)
    # return zeros(3, 3)
    SMatrix{10, 7}([Diagonal(ones(7)); zeros(3, 7)])
end

function kinematics_ee(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state, kuka_q(q))
    return transform(state, m.ee_point, m.world).v
end

function ϕ_func(m::KukaParticle, q::AbstractVector{T}) where T
	p_ee = kinematics_ee(m, q)
	p_p = q[8:10]
	diff = (p_ee - p_p)
	d_ee_p = sqrt(diff' * diff)
    SVector{1}([0.5 - d_ee_p])
end

function P_func(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]

    set_configuration!(state, kuka_q(q))

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T, 3, 7))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path,ee_in_world) #TODO confirm this is correct

	p_ee = ee_in_world.v
	p_p = q[8:10]
	diff = (p_ee - p_p)

    return Array(-1.0 * 2.0 * diff' * [pj1.J[1:3,:] -1.0 * Diagonal(ones(3))])
end

ϕ(z) = ϕ_func(_model, z)
norm(ForwardDiff.jacobian(ϕ, q0) - P_func(_model, q1))

nq = 7 + 3
nu = 7
nw = 0
nc = 1

h = 0.1

_model = KukaParticle(
	Dimensions(nq, nu, nw, nc),
	mp,
    state_cache1, state_cache2, state_cache3,
    results_cache1, results_cache2, results_cache3,
    world,
	ee,
	ee_point,
	ee_jacobian_frame,
	ee_jacobian_path)

q0 = zeros(_model.dim.q)
q0[1] = 0
q0[3] = 0
q0[4] = -pi/2
q0[5] = 0.0
p_pos0 = Array(kinematics_ee(_model, q0))
p_pos0[1] += 0.0
p_pos0[3] -= 0.5
q0[8:10] = p_pos0

# visualize!(mvis, _model, [q0], Δt = h)

q1 = copy(q0)

qN = zeros(_model.dim.q)
qN[1] = 0
qN[3] = 0
qN[4] = -pi/2
qN[5] = 0.

ee_posN = Array(kinematics_ee(_model, qN))
p_posN = ee_posN
p_posN[1] += 0.0
p_posN[3] += 0.1
qN[8:10] = p_posN

qD = zeros(_model.dim.q)
qD[1] = 0
qD[3] = 0
qD[4] = -pi/2
qD[5] = 0.

ee_posD = Array(kinematics_ee(_model, qD))
p_posD = ee_posD
# p_posD[1] += 0.5
qD[8:10] = p_posD

visualize!(mvis, _model, [qD], Δt = h)

# Visualization
function visualize!(mvis, model::KukaParticle, q;
		verbose = false, r_ball = 0.035, Δt = 0.1)

	setobject!(vis["ball"], Sphere(Point3f0(0),
				convert(Float32, r_ball)),
				MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0,1.0)))

	settransform!(vis["ball"], compose(Translation(0.66,3.0,0.0)))

	state = model.state_cache1[Float64]

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T
        q_kuka = kuka_q(q[t])
		q_particle = particle_q(q[t])
		set_configuration!(state, kuka_q(q[t]))

        MeshCat.atframe(anim,t) do
			set_configuration!(mvis,q_kuka)
            settransform!(vis["ball"], compose(Translation(q_particle), LinearMap(RotZ(0))))
		end
    end

    MeshCat.setanimation!(vis, anim)
end

include(joinpath(contact_control_path, "simulator/environment.jl"))
include(joinpath(contact_control_path, "simulation/environments/flat.jl"))

env = flat_3D_lc

function lagrangian_derivatives(model::KukaParticle, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function gravity_compensation(model::KukaParticle, h, q0, q1)
    q2 = q1
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	# return 0.0
	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2)[1:7]
end

function dynamics(model::KukaParticle, h, q0, q1, u1, λ1, q2)
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    u_gravity = gravity_compensation(model, h, q0, q1)

	# return 0.0
	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * (u1 - u_gravity)
        + transpose(P_func(model, q2)) * λ1)
        # - h[1] * 0.5 .* vm2)
end

function residual(model::KukaParticle, env::Environment{<:World,LinearizedCone}, z, θ, κ)
	# nc = model.dim.c
	# nb = nc * friction_dim(env)
	# nf = Int(nb / nc)
	# np = dim(env)

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    λ1 = z[nq .+ (1:nc)]
    s1 = z[nq + nc .+ (1:nc)]

	# q0, q1, u1, w1, μ, h = unpack_θ(model, θ)
	# q2, γ1, b1, ψ1, η1, s1, s2 = unpack_z(model, env, z)

	# ϕ = ϕ_func(model, env, q2)
    #
	# k = kinematics(model, q2)
	# λ1 = contact_forces(model, env, γ1, b1, q2, k)
	# Λ1 = transpose(J_func(model, env, q2)) * λ1 #@@@@ maybe need to use J_fast
	# vT_stack = velocity_stack(model, env, q1, q2, k, h)
	# ψ_stack = transpose(E_func(model, env)) * ψ1
    # return dynamics(model, h, q0, q1, u1, zeros(nc), q2)

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
     s1 .- ϕ_func(model, q2);
     λ1 .* s1 .- κ;
    ]
	# [
    #  dynamics(model, h, q0, q1, u1, q2);
	#  # s1 - ϕ;
	#  # vT_stack + ψ_stack - η1;
	#  # s2 .- (μ[1] * γ1 .- E_func(model, env) * b1);
	#  # γ1 .* s1 .- κ;
	#  # b1 .* η1 .- κ;
	#  # ψ1 .* s2 .- κ
    #  ]
end

nz = nq + nc + nc
nθ = nq + nq + nu + 1


u0 = zeros(_model.dim.u)
z0 = copy([q1; 0.1 * ones(2 * nc)])
θ0 = copy([q0; q1; u0; h])

residual(_model, env, z0, θ0, [1.0])

function r_func(r, z, θ, κ)
    r .= residual(_model, env, z, θ, κ)
end

function rz_func(rz, z, θ)
    r(a) = residual(_model, env, a, θ, 0.0)
    rz .= ForwardDiff.jacobian(r, z)
end

function rθ_func(rθ, z, θ)
    r(a) = residual(_model, env, z, a, 0.0)
    rθ .= ForwardDiff.jacobian(r, θ)
end

r0 = zeros(nz)
rz0 = zeros(nz, nz)
rθ0 = zeros(nz, nθ)
r_func(r0, z0, θ0, [1.0])
rz_func(rz0, z0, θ0)
rθ_func(rθ0, z0, θ0)

# options
opts = InteriorPointOptions(
   κ_init = 0.1,
   κ_tol = 1.0e-4,
   r_tol = 1.0e-8,
   diff_sol = true)

idx_ineq = collect([nq .+ (1:(nc + nc))]...)

# solver
ip = interior_point(z0, θ0,
   r! = r_func, rz! = rz_func,
   rz = similar(rz0, Float64),
   rθ! = rθ_func,
   rθ = similar(rθ0, Float64),
   idx_ineq = idx_ineq,
   opts = opts)

# simulate
T = 26
q_hist = [q0, q0]
u_hist = []
h = 0.1
for t = 1:T-1
    u_gravity = zeros(_model.dim.u)#-1.0 * gravity_compensation(_model, h, q_hist[end-1], q_hist[end])
    push!(u_hist, 0.0 * u_gravity)
    # if t == 5
    #     u_gravity[6] = -1.0
    # end
    # u_gravity = -1.0 * gravity_compensation(_model, h, q0, q1)

    ip.z .= copy([q_hist[end]; 0.1 * ones(nc + nc)])
    ip.θ .= copy([q_hist[end-1]; q_hist[end];
        # u0;
        u_hist[end];
        h])
    status = interior_point_solve!(ip)

    if status
        push!(q_hist, ip.z[1:nq])
    else
        println("dynamics failure t = $t")
        println("res norm = $(norm(ip.r, Inf))")
        break
    end
end

vz_particle = [(((q_hist[t+2] - q_hist[t+1]) / h - (q_hist[t+1] - q_hist[t]) / h) / h)[10] for t = 1:T-2]

visualize!(mvis, _model, q_hist, Δt = h)

s = Simulation(_model, env)
s.res.r! = r_func
s.res.rz! = rz_func
s.res.rθ! = rθ_func
s.rz = ip.rz
s.rθ = ip.rθ

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

	z = zeros(nz)
	θ = zeros(nθ)

	ip_dyn = interior_point(z, θ,
		idx_ineq = idx_ineq,
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = dyn_opts)

	ip_dyn.opts.diff_sol = false

	ip_jac = interior_point(z, θ,
		idx_ineq = idx_ineq,
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
	dyn_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-4, κ_init = 0.1),
	jac_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-2, κ_init = 0.1))

function f!(d::Dynamics, q0, q1, u1, mode = :dynamics)
	s = d.s
	ip = (mode == :dynamics ? d.ip_dyn : d.ip_jac)
	h = d.h

    ip.z .= copy([q1; 0.1 * ones(nc + nc)])
	ip.θ .= copy([q0; q1; u1; h])

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

struct KukaParticleI{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::Dynamics
end

model = KukaParticleI{Midpoint, FixedTime}(2 * s.model.dim.q, s.model.dim.u, 0, d)

function fd(model::KukaParticleI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]

	q2 = f(model.dynamics, q0, q1, u)

	return [q1; q2]
end

fd(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)

function fdx(model::KukaParticleI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2dx1 = fx1(model.dynamics, q0, q1, u)

	return [zeros(nq, nq) I; dq2dx1]
end

fdx(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)

function fdu(model::KukaParticleI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2du1 = fu1(model.dynamics, q0, q1, u)
	return [zeros(nq, model.m); dq2du1]
end

fdu(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)

T = 21
x1 = [q1; q1]
xT = [qN; qN]
ū = [0.1 * randn(_model.dim.u) for t = 1:T-1]#[t == 1 ? [0.0, 10.0, 0.0, 0.0, 0.0, -0.0, 0.0] : t == 2 ? [0.0, -10.0, 0.0, -0.0, 0.0, 0.0, 0.0] : [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0] for t = 1:T-1] #u_hist #[-1.0 * gravity_compensation(_model, h, q1, q1) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)
visualize!(mvis, _model, q̄, Δt = h)


# visualize!(mvis, _model, [x_track_extend[1:nq]], Δt = h)

# Objective
V = Diagonal(ones(s.model.dim.q))
Q_velocity = 1.0e-3 * [V -V; -V V] ./ h^2.0
Q_track = 1.0 * Diagonal(ones(model.n))
Q_track_extend = copy(Q_track)
Q_track_extend[8, 8] = 10.0
Q_track_extend[nq + 8, nq + 8] = 10.0
x_track = [qD; qD]
x_track_extend = copy(x_track)
x_track_extend[8] += 0.5
x_track_extend[nq + 8] += 0.5
Q = [t < T ? Q_velocity + (t == 11 ? Q_track_extend : 1.0 * Q_track) : Q_velocity + 1.0 * Q_track for t = 1:T]
q = [t < T ? -2.0 * (t == 11 ? Q_track_extend * x_track_extend : 1.0 * Q_track * x_track) : -2.0 * 1.0 * Q_track * x_track for t = 1:T]
R = [Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

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
ul = -100.0 * ones(model.m)
uu = 100.0 * ones(model.m)
p = [t < T ? (t == 11 ? 2 * m + 0 * nq : 2 * m) : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_m = Dict(:qD => x_track_extend[1:nq], :ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 11 ? info_m : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c[1:2 * m] .= [ul - u; u - uu]
        # if t == 11
        #     c[2 * m .+ (1:nq)] = x[1:nq] - cons.con[t].info[:qD]
        # end
	else
		c[1:n] .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

ū = [t == 1 ? [0.0, 6.5, 0.0, 0.0, 0.0, -0.0, 0.0] : t == 2 ? [0.0, -6.5, 0.0, -0.0, 0.0, 0.0, 0.0] : [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0] for t = 1:T-1] #u_hist #[-1.0 * gravity_compensation(_model, h, q1, q1) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
q̄ = state_to_configuration(x̄)
visualize!(mvis, _model, q̄, Δt = h)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.005)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
visualize!(mvis, _model, q̄, Δt = h)
