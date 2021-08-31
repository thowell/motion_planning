using MeshCat, MeshCatMechanisms, RigidBodyDynamics
using FileIO, MeshIO, GeometryBasics, CoordinateTransformations, Rotations, Meshing
using Plots

include(joinpath(pwd(),"models/kuka/kuka_utils.jl"))

urdf_path = joinpath(pwd(), "models/kuka/temp/kuka.urdf")

kuka = MeshCatMechanisms.parse_urdf(urdf_path, remove_fixed_tree_joints = true)
kuka_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(kuka)
state_cache = StateCache(kuka)
result = DynamicsResult(kuka)
result_cache = DynamicsResultCache(kuka)
nq = num_positions(kuka)
ee = findbody(kuka, "iiwa_link_7")
ee_point = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
ee_jacobian_frame = ee_point.frame
ee_jacobian_path = path(kuka, root_body(kuka), ee)
world = root_frame(kuka)

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
# + 3D particle
struct KukaParticle
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
    SMatrix{10, 7}([Diagonal(ones(7)); zeros(3, 7)])
end

function kinematics_ee(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state, kuka_q(q))
    return transform(state, m.ee_point, m.world).v
end

l_string = 0.5

function ϕ_func(m::KukaParticle, q::AbstractVector{T}) where T
	p_ee = kinematics_ee(m, q)
	p_p = q[8:10]
	diff = (p_ee - p_p)
	d_ee_p = sqrt(diff' * diff)
    SVector{1}([l_string - d_ee_p])
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

# dimensions
nq = 7 + 3
nu = 7
nw = 0
nc = 1

# particle mass
mp = 0.001

model = KukaParticle(
	Dimensions(nq, nu, nw, nc),
	mp,
    state_cache1, state_cache2, state_cache3,
    results_cache1, results_cache2, results_cache3,
    world,
	ee,
	ee_point,
	ee_jacobian_frame,
	ee_jacobian_path)

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

	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * (u1 - u_gravity)
        + transpose(P_func(model, q2)) * λ1)
end

function residual(model::KukaParticle, z, θ, κ)
    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c

	q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    λ1 = z[nq .+ (1:nc)]
    s1 = z[nq + nc .+ (1:nc)]

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
     s1 .- ϕ_func(model, q2);
     λ1 .* s1 .- κ;
    ]
end

nz = nq + nc + nc
nθ = nq + nq + nu + 1

function r_func(r, z, θ, κ)
    r .= residual(model, z, θ, κ)
end

function rz_func(rz, z, θ)
    r(a) = residual(model, a, θ, 0.0)
    rz .= ForwardDiff.jacobian(r, z)
end

function rθ_func(rθ, z, θ)
    r(a) = residual(model, z, a, 0.0)
    rθ .= ForwardDiff.jacobian(r, θ)
end

idx_ineq = collect([nq .+ (1:(nc + nc))]...)
z_subset_init = 1.0 * ones(nz - nq)
rz_array = zeros(nz, nz)
rθ_array = zeros(nz, nθ)
