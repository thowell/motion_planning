using MeshCat, MeshCatMechanisms, RigidBodyDynamics
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, Rotations, Meshing
using Plots

include(joinpath(pwd(),"src/models/kuka/kuka_utils.jl"))

# Mini golf surface
t = range(-1.0, stop = 3.0, length = 100)
_k = 10.0
shift = 1.5
softplus(x) = log(1 + exp(_k * (x - shift))) / _k
dsoftplus(x) = 1.0 / (1.0 + exp(-1.0 * _k * (x - shift)))
plot(t, softplus.(t))

# ellipsoid
ellipsoid(x, y, rx, ry) = (x^2.0) / (rx^2) + (y^2) / (ry^2) - 1.0
∇ellipsoid(x,y,rx,ry) = [2.0 * x / (rx^2.0); 2.0 * y / (ry^2.0); 0.0]
get_y(x,rx,ry) = sqrt((1.0 - ((x)^2.0) / ((rx)^2.0)) * (ry^2))

ellipsoid(x,y,xc,yc,rx,ry) = ((x - xc)^2.0) / (rx^2.0) + ((y - yc)^2.0) / (ry^2.0) - 1.0
∇ellipsoid(x,y,xc,yc,rx,ry) = [2.0 * (x - xc) / (rx^2.0); 2.0*(y - yc) / (ry^2.0); 0.0]
get_y(x,xc,yc,rx,ry) = sqrt((1.0 - ((x - xc)^2.0) / ((rx)^2.0)) * (ry^2.0)) + yc

rx1 = 1.5
ry1 = 1.5

rx2 = 2.5
ry2 = 2.5

urdf_path = joinpath(pwd(), "src/models/kuka/temp/kuka.urdf")

kuka = MeshCatMechanisms.parse_urdf(urdf_path, remove_fixed_tree_joints = true)
kuka_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(kuka)
state_cache = StateCache(kuka)
result = DynamicsResult(kuka)
result_cache = DynamicsResultCache(kuka)

vis = Visualizer()
mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])
open(vis)

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
ee_point = Point3D(default_frame(ee), 0.374, 0.0, 0.0)
setelement!(mvis, ee_point, 0.01)
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
struct KukaParticle{T} <: Model
	n::Int
	m::Int
	d::Int

	# particle
	mp::T
	rp::T

    qL::Vector{T}
    qU::Vector{T}

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

	μ_ee_p
	μ1
	μ2
	μ3

	nq
    nu
    nc
    nf
    nb

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s

end

# Dimensions
nq = 7 + 3 # configuration dim
nu = 7 # control dim
nc = 3 # number of contact points
nf = 4 # number of faces for friction cone
nb = nc * nf
ns = 1

n = 2 * nq
m_contact = nc + nb + nc + nb + 1
m = nu + m_contact
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

mp = 0.05
rp = 0.05
μ_ee_p = 1.0
μ1 = 0.01
μ2 = 0.1
μ3 = 0.1

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
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state, kuka_q(q))
    mass_matrix!(result.massmatrix, state)
    return cat(result.massmatrix, Diagonal(m.mp * ones(3)), dims = (1, 2))
end

function C_func(m::KukaParticle, qk, qn, h)
    if eltype(kuka_q(qk)) <: ForwardDiff.Dual
		return [dynamics_bias_qk(m, kuka_q(qk), kuka_q(qn), h)..., 0.0, 0.0, m.mp * 9.81]
	elseif eltype(kuka_q(qn)) <: ForwardDiff.Dual
		return [dynamics_bias_qn(m,kuka_q(qk), kuka_q(qn), h)..., 0.0 ,0.0, m.mp * 9.81]
	elseif typeof(h) <: ForwardDiff.Dual
		return [dynamics_bias_h(m, kuka_q(qk), kuka_q(qn), h)..., 0.0, 0.0, m.mp * 9.81]
	else
		return [_dynamics_bias(m, kuka_q(qk), kuka_q(qn), h)..., 0.0, 0.0, m.mp * 9.81]
	end
end

function _dynamics_bias(m::KukaParticle, qk, qn, h)
	T = eltype(m.qL)
	state = m.state_cache1[T]
	result = m.result_cache1[T]
	set_configuration!(state,qk)
	set_velocity!(state, (qn - qk) / h)
	dynamics_bias!(result, state)
    return result.dynamicsbias
end

function dynamics_bias_qk(m::KukaParticle, qk::AbstractVector{T}, qn, h) where T
    state = m.state_cache2[T]
    result = m.result_cache2[T]
    set_configuration!(state, qk)
    set_velocity!(state,(qn - qk) / h)
    dynamics_bias!(result, state)
    return result.dynamicsbias
end

function dynamics_bias_qn(m::KukaParticle, qk, qn::AbstractVector{T}, h) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state, qk)
    set_velocity!(state, (qn - qk) / h)
    dynamics_bias!(result, state)
    return result.dynamicsbias
end

function dynamics_bias_h(m::KukaParticle,qk,qn,h::T) where T
    state = m.state_cache1[T]
    result = m.result_cache1[T]
    set_configuration!(state, qk)
    set_velocity!(state,(qn - qk) / h)
    dynamics_bias!(result, state)
    return result.dynamicsbias
end

function B_func(m::KukaParticle, q)
    [Diagonal(ones(7)); zeros(3, 7)]
end

function friction_cone(model::KukaParticle, u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]

    @SVector [model.μ_ee_p * λ[1] - sum(b[1:4]),#,
			  model.μ1 * λ[2] - sum(b[5:8]),
			  model.μ2 * λ[3] - sum(b[9:12])]
			  # model.μ3*λ[3] - sum(b[9:12])]
end

function maximum_dissipation(model::KukaParticle, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

	ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1] * ones(4);
			   ψ[2] * ones(4);
			   ψ[3] * ones(4)]

    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function fd(model::KukaParticle, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

    [q2⁺ - q2⁻;
    ((1.0 / h) * (M_func(model, q1) * (SVector{10}(q2⁺) - SVector{10}(q1))
    - M_func(model, q2⁺) * (SVector{10}(q3) - SVector{10}(q2⁺)))
    + B_func(model, q3) * SVector{7}(u_ctrl)
    + transpose(N_func(model, q3)) * SVector{3}(λ)
    + transpose(P_func(model, q3)) * SVector{12}(b)
    - h * C_func(model, q2⁺, q3, h))]
end

function kinematics_ee(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state, kuka_q(q))
    return transform(state, m.ee_point, m.world).v
end

function ϕ_func(m::KukaParticle, q::AbstractVector{T}) where T
    # state = m.state_cache3[T]
    # set_configuration!(state,kuka_q(q))
	# p_ee = transform(state, m.ee_point, m.world).v

	p_ee = kinematics_ee(m, q)
	p_p = q[8:10]
	diff = (p_ee - p_p)
	d_ee_p = diff' * diff
    @SVector [d_ee_p - m.rp^2,#,
		      ellipsoid(p_p[1], p_p[2], 0.25, -1.5, 1.0, 1.0),
	          # # -1.0*ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2),
	          p_p[3] - softplus(p_p[1])]
end

function N_func(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]

    # N = zeros(T,m.nc,m.nq)

    set_configuration!(state, kuka_q(q))

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T, 3, 7))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path,ee_in_world) #TODO confirm this is correct

    # N[1,:] = pj1.J[3,:]
    # N[2,:] = pj2.J[3,:]

	p_ee = ee_in_world.v
	p_p = q[8:10]
	diff = (p_ee - p_p)
	# d_ee_p = diff'*diff

    return [2.0 * diff' * [pj1.J[1:3,:] -1.0 * Diagonal(ones(3))];#;
			zeros(1, 7) transpose(∇ellipsoid(p_p[1], p_p[2], 0.25, -1.5, 1.0, 1.0));
			# # zeros(1,7) -1.0*transpose(∇ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2));
			zeros(1, 7) -dsoftplus(p_p[1]) 0.0 1.0]
	# ϕ_tmp(y) = ϕ_func(model,y)
	# ForwardDiff.jacobian(ϕ_tmp,q)
end

# tmp(x) = ϕ_func(model,x)
# vec(ForwardDiff.jacobian(tmp,x1)) - vec(N_func(model,x1))

function P_func(m::KukaParticle, q::AbstractVector{T}) where T
    state = m.state_cache3[T]

	map = @SMatrix [1.0 0.0;
				    -1.0 0.0;
				    0.0 1.0;
				    0.0 -1.0]

    set_configuration!(state, kuka_q(q))

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T, 3, 7))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path, ee_in_world) #TODO confirm this is correct

	y0 = [0.0; 1.0; 0.0]
	z0 = [0.0; 0.0; 1.0]
	p_p = particle_q(q)
    return [map * pj1.J[2:3,:] zeros(4,3);
			zeros(4,7) map * [transpose(z0); transpose(cross(z0, ∇ellipsoid(p_p[1], p_p[2], 0.25, -1.5, 1.0, 1.0)))];
			# # zeros(4,7) map*[transpose(z0); transpose(cross(z0,∇ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2)))];
			zeros(4,7) map * [transpose(y0); transpose(cross(y0, [-1.0 * dsoftplus(p_p[1]); 0.0; 1.0]))]]

end

qL = -Inf * ones(nq)
qU = Inf * ones(nq)

model = KukaParticle(
	n, m, d,
	mp, rp,
	qL, qU,
    state_cache1, state_cache2, state_cache3,
    results_cache1, results_cache2, results_cache3,
    world,
	ee,
	ee_point,
	ee_jacobian_frame,
	ee_jacobian_path,
	μ_ee_p,
	μ1, μ2, μ3,
	nq, nu, nc, nf, nb,
	idx_u,
	idx_λ,
	idx_b,
	idx_ψ,
	idx_η,
	idx_s)

# Visualization
function visualize!(mvis, model::KukaParticle, q;
		verbose = false, r_ball = 0.1, Δt = 0.1)

	p_hole = [0.66; -3.0; 0.0]
	f = x -> x[3] - softplus(x[1])

	sdf = SignedDistanceField(f, Rect(Vec(-5, -10, -1), Vec(10, 10, 4)))
	mesh = HomogenousMesh(sdf, MarchingTetrahedra())
	setobject!(vis["slope"], mesh,
			   MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
	settransform!(vis["slope"], compose(Translation(0.0, 5.0, 0.0)))

	circle1 = Cylinder(Point3f0(0,0,0), Point3f0(0,0,0.25), convert(Float32,1.0 - r_ball))
		setobject!(vis["circle1"], circle1,
		MeshPhongMaterial(color = RGBA(86/255, 125/255, 20/255, 1.0)))
	settransform!(vis["circle1"], compose(Translation(0.25, -1.5, 0.0)))


	setobject!(vis["ball"], Sphere(Point3f0(0),
				convert(Float32, r_ball)),
				MeshPhongMaterial(color = RGBA(1,1,1,1.0)))

	settransform!(vis["ball"], compose(Translation(0.66,3.0,0.0)))

	hole = Cylinder(Point3f0(0,0,0), Point3f0(0,0,0.01),
		convert(Float32, r_ball * 1.5))
		setobject!(vis["hole"], hole,
		MeshPhongMaterial(color = RGBA(0,0,0,1.0)))
	settransform!(vis["hole"], compose(Translation(p_hole...)))

	club = Cylinder(Point3f0(0,0,0), Point3f0(0,0,0.374), convert(Float32,0.025))
   	setobject!(vis["club"], club, MeshPhongMaterial(color = RGBA(0,0,0,1.0)))

	state = model.state_cache1[Float64]

	ee = findbody(kuka, "iiwa_link_7")
	ee_body = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
	ee_end = Point3D(default_frame(ee), 0.374, 0.0, 0.0)
	ee_body_jacobian_frame = ee_body.frame
	ee_body_jacobian_path = path(kuka, root_body(kuka), ee)
	ee_end_jacobian_frame = ee_end.frame
	ee_end_jacobian_path = path(kuka, root_body(kuka), ee)

	world = root_frame(kuka)
	ee_body_in_world = transform(state, ee_body, world).v
	ee_end_in_world = transform(state, ee_end, world).v

	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T
        q_kuka = kuka_q(q[t])
		q_particle = particle_q(q[t])
		set_configuration!(state, kuka_q(q[t]))
		ee_body_in_world = transform(state, ee_body, world).v
		ee_end_in_world = transform(state, ee_end, world).v

        MeshCat.atframe(anim,t) do
			set_configuration!(mvis,q_kuka)
            settransform!(vis["ball"], compose(Translation(q_particle + [0.0;0.0;0.5 * r_ball]), LinearMap(RotZ(0))))
			settransform!(vis["club"], cable_transform(ee_body_in_world, ee_end_in_world))

			if norm(particle_q(q[t])[1:2] - p_hole[1:2]) < 1.0e-2
				setvisible!(vis["ball"], false)
			else
				setvisible!(vis["ball"], true)
			end
		end
    end

    MeshCat.setanimation!(vis, anim)
end
