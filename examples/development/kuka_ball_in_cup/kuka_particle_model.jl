# Lagrangian for kuka iiwa via RigidBodyDynamics

using LinearAlgebra
using ForwardDiff
using RigidBodyDynamics
using MeshCatMechanisms

nq = 7 + 3 # number of states - kuka arm joint positions + particle in 3-space
nu = 7 # number of controls - kuka arm joint torques
nγ = 1 # contact points/ number of contact forces
ns = nγ

h = 0.05 # s

m_p = 0.01 # kg - mass of particle

urdf = "/home/taylor/Research/contact_implicit_trajectory_optimization/models/kuka/temp/kuka.urdf"
mechanismkuka = parse_urdf(urdf)
visualskuka = URDFVisuals(urdf)

function get_kuka_ee(kuka)
    ee_body = findbody(kuka, "iiwa_link_ee")
    ee_point = Point3D(default_frame(ee_body),0.,0.,0.)
    return ee_body, ee_point
end

function get_kuka_ee_position_fun(kuka::Mechanism,statecache=StateCache(kuka)) where {O}
    ee_body, ee_point = get_kuka_ee(kuka)
    world = root_frame(kuka)
    nn = num_positions(kuka)

    function ee_position(x::AbstractVector{T}) where T
        state = statecache[T]
        set_configuration!(state, x[1:nn])
        RigidBodyDynamics.transform(state, ee_point, world).v
    end
end

end_effector_function = get_kuka_ee_position_fun(parse_urdf(urdf,remove_fixed_tree_joints=false))

# ee_pos = end_effector_function(zeros(7))
# state = MechanismState(mechanism)

const statecachekuka = StateCache(mechanismkuka)

function 𝓛(q::AbstractVector{T},q̇::AbstractVector{T}) where T
    q_kuka = q[1:7]
    q̇_kuka = q̇[1:7]

    q_p = q[8:10]
    q̇_p = q̇[8:10]

    state = statecachekuka[T]

    set_configuration!(state,q_kuka)
    set_velocity!(state,q̇_kuka)

    kinetic_energy(state)  + 0.5*m_p*q̇_p'*q̇_p - m_p*9.81*q_p[3] - gravitational_potential_energy(state)
end

𝓛d(q1,q2) = let 𝓛=𝓛, h=h
    h*𝓛(0.5*(q1+q2),(q2-q1)/h)
end
𝓛d(z) = 𝓛d(z[1:nq],z[nq .+ (1:nq)])

D𝓛d(z) = ForwardDiff.gradient(𝓛d,z)
D1𝓛d(z) = D𝓛d(z)[1:nq]
D2𝓛d(z) = D𝓛d(z)[nq .+ (1:nq)]
D1𝓛d(q1,q2) = D1𝓛d([q1;q2])
D2𝓛d(q1,q2) = D2𝓛d([q1;q2])

δD1𝓛dδq2(z) = ForwardDiff.jacobian(D1𝓛d,z)[1:nq,nq .+ (1:nq)]
δD1𝓛dδq2(q1,q2) = δD1𝓛dδq2([q1;q2])

d = 0.5

ϕ(q) = let d=d
    q_kuka = q[1:7]
    ee = end_effector_function(q_kuka)
    q_p = q[8:10]
    # println(ee)
    # println(q_p)
    d - norm(ee - q_p)
end
∇ϕ(q) = ForwardDiff.gradient(ϕ,q)


B = zeros(nq,nu)
B[1:nu,1:nu] = Diagonal(ones(nu))
B

F(q,q̇) = let m=mass,g=9.81
    zero(q)
end

Fd(q1,q2) = let F=F, h=h
    h*F(0.5*(q1+q2), (q2-q1)/h)
end

var_int(q1,q2,u,γ,s,q⁺) = let B=B
    D2𝓛d(q1,q2) + D1𝓛d(q2,q⁺) + 0.5*(Fd(q1,q2) + Fd(q2,q⁺)) + B[:,:]*u[:] + ∇ϕ(q⁺)[:,:]*γ[:]
end
var_int(rand(nq),rand(nq),rand(nu),rand(nγ),rand(ns),rand(nq))


function var_int(z)
    q1 = z[1:nq]
    q2 = z[nq .+ (1:nq)]
    u = z[2nq .+ (1:nu)]
    γ = z[(2nq + nu) .+ (1:nγ)]
    s = z[(2nq + nu + nγ) .+ (1:ns)]
    q⁺ = z[(2nq + nu + nγ + ns) .+ (1:nq)]
    var_int(q1,q2,u,γ,s,q⁺)
end
∇var_int(z) = ForwardDiff.jacobian(var_int,z)
∇var_int(q1,q2,u,γ,s,q⁺) = ∇var_int([q1;q2;u;γ;s;q⁺])
∇var_int(rand(nq),rand(nq),rand(nu),rand(nγ),rand(ns),rand(nq))
