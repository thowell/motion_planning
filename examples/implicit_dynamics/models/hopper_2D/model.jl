include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))

"""
    Hopper2D
    	model inspired by "Dynamically Stable Legged Locomotion"
		s = (x, z, t, r)
			x - lateral position
			z - vertical position
			t - body orientation
			r - leg length
"""
mutable struct Hopper2D{T}
    dim::Dimensions

    mb::T # mass of body
    ml::T # mass of leg
    Jb::T # inertia of body
    Jl::T # inertia of leg

    μ_world::T  # coefficient of friction
    μ_joint::T  # gravity
	g::T

	joint_friction::SVector
end

lagrangian(model::Hopper2D, q, q̇) = 0.0

# Kinematics
function kinematics(::Hopper2D, q)
	[q[1] + q[4] * sin(q[3]),
	 q[2] - q[4] * cos(q[3])]
end

# Methods
function M_func(model::Hopper2D, q)
	Diagonal([model.mb + model.ml,
    					   model.mb + model.ml,
    					   model.Jb + model.Jl,
    					   model.ml])
 end

function C_func(model::Hopper2D, q, q̇)
	[0.0,
	  (model.mb + model.ml) * model.g,
	  0.0,
	  0.0]
end

function ϕ_func(model::Hopper2D, q)
    [q[2] - q[4] * cos(q[3])]
end

function J_func(::Hopper2D, q)
    [1.0 0.0 (q[4] * cos(q[3])) sin(q[3]);
		          0.0 1.0 (q[4] * sin(q[3])) (-1.0 * cos(q[3]))]
end

function B_func(::Hopper2D, q)
	[0.0 0.0 1.0 0.0;
                -sin(q[3]) cos(q[3]) 0.0 1.0]
end

# Working Parameters
gravity = 9.81 # gravity
μ_world = 0.8 # coefficient of friction
μ_joint = 0.0

# TODO: change to Raibert parameters
mb = 3.0 # body mass
ml = 0.3  # leg mass
Jb = 0.75 # body inertia
Jl = 0.075 # leg inertia

# Dimensions
nq = 4
nu = 2
nw = 2
nc = 1
nb = 2

model = Hopper2D(Dimensions(nq, nu, nw, nc),
			   mb, ml, Jb, Jl,
			   μ_world, μ_joint, gravity,
			   SVector{4}(zeros(4)))

function lagrangian_derivatives(model::Hopper2D, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

lagrangian_derivatives(model, rand(nq), rand(nq))

function dynamics(model::Hopper2D, h, q0, q1, u1, λ1, q2)
    # evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
    	+ transpose(B_func(model, qm2)) * u1
        + transpose(J_func(model, q2)) * λ1)
        # -h[1] * model.joint_friction .* vm2)
end

function residual(model, z, θ, κ)
    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c
    nb = nc * 2

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    μ = model.μ_world

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc)]
    b1 = z[nq + nc .+ (1:nb)]
    ψ1 = z[nq + nc + nb .+ (1:nc)]
    η1 = z[nq + nc + nb + nc .+ (1:nb)]
    s1 = z[nq + nc + nb + nc + nb .+ (1:nc)]
    s2 = z[nq + nc + nb + nc + nb + nc .+ (1:nc)]

	ϕ = ϕ_func(model, q2)

	λ1 = [b1[1] - b1[2]; γ1]
    vT = (J_func(model, q2) * (q2 - q1) / h[1])[1]
	vT_stack = [vT; -vT]
	ψ_stack = ψ1 .* ones(nb)

	[
     dynamics(model, h, q0, q1, u1, λ1, q2);
	 s1 - ϕ;
	 vT_stack + ψ_stack - η1;
	 s2 .- (μ[1] * γ1 .- sum(b1));
	 γ1 .* s1 .- κ;
	 b1 .* η1 .- κ;
	 ψ1 .* s2 .- κ
    ]
end

nz = nq + nc + nb + nc + nb + nc + nc
nθ = nq + nq + nu + 1

idx_ineq = collect(nq .+ (1:(nc + nb + nc + nb + nc + nc)))
z_subset_init = 0.1 * ones(nc + nb + nc + nb + nc + nc)

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r = residual(model, z, θ, κ)
r = Symbolics.simplify.(r)
rz = Symbolics.jacobian(r, z, simplify = true)
rθ = Symbolics.jacobian(r, θ, simplify = true) # TODO: sparse version

# Build function
r_func = eval(build_function(r, z, θ, κ)[2])
rz_func = eval(build_function(rz, z, θ)[2])
rθ_func = eval(build_function(rθ, z, θ)[2])

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

@save joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array
