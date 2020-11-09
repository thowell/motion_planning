"""
	Rocket
"""

abstract type Rocket end

struct RocketNominal{T} <: Rocket
	n::Int
	m::Int
	d::Int

	g::T  # gravity
	m1::T # mass
	l1::T # length
	J::T  # inertia
end

struct RocketSlosh{T} <: Rocket
	n::Int
	m::Int
	d::Int

	g::T  # gravity
	m1::T # mass
	l1::T # length
	J::T  # inertia

	m2::T # mass of fuel
	l2::T # length of fuel pendulum
	l3::T # length from rocket COM to fuel pendulum
end

g = 9.81 # gravity
m1 = 1.0 # mass of rocket
l1 = 0.5 # length from COM to thruster
J = 1.0 / 12.0 * m1 * (2.0 * l1)^2 # inertia of rocket

m2 = 0.1 # mass of pendulum
l2 = 0.1 # length from COM to pendulum
l3 = 0.1 # length of pendulum

# models
m = 2
d = 0

# nominal model
n_nominal = 6
model_nominal = RocketNominal(n_nominal, m, d, g, m1, l1, J)

# slosh model
n_slosh = 8
model_slosh = RocketSlosh(n_slosh, m, d, g, m1 - m2, l1, J, m2, l2, l3)

function k_thruster(model::Rocket, x)
	y, z, θ = x[1:3]
	px = y + model.l1 * sin(θ)
	pz = z - model.l1 * cos(θ)

	return [py;
		    pz]
end

function jacobian_thruster(model::RocketNominal, x)
	y, z, θ = x[1:3]

	return [1.0 0.0 model.l1 * cos(θ);
			0.0 1.0 model.l1 * sin(θ)]
end

function jacobian_thruster(model::RocketSlosh, x)
	y, z, θ = x[1:3]

	return [1.0 0.0 model.l1 * cos(θ) 0.0;
			0.0 1.0 model.l1 * sin(θ) 0.0]
end

function k_mass(model::RocketSlosh, x)
	yp = x[1] + l2 * sin(x[3]) + l3 * sin(x[4])
	zp = x[2] - l2 * cos(x[3]) - l3 * cos(x[4])

	return [yp;
			zp]
end

function jacobian_mass(model::RocketSlosh, x)
	return [1.0 0.0 l2 * cos(x[3]) l3 * cos(x[4]);
	        0.0 1.0 l2 * sin(x[3]) l3 * sin(x[4])]
end

function lagrangian(model::RocketNominal, q, q̇)
	return (0.5 * model.m1 * (q̇[1]^2 + q̇[2]^2.0) + 0.5 * model.J * q̇[3]^2.0
			- model.m1 * model.g * q[2])
end

function lagrangian(model::RocketSlosh, q, q̇)
	zp = k_mass(model, q)[2]
	vp = jacobian_mass(model, q) * q̇

	return (0.5 * model.m1 * (q̇[1]^2 + q̇[2]^2) + 0.5 * model.J * q̇[3]^2.0
			- model.m1 * model.g * q[2]
			+ 0.5 * model.m2 * vp' * vp
			- model.m2 * model.g * zp)
end

function dLdq(model::Rocket, q, q̇)
	Lq(x) = lagrangian(model, x, q̇)
	ForwardDiff.gradient(Lq, q)
end

function dLdq̇(model::Rocket, q, q̇)
	Lq̇(x) = lagrangian(model, q, x)
	ForwardDiff.gradient(Lq̇, q̇)
end

function f(model::Rocket, x, u, w)
	nq = convert(Int, floor(model.n / 2.0))
	q = x[1:nq]
	q̇ = x[nq .+ (1:nq)]
	tmp_q(z) = dLdq̇(model, z, q̇)
	tmp_q̇(z) = dLdq̇(model, q, z)
	[q̇;
	 ForwardDiff.jacobian(tmp_q̇,q̇) \ (-1.0 * ForwardDiff.jacobian(tmp_q, q) * q̇
	 	+ dLdq(model, q, q̇)
		+ jacobian_thruster(model, q)' * u[1:2])]
end

function state_output(model::RocketSlosh, x)
	x[collect([1, 2, 3, 5, 6, 7])]
end

function state_output_idx(mode::RocketSlosh, idx)
	idx[collect([1, 2, 3, 5, 6, 7])]
end

function visualize!(vis, model::Rocket, x;
       Δt = 0.1, r_rocket = 0.1, r_pad = 0.25)

	obj_rocket = joinpath(pwd(), "src/models/rocket/space_x_booster.obj")
	mtl_rocket = joinpath(pwd(), "src/models/rocket/space_x_booster.mtl")

	rkt_offset = [4.0, -6.35, 0.2]
	ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale = 1.0)
	setobject!(vis["rocket"], ctm)
	settransform!(vis["rocket"], compose(Translation((x[T][1:3] + rkt_offset)...),
		LinearMap(RotZ(-pi) * RotX(pi / 2.0))))

	obj_platform = joinpath(pwd(), "src/models/rocket/space_x_platform.obj")
	mtl_platform = joinpath(pwd(), "src/models/rocket/space_x_platform.mtl")

	ctm_platform = ModifiedMeshFileObject(obj_platform, mtl_platform, scale = 1.0)
	setobject!(vis["platform"], ctm_platform)
	settransform!(vis["platform"], compose(Translation(0.0, 0.0, 0.0),
		LinearMap(RotZ(pi) * RotX(pi / 2.0))))

   	anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(x)
        MeshCat.atframe(anim,t) do
			settransform!(vis["rocket"],
				compose(Translation(x[t][1] + rkt_offset[1],
					0.0 + rkt_offset[2],
					x[t][2] + rkt_offset[3]),
				LinearMap(RotY(-1.0 * x[t][3]) * RotZ(pi) * RotX(pi / 2.0))))
        end
    end

    # settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
	# 	LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
