include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))
"""
    planar push block
        particle with contacts at each corner
"""
struct PlanarPush{T}
    dim::Dimensions

    mass_block::T
	mass_pusher::T

    inertia
    μ_surface
	μ_pusher
    gravity

    contact_corner_offset
	block_dim
	block_rnd
end

# signed distance for a box
function sd_box(p, dim)
    q = abs.(p) - dim
	return norm(max.(q, 1.0e-32)) + min(maximum(q), 0.0)
end

function sd_2d_box(p, pose, dim, rnd)
	x, y, θ = pose
	R = rotation_matrix(-θ)
	p_rot = R * (p - pose[1:2])

	return sd_box(p_rot, dim) - rnd
end

# Kinematics
r_dim = 0.1

# contact corner
cc1 = @SVector [r_dim, r_dim]
cc2 = @SVector [-r_dim, r_dim]
cc3 = @SVector [r_dim, -r_dim]
cc4 = @SVector [-r_dim, -r_dim]

contact_corner_offset = @SVector [cc1, cc2, cc3, cc4]

# Parameters
μ_surface = 0.5  # coefficient of friction
μ_pusher = 0.5
gravity = 9.81
mass_block = 1.0   # mass
mass_pusher = 10.0
inertia = 1.0 / 12.0 * mass_block * ((2.0 * r_dim)^2 + (2.0 * r_dim)^2)

rnd = 0.01
dim_rnd = [r_dim - rnd, r_dim - rnd]

# Methods
M_func(model::PlanarPush, q) = Diagonal(@SVector [model.mass_block, model.mass_block,
	model.inertia, model.mass_pusher, model.mass_pusher])

function C_func(model::PlanarPush, q, q̇)
	SVector{5}([0.0, 0.0, 0.0, 0.0, 0.0])
end

function rotation_matrix(x)
	SMatrix{2,2}([cos(x) -sin(x); sin(x) cos(x)])
end

function ϕ_func(model::PlanarPush, q)
    p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sdf = sd_2d_box(p_pusher, p_block, model.block_dim, model.block_rnd)

    @SVector [sdf]
end

function B_func(model::PlanarPush, q)
	SMatrix{5,2}([0.0 0.0;
				  0.0 0.0;
				  0.0 0.0;
				  1.0 0.0;
				  0.0 1.0])
end

function N_func(model::PlanarPush, q)
    tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function p_func(x)
    pos = view(x, 1:2)
    θ = x[3]
    R = rotation_matrix(θ)

    [(pos + R * model.contact_corner_offset[1])[1:2];
     (pos + R * model.contact_corner_offset[2])[1:2];
     (pos + R * model.contact_corner_offset[3])[1:2];
     (pos + R * model.contact_corner_offset[4])[1:2]]
end

function P_func(model::PlanarPush, q)

    P_block = ForwardDiff.jacobian(p_func, q)

	# pusher block
	p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sd_p(x) = sd_2d_box(x, p_block, model.block_dim, model.block_rnd)
	Np = ForwardDiff.gradient(sd_p, p_pusher)

	n_dir = Np[1:2] ./ norm(Np[1:2])
	t_dir = rotation_matrix(0.5 * π) * n_dir

	r = p_pusher - p_block[1:2]
	m = cross([r; 0.0], [t_dir; 0.0])[3]

	P_pusherblock = [t_dir[1]; t_dir[2]; m; -t_dir[1]; -t_dir[2]]'

	return [P_block; P_pusherblock]
end

# Dimensions
nq = 5 # configuration dimension
nu = 2 # control dimension
nc = 5 # number of contact points
nc_impact = 1
nf = 3 # number of faces for friction cone pyramid
nb = (nc - nc_impact) * nf + (nf - 1) * nc_impact

model = PlanarPush(Dimensions(nq, nu, 0, nc),
			mass_block, mass_pusher, inertia,
			[μ_surface for i = 1:nc], μ_pusher,
			gravity,
			contact_corner_offset,
			dim_rnd,
			rnd)

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function dynamics(model::PlanarPush, h, q0, q1, u1, λ1, q2)
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
            + B_func(model, qm2) * u1
            + transpose(N_func(model, q2)) * λ1[end:end]
            + transpose(P_func(model, q2)) * λ1[1:end-1])
end

function residual(model, z, θ, κ)
    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c
    nc_impact = 1
    nc_friction = 3

    nb = 3 * 4 + 2 * 1
    nλ = nb + nc_impact

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc_impact)]
	s1 = z[nq + nc_impact .+ (1:nc_impact)]
	b1 = z[nq + 2 * nc_impact .+ (1:nb)]
	η1 = z[nq + 2 * nc_impact + nb .+ (1:nb)]

	ϕ = ϕ_func(model, q2)

	λ1 = [b1[2:3]; b1[5:6]; b1[8:9]; b1[11:12]; b1[14]; γ1]
    vT = P_func(model, q2) * (q2 - q1) / h[1]

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
	 s1 .- ϕ;
	 γ1 .* s1 .- κ;

	 vT[1:2] - η1[2:3];
	 b1[1] .- model.μ_surface[1] * model.mass_block * model.gravity * h[1] * 0.25;
	 second_order_cone_product(η1[1:3], b1[1:3]) - κ .* [1.0; 0.0; 0.0];

	 vT[3:4] - η1[5:6];
	 b1[4] .- model.μ_surface[2] * model.mass_block * model.gravity * h[1] * 0.25;
	 second_order_cone_product(η1[4:6], b1[4:6]) - κ .* [1.0; 0.0; 0.0];

	 vT[5:6] - η1[8:9];
	 b1[7] .- model.μ_surface[3] * model.mass_block * model.gravity * h[1] * 0.25;
	 second_order_cone_product(η1[7:9], b1[7:9]) - κ .* [1.0; 0.0; 0.0];

	 vT[7:8] - η1[11:12];
	 b1[10] .- model.μ_surface[4] * model.mass_block * model.gravity * h[1] * 0.25;
	 second_order_cone_product(η1[10:12], b1[10:12]) - κ .* [1.0; 0.0; 0.0];

	 vT[9] - η1[14];
	 b1[13] .- model.μ_pusher * γ1;
	 second_order_cone_product(η1[13:14], b1[13:14]) - κ .* [1.0; 0.0];

    ]
end

nz = nq + 2 * nc_impact + 2 * nb
nθ = nq + nq + nu + 1

idx_ineq = collect(nq .+ (1:(2 * nc_impact)))
idx_soc = [collect(nq + 2 * nc_impact .+ (1:3)),
		   collect(nq + 2 * nc_impact + 3 .+ (1:3)),
		   collect(nq + 2 * nc_impact + 3 + 3 .+ (1:3)),
		   collect(nq + 2 * nc_impact + 3 + 3 + 3 .+ (1:3)),
		   collect(nq + 2 * nc_impact + 4 * 3 .+ (1:2)),
		   collect(nq + 2 * nc_impact + nb .+ (1:3)),
   		   collect(nq + 2 * nc_impact + nb + 3 .+ (1:3)),
		   collect(nq + 2 * nc_impact + nb + 3 + 3 .+ (1:3)),
		   collect(nq + 2 * nc_impact + nb + 3 + 3 + 3 .+ (1:3)),
   		   collect(nq + 2 * nc_impact + nb + 4 * 3 .+ (1:2))]

z_subset_init = [0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1, 0.1,
				 1.0, 0.1]

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

rz_array = zeros(nz, nz)
rθ_array = zeros(nz, nθ)

residual(model, ones(nz), ones(nθ), [1.0])
rz_test = zeros(nz, nz)
rz_func(rz_test, ones(nz), ones(nθ))
rθ_func(zeros(nz, nθ), ones(nz), ones(nθ))
