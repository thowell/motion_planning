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

function rotation_matrix(x)
	SMatrix{2,2}([cos(x) -sin(x); sin(x) cos(x)])
end

# signed distance for a box
function sd_box(p, dim)
	q = abs.(p) - dim
	norm(max.(q, 1.0e-32)) + min(maximum(q), 0.0)
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

function P_func(model::PlanarPush, q)
	map1 = [1.0;
	        -1.0]

    map2 = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]

    function p(x)
        pos = view(x, 1:2)
		θ = x[3]
        R = rotation_matrix(θ)

        [map2 * (pos + R * model.contact_corner_offset[1])[1:2];
         map2 * (pos + R * model.contact_corner_offset[2])[1:2];
         map2 * (pos + R * model.contact_corner_offset[3])[1:2];
         map2 * (pos + R * model.contact_corner_offset[4])[1:2]]
    end

    P_block = ForwardDiff.jacobian(p, q)

	# pusher block
	p_block = view(q, 1:3)
	p_pusher = view(q, 4:5)

	sd_p(x) = sd_2d_box(x, p_block, model.block_dim, model.block_rnd)
	Np = ForwardDiff.gradient(sd_p, p_pusher)

	n_dir = Np[1:2] ./ norm(Np[1:2])
	t_dir = rotation_matrix(0.5 * π) * n_dir

	r = p_pusher - p_block[1:2]
	m = cross([r; 0.0], [t_dir; 0.0])[3]

	P_pusherblock = map1 * [t_dir[1]; t_dir[2]; m; -t_dir[1]; -t_dir[2]]'

	return [P_block; P_pusherblock]
end

# function friction_cone(model::PlanarPush,u)
#     λ = u[model.idx_λ]
#     b = u[model.idx_b]
#
#     @SVector [model.μ_surface[1] * λ[1] - sum(b[1:4]),
#               model.μ_surface[2] * λ[2] - sum(b[5:8]),
#               model.μ_surface[3] * λ[3] - sum(b[9:12]),
#               model.μ_surface[4] * λ[4] - sum(b[13:16]),
# 			  model.μ_pusher * λ[5] - sum(b[17:18])]
# end
#
# function maximum_dissipation(model::PlanarPush, x⁺, u, h)
#     q3 = x⁺[model.nq .+ (1:model.nq)]
# 	q2 = x⁺[1:model.nq]
#
#     ψ = u[model.idx_ψ]
#     ψ_stack = [ψ[1] * ones(4);
#                ψ[2] * ones(4);
#                ψ[3] * ones(4);
#                ψ[4] * ones(4);
# 			   ψ[5] * ones(2)]
#
#     η = u[model.idx_η]
#
#     P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
# end

# Dimensions
nq = 5 # configuration dimension
nu = 2 # control dimension
nc = 5 # number of contact points
nc_impact = 1
nf = 4 # number of faces for friction cone pyramid
nb = (nc - 1) * nf + 2

model = PlanarPush(Dimensions(nq, nu, 0, nc),
			mass_block, mass_pusher, inertia,
			[μ_surface for i = 1:nc], μ_pusher,
			gravity,
			contact_corner_offset,
			dim_rnd,
            # [r_dim, r_dim],
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
    nc_friction = 5

    nb = 1 * 2 + 4 * 4
    nλ = nb + nc_impact

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc_impact)]
    b1 = z[nq + nc_impact .+ (1:nb)]
    ψ1 = z[nq + nc_impact + nb .+ (1:nc_friction)]
    η1 = z[nq + nc_impact + nb + nc_friction .+ (1:nb)]
    s1 = z[nq + nc_impact + nb + nc_friction + nb .+ (1:nc_impact)]
    s2 = z[nq + nc_impact + nb + nc_friction + nb + nc_impact .+ (1:nc_friction)]

	ϕ = ϕ_func(model, q2)

	λ1 = [b1; γ1]
    vT = P_func(model, q2) * (q2 - q1) / h[1]
	ψ_stack = [ψ1[1] .* ones(4);
               ψ1[2] .* ones(4);
               ψ1[3] .* ones(4);
               ψ1[4] .* ones(4);
               ψ1[5] .* ones(2)]

   fc = [model.μ_surface[1] * (0.25 * model.mass_block * model.gravity * h[1]) - sum(b1[1:4]);
         model.μ_surface[2] * (0.25 * model.mass_block * model.gravity * h[1]) - sum(b1[5:8]);
         model.μ_surface[3] * (0.25 * model.mass_block * model.gravity * h[1]) - sum(b1[9:12]);
         model.μ_surface[4] * (0.25 * model.mass_block * model.gravity * h[1]) - sum(b1[13:16]);
		 model.μ_pusher * γ1[1] - sum(b1[17:18])]

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
	 s1 .- ϕ;
	 vT + ψ_stack - η1;
	 s2 .- fc
	 γ1 .* s1 .- κ;
	 b1 .* η1 .- κ;
	 ψ1 .* s2 .- κ
    ]
end

nz = nq + nc_impact + nb + nc + nb + nc_impact + nc
nθ = nq + nq + nu + 1

idx_ineq = collect(nq .+ (1:(nc_impact + nb + nc + nb + nc_impact + nc)))
z_subset_init = 0.1 * ones(nc_impact + nb + nc + nb + nc_impact + nc)

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


# residual(model, ones(nz), ones(nθ), [1.0])
