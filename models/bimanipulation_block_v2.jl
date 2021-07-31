"""
    bimanual block
        particle with contacts at each corner
"""
struct BimanipulationBlockV2{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int

    mass_block
	mass_pusher

    J
    μ_surface
	μ_pusher
    g

    contact_corner_offset
	block_dim
	block_rnd

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

# signed distance for a box
function sd_box(p, dim)
	q = abs.(p) - dim
	norm(max.(q, 1.0e-32)) + min(maximum(q), 0.0)
end

function sd_2d_box(p, pose, dim, rnd)
	x, y, θ = pose
	R = rotation_matrix(-θ)
	p_rot = R * (p - pose[1:2])

	return sd_box(p_rot, dim .- rnd) - rnd
end

# Dimensions
nq = 7 # configuration dimension
nu = 4 # control dimension
nc = 8 # number of contact points
nf = 2 # number of faces for friction cone pyramid
nb = nc * nf
ns = 1

n = 2 * nq
m = nu + nc + nb + nc + nb + 1
d = 0

idx_u = (1:nu)
idx_λ = nu .+ (1:nc)
idx_b = nu + nc .+ (1:nb)
idx_ψ = nu + nc + nb .+ (1:nc)
idx_η = nu + nc + nb + nc .+ (1:nb)
idx_s = nu + nc + nb + nc + nb .+ (1:ns)

# Kinematics
r = 0.1

# contact corner
cc1 = @SVector [r, r]
cc2 = @SVector [-r, r]
cc3 = @SVector [r, -r]
cc4 = @SVector [-r, -r]

# control location
# cu1 = @SVector [r, r]
# cu2 = @SVector [-r, r]
# cu3 = @SVector [-r, -r]
# cu4 = @SVector [r, -r]
# cu5 = @SVector [r, 0.0]
# cu6 = @SVector [0.0, r]
# cu7 = @SVector [-r, 0.0]
# cu8 = @SVector [0.0, -r]

contact_corner_offset = @SVector [cc1, cc2, cc3, cc4]
# control_input_offset = @SVector [cu1, cu2, cu3, cu4, cu5, cu6, cu7, cu8]
# control_input_offset = @SVector [cu5, cu6, cu7, cu8]

# Parameters
μ_surface = 1.0  # coefficient of friction
μ_pusher = 1.0
g = 9.81
mass_block = 1.0   # mass
mass_pusher = 0.1
J = 1.0 / 12.0 * mass_block * ((2.0 * r)^2 + (2.0 * r)^2)

rnd = 0.01
dim = [r, r]

# Methods
M_func(model::BimanipulationBlockV2, q) = Diagonal(@SVector [model.mass_block, model.mass_block,
	model.J, model.mass_pusher, model.mass_pusher, model.mass_pusher, model.mass_pusher])

function C_func(model::BimanipulationBlockV2, q, q̇)
	SVector{7}([0.0, model.mass_block * model.g, 0.0, 0.0, 0.0, 0.0, 0.0])
end

function rotation_matrix(x)
	SMatrix{2,2}([cos(x) -sin(x); sin(x) cos(x)])
end

function ϕ_func(model::BimanipulationBlockV2, q)
    p_block = view(q, 1:3)
	p_pusher1 = view(q, 4:5)
	p_pusher2 = view(q, 6:7)

	θ = q[3]
    R = rotation_matrix(θ)

	@SVector [(p_block[1:2] + R * model.contact_corner_offset[1])[2],
              (p_block[1:2] + R * model.contact_corner_offset[2])[2],
              (p_block[1:2] + R * model.contact_corner_offset[3])[2],
              (p_block[1:2] + R * model.contact_corner_offset[4])[2],
			  q[5],
			  q[7],
			  sd_2d_box(p_pusher1, p_block, model.block_dim, model.block_rnd),
			  sd_2d_box(p_pusher1, p_block, model.block_dim, model.block_rnd)]
end

function B_func(model::BimanipulationBlockV2, q)
	SMatrix{7,4}([0.0 0.0 0.0 0.0;
				  0.0 0.0 0.0 0.0;
				  0.0 0.0 0.0 0.0;
				  1.0 0.0 0.0 0.0;
				  0.0 1.0 0.0 0.0;
				  0.0 0.0 1.0 0.0;
				  0.0 0.0 0.0 1.0;])
end

function N_func(model::BimanipulationBlockV2, q)
    tmp(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(tmp, q)
end

function P_func(model::BimanipulationBlockV2, q)
	map1 = [1.0;
	        -1.0]

    # map2 = [1. 0.;
    #        0. 1.;
    #        -1. 0.;
    #        0. -1.]



    function p(x)
        pos = view(x, 1:2)
		θ = x[3]
        R = rotation_matrix(θ)

        [map1 * (pos + R * model.contact_corner_offset[1])[1];
         map1 * (pos + R * model.contact_corner_offset[2])[1];
         map1 * (pos + R * model.contact_corner_offset[3])[1];
         map1 * (pos + R * model.contact_corner_offset[4])[1]]
    end

    P_block = ForwardDiff.jacobian(p, q)

	# pusher block
	p_block = view(q, 1:3)
	p_pusher1 = view(q, 4:5)
	p_pusher2 = view(q, 6:7)

	sd_p(x) = sd_2d_box(x, p_block, model.block_dim, model.block_rnd)

	Np1 = ForwardDiff.gradient(sd_p, p_pusher1)

	n_dir1 = Np1[1:2] ./ norm(Np1[1:2])
	t_dir1 = rotation_matrix(0.5 * π) * n_dir1

	r1 = p_pusher1 - p_block[1:2]
	m1 = cross([r1; 0.0], [t_dir1; 0.0])[3]

	P_pusherblock1 = map1 * [t_dir1[1]; t_dir1[2]; m1; -t_dir1[1]; -t_dir1[2]; 0.0; 0.0]'

	Np2 = ForwardDiff.gradient(sd_p, p_pusher2)

	n_dir2 = Np2[1:2] ./ norm(Np2[1:2])
	t_dir2 = rotation_matrix(0.5 * π) * n_dir2

	r2 = p_pusher2 - p_block[1:2]
	m2 = cross([r2; 0.0], [t_dir2; 0.0])[3]

	P_pusherblock2 = map1 * [t_dir2[1]; t_dir2[2]; m2; 0.0; 0.0; -t_dir2[1]; -t_dir2[2]]'


	return [P_block; map1 * [0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0]'; map1 * [0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0]'; P_pusherblock1; P_pusherblock2]
end

function friction_cone(model::BimanipulationBlockV2,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ_surface[1] * λ[1] - sum(b[1:2]),
              model.μ_surface[2] * λ[2] - sum(b[3:4]),
              model.μ_surface[3] * λ[3] - sum(b[5:6]),
              model.μ_surface[4] * λ[4] - sum(b[7:8]),
			  model.μ_surface[5] * λ[5] - sum(b[9:10]),
			  model.μ_surface[6] * λ[6] - sum(b[11:12]),
			  model.μ_pusher * λ[7] - sum(b[13:14]),
			  model.μ_pusher * λ[8] - sum(b[15:16])]
end

function maximum_dissipation(model::BimanipulationBlockV2, x⁺, u, h)
    q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]

    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1] * ones(2);
               ψ[2] * ones(2);
               ψ[3] * ones(2);
               ψ[4] * ones(2);
			   ψ[5] * ones(2);
			   ψ[6] * ones(2);
			   ψ[7] * ones(2);
			   ψ[8] * ones(2)]

    η = u[model.idx_η]

    P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function lagrangian_derivatives(model, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end


function fd(model::BimanipulationBlockV2{Discrete, FixedTime}, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)

	qm1 = 0.5 * (q1 + q2⁺)
    vm1 = (q2⁺ - q1) / h
    qm2 = 0.5 * (q2⁺ + q3)
    vm2 = (q3 - q2⁺) / h

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	[q2⁺ - q2⁻;
     (0.5 * h * D1L1 + D2L1 + 0.5 * h * D1L2 - D2L2
     + B_func(model, qm2) * SVector{4}(u_ctrl[1:4])
     + transpose(N_func(model, q3)) * SVector{8}(λ)
     + transpose(P_func(model, q3)) * SVector{16}(b))]
end

model = BimanipulationBlockV2{Discrete, FixedTime}(n, m, d,
			mass_block, mass_pusher, J,
			[μ_surface for i = 1:nc], μ_pusher,
			g,
			contact_corner_offset,
			dim,
			rnd,
            nq, nu, nc, nf, nb,
            idx_u,
            idx_λ,
            idx_b,
            idx_ψ,
            idx_η,
            idx_s)


function visualize!(vis, model::BimanipulationBlockV2, q, u; r = r,
        Δt = 0.1)

	default_background!(vis)

    setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r,
		-1.0 * r,
		-1.0 * r),
		Vec(2.0 * r, 2.0 * r, 2.0 * r)),
		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    for i = 1:4
        setobject!(vis["contact$i"], GeometryBasics.Sphere(Point3f0(0),
            convert(Float32, 0.02)),
            MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
    end

	force_vis1 = ArrowVisualizer(vis[:force1])
	setobject!(force_vis1, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

	us = u[1] / 2.0

	settransform!(force_vis1,
				Point(q[1][4] - us[1], 0, q[1][5] - us[1]),
				Vec(us[1], 0, us[2]),
				shaft_radius=0.01,
				max_head_radius=0.025)

	force_vis2 = ArrowVisualizer(vis[:force2])
	setobject!(force_vis2, MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

	settransform!(force_vis2,
				Point(q[1][6] - us[3], 0, q[1][7] - us[4]),
				Vec(us[3], 0, us[4]),
				shaft_radius=0.01,
				max_head_radius=0.025)


    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
			if t < T-1

				# pusher 1
				if norm(u[t][1:2]) < 1.0e-6
					setvisible!(vis[:force1], false)
				else
					setvisible!(vis[:force1], true)

					us = u[t] / 10.0
					settransform!(force_vis1,
								Point(q[t+1][4] - us[1], 0, q[t+1][5] - us[2]),
								Vec(us[1], 0, us[2]),
								shaft_radius=0.01,
								max_head_radius=0.025)
				end

				# pusher 2
				if norm(u[t][3:4]) < 1.0e-6
					setvisible!(vis[:force2], false)
				else
					setvisible!(vis[:force2], true)

					us = u[t] / 2.0
					settransform!(force_vis2,
								Point(q[t+1][6] - us[3], 0.0, q[t+1][7] - us[4]),
								Vec(us[3], 0.0, us[4]),
								shaft_radius=0.01,
								max_head_radius=0.025)
				end
			end

            settransform!(vis["box"],
				compose(Translation(q[t+1][1], 0.0, q[t+1][2]), LinearMap(RotY(q[t+1][3]))))

            for i = 1:4
                settransform!(vis["contact$i"],
                    Translation(([q[t+1][1]; 0.0; q[t+1][2]] + RotY(q[t+1][3]) * [contact_corner_offset[i][1]; 0.0; contact_corner_offset[i][2];])...))
            end
        end
    end

	settransform!(vis["/Cameras/default"],
		compose(Translation(0.0, -90.0, -1.0),LinearMap(RotZ(pi / 2.0))))
	setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 75)

    MeshCat.setanimation!(vis, anim)
end
