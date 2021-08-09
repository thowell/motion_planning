nq = 7
nv = 6
nx = nq + nv

r = 0.5
c1 = @SVector [r, r, r]
c2 = @SVector [r, r, -r]
c3 = @SVector [r, -r, r]
c4 = @SVector [r, -r, -r]
c5 = @SVector [-r, r, r]
c6 = @SVector [-r, r, -r]
c7 = @SVector [-r, -r, r]
c8 = @SVector [-r, -r, -r]

corner_offset = [c1, c2, c3, c4, c5, c6, c7, c8]

function attitude_jacobian(q)
	s = q[1]
	v = q[2:4]

	[-transpose(v);
	 s * I + skew(v)]
end

function G_func(q)
	return [I zeros(eltype(q), 3, 3);
		zeros(eltype(q), 4, 3) attitude_jacobian(q)]
end

function quaternion_rotation_matrix(q)
	r, i, j, k  = q

	r11 = 1.0 - 2.0 * (j^2.0 + k^2.0)
	r12 = 2.0 * (i * j - k * r)
	r13 = 2.0 * (i * k + j * r)

	r21 = 2.0 * (i * j + k * r)
	r22 = 1.0 - 2.0 * (i^2.0 + k^2.0)
	r23 = 2.0 * (j * k - i * r)

	r31 = 2.0 * (i * k - j * r)
	r32 = 2.0 * (j * k + i * r)
	r33 = 1.0 - 2.0 * (i^2.0 + j^2.0)

	SMatrix{3,3}([r11 r12 r13;
	              r21 r22 r23;
				  r31 r32 r33])
end

function kinematics(q)
    p = q[1:3]
    quat = q[4:7]

    R = quaternion_rotation_matrix(quat)

	p1 = p + R * c1
	p2 = p + R * c2
	p3 = p + R * c3
	p4 = p + R * c4
	p5 = p + R * c5
	p6 = p + R * c6
	p7 = p + R * c7
	p8 = p + R * c8

    SVector{24}([p1;
	              p2;
				  p3;
				  p4;
				  p5;
				  p6;
				  p7;
				  p8])
end

ϕ(x1)

function ϕ(q)
	k = kinematics(q)
	idx = collect([3, 6, 9, 12, 15, 18, 21, 24])

	return k[idx]
end

P(q) = ForwardDiff.jacobian(kinematics, q) * G_func(q)
N(q) = ForwardDiff.jacobian(ϕ, q) * G_func(q)

mass = 1.0
inertia = 1.0
gravity = 9.81
mass_matrix = Diagonal(mass * ones(3))
inertia_matrix = Diagonal(inertia * ones(3))
mi_matrix = cat(mass_matrix, inertia_matrix, dims=(1,2))
function bias_vector(q, v)
	ω = v[4:6]
	[0.0; 0.0; gravity * mass; cross(ω, inertia_matrix * ω)]
end

h = 0.01

softplus(x) = log.(1.0 .+ exp.(x))
softminus(x) = x .- softplus.(x)

function L_multiply(q)
	s = q[1]
	v = q[2:4]

	SMatrix{4,4}([s -transpose(v);
	              v s * I + skew(v)])
end

softminus(1.0)
N(ones(nq))
function dynamics(x, u, t)
	xq = x[1:nq]
	pos = xq[1:3]
	quat = xq[4:7]

	xv = x[nq .+ (1:nv)]
	vel = xv[1:3]
	omg = xv[4:6]

	τ = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

	# k_impact = 10.0
	# λ = [0.0; -k_impact * softminus(ϕ(q))]
	k_impact = 2.35
	b_impact = 1.0
	λ = zeros(24)
	impact_idx = collect([3, 6, 9, 12, 15, 18, 21, 24])
	@show λ[impact_idx] = -k_impact * softminus.(ϕ(xq)) + max.(0.0, - b_impact .* N(xq) * xv)
	# λ = [0.0; -k_impact * min(0.0, ϕ(q))]
	# λ = [0.0; -k_impact * softminus(ϕ(q)) + max(0.0, - b_impact * v[2])]

	xv⁺ = xv + h * (mi_matrix \ (τ - bias_vector(xq, xv) + transpose(P(xq)) * λ))
	pos⁺ = pos + h * xv⁺[1:3]
	quat⁺ = 0.5 * h * L_multiply(quat) * [sqrt((0.5 / h)^2.0 - xv⁺[4:6]' * xv⁺[4:6]); xv⁺[4:6]]

	return [pos⁺; quat⁺; xv⁺]
end
sqrt(2.0)
pos1 = [0.0; 0.0; 1.0]
quat1 = [0.5 * sqrt(2.0); 0.5 * sqrt(2.0); 0.0; 0.0]
vel1 = [0.0; 0.0; 0.0]
omg1 = [0.0; 0.0; 0.0]

x1 = [pos1; quat1; vel1; omg1]

x_hist = [x1]
T = 500

for t = 1:T-1
	push!(x_hist, dynamics(x_hist[end], nothing, t))
	# println(ϕ(x_hist[end][1:nq]))
end
visualize!(vis, x_hist, Δt = h)

x_hist[end]
# plot(hcat(x_hist...)[1:2, :]', )

# include(joinpath(pwd(), "models/visualize.jl"))
# vis = Visualizer()
# render(vis)
# visualize!(vis, x_hist, Δt = h)
#
# function visualize!(vis, q;
#         Δt = 0.1)
#
# 	default_background!(vis)
#
#     setobject!(vis["box"], GeometryBasics.Rect(Vec(-1.0 * r,
# 		-1.0 * r,
# 		-1.0 * r),
# 		Vec(2.0 * r, 2.0 * r, 2.0 * r)),
# 		MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
#
#     for i = 1:8
#         setobject!(vis["corner$i"], GeometryBasics.Sphere(Point3f0(0),
#             convert(Float32, 0.05)),
#             MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0.0, 1.0)))
#     end
#
#     anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
#
#     for t = 1:length(q)
#         MeshCat.atframe(anim, t) do
#             settransform!(vis["box"],
# 				compose(Translation(q[t][1:3]...), LinearMap(UnitQuaternion(q[t][4:7]...))))
#
#             for i = 1:8
#                 settransform!(vis["corner$i"],
#                     Translation((q[t][1:3] + UnitQuaternion(q[t][4:7]...) * (corner_offset[i]))...))
#             end
#         end
#     end
#     # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
#     MeshCat.setanimation!(vis, anim)
# end
