include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))
# include_implicit_dynamics()

# """
#     floating-base quadruped model
#         rigid body with force inputs via pre-specified contact sequence
# """
# struct QuadrupedFloatingBase{T}
#     dim::Dimensions
#
#     mass::T
#     inertia
#
#     gravity
# end
#
# nq = 6
# nu = 4 * 3
# nc = 0
# nw = 0
#
# model = QuadrupedFloatingBase(Dimensions(nq, nu, nc, nw),
#     1.0, Diagonal(ones(3)), [0.0; 0.0; 9.81])
#
# # eq. 14 http://roboticexplorationlab.org/papers/planning_with_attitude.pdf
# function attitude_jacobian(q)
# 	s = q[1]
# 	v = q[2:4]
#
# 	[-transpose(v);
# 	 s * I + skew(v)]
# end
#
# function G_func(q)
# 	quat = q[4:7]
# 	[1.0 0.0 0.0 0.0 0.0 0.0;
#      0.0 1.0 0.0 0.0 0.0 0.0;
# 	 0.0 0.0 1.0 0.0 0.0 0.0;
#      zeros(4, 3) attitude_jacobian(quat)]
# end
#
# function conjugate(q)
# 	s = q[1]
# 	v = q[2:4]
#
# 	return [s; -v]
# end
#
# function L_multiply(q)
# 	s = q[1]
# 	v = q[2:4]
#
# 	SMatrix{4,4}([s -transpose(v);
# 	              v s * I + skew(v)])
# end
#
# function R_multiply(q)
# 	s = q[1]
# 	v = q[2:4]
#
# 	SMatrix{4,4}([s -transpose(v);
# 	              v s * I - skew(v)])
# end
#
# function multiply(q1, q2)
# 	L_multiply(q1) * q2
# end
#
# # eq. 16 http://roboticexplorationlab.org/papers/maximal_coordinate_dynamics.pdf
# function ω_finite_difference(q1, q2, h)
# 	2.0 * multiply(conjugate(q1), (q2 - q1) ./ h)[2:4]
# end
#
# # Cayley map
# function cayley_map(ϕ)
# 	1.0 / sqrt(1.0 + norm(ϕ)^2.0) * [1.0; ϕ]
# end

# function dynamics(model::QuadrupedFloatingBase, h, q0, q1, q2, f, τ)
#
# 	p0 = q0[1:3]
#     quat0 = cayley_map(q0[4:6])
#
# 	p1 = q1[1:3]
# 	quat1 = cayley_map(q1[4:6])
#
# 	p2 = q2[1:3]
# 	quat2 = cayley_map(q2[4:6])
#
# 	# velocities
#     vm1 = (p1 - p0) / h[1]
#     vm2 = (p2 - p1) / h[1]
#
# 	ω1 = ω_finite_difference(quat0, quat1, h)
# 	ω2 = ω_finite_difference(quat1, quat2, h)
#
# 	d_linear = model.mass * (vm1 - vm2) - h[1] * model.mass * model.gravity
# 	d_angular = -1.0 * (model.inertia * ω2 * sqrt(4.0 / h[1]^2.0 - transpose(ω2) * ω2)
# 		+ cross(ω2, model.inertia * ω2)
# 		- model.inertia * ω1 * sqrt(4.0 / h[1]^2.0 - transpose(ω1) * ω1)
# 		+ cross(ω1, model.inertia * ω1))
#
# 	return [d_linear + f; d_angular - 2.0 * τ]
# end
#
# function residual(model::QuadrupedFloatingBase, z, θ, κ)
#     q0 = θ[1:nq]
#     q1 = θ[nq .+ (1:nq)]
#     h = θ[2nq .+ (1:1)]
#     f = [θ[2nq + 1 + (i - 1) * 3 .+ (1:3)] for i = 1:4]
#     p = [θ[2nq + 1 + 3 * 4 + (i - 1) * 3 .+ (1:3)] for i = 1:4]
#
#     q2 = z[1:nq]
#     p_body = q2[1:3]
#
#     τ1 = sum([cross(p[i] - p_body, f[i]) for i = 1:4])
#     f1 = sum(f)
#
#     return dynamics(model, h, q0, q1, q2, f1, τ1);
# end
#
# # generate residual methods
# nq = model.dim.q
# nu = model.dim.u
# nc = model.dim.c
# nz = nq
# nθ = nq + nq + 1 + 3 * 4 + 3 * 4
#
# # Declare variables
# @variables z[1:nz]
# @variables θ[1:nθ]
# @variables κ[1:1]
#
# # Residual
# r = residual(model, z, θ, κ)
# r = Symbolics.simplify.(r)
# rz = Symbolics.jacobian(r, z, simplify = true)
# rθ = Symbolics.jacobian(r, θ, simplify = true)
#
# # Build function
# r_func = eval(build_function(r, z, θ, κ)[2])
# rz_func = eval(build_function(rz, z, θ)[2])
# rθ_func = eval(build_function(rθ, z, θ)[2])
#
# rz_array = similar(rz, Float64)
# rθ_array = similar(rθ, Float64)
#
# u_gravity = [0.0; 0.0; 1.0 * model.mass * 9.81 / 4.0 * h[1]]
# u_stand = vcat([u_gravity for i = 1:4]...)
# z0 = copy(q_body)
# θ0 = [q_body; q_body; h; u_stand; pf1_ref[1]; pf2_ref[1]; pf3_ref[1]; pf4_ref[1]]
#
# # test dynamics
# include_implicit_dynamics()
#
# # options
# opts = InteriorPointOptions(
#    κ_init = 0.1,
#    κ_tol = 1.0e-4,
#    r_tol = 1.0e-8,
#    diff_sol = true)
#
#
# # solver
# ip = interior_point(z0, θ0,
#    r! = r_func, rz! = rz_func,
#    rz = similar(rz, Float64),
#    rθ! = rθ_func,
#    rθ = similar(rθ, Float64),
#    # idx_ineq = idx_ineq,
#    opts = opts)
#
# ip.methods.r! = r_func
# ip.methods.rz! = rz_func
# ip.methods.rθ! = rθ_func
#
# # simulate
# Tsim = 101 - 2
# q_hist = [q_body, q_body]
#
# for t = 1:Tsim
#     ip.z .= copy([q_hist[end]; 1.0 * ones(0)])
#     ip.θ .= copy([q_hist[end-1]; q_hist[end]; h; u_stand; pf1_ref[1]; pf2_ref[1]; pf3_ref[1]; pf4_ref[1]])
#     status = interior_point_solve!(ip)
#
#     if status
#         push!(q_hist, ip.z[1:nq])
#     else
#         println("dynamics failure t = $t")
#         println("res norm = $(norm(ip.r, Inf))")
#         break
#     end
# end
#
# visualize!(vis, model, q_hist,
#     [pf1_ref[1] for t = 1:T],
#     [pf2_ref[1] for t = 1:T],
#     [pf3_ref[1] for t = 1:T],
#     [pf4_ref[1] for t = 1:T],
#     Δt = h)
#

struct QuadrupedFB{I, T} <: Model{I, T}
      n::Int
      m::Int
      d::Int

      mass          # mass
      inertia       # inertia matrix
      inertia_inv   # inertia matrix inverse
      gravity       # gravity
end

function f(model::QuadrupedFB, z, u, w)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      contact_mode = w[12 .+ (1:4)]

      # force in world frame
      F_world = [contact_mode[i] * view(u, (i - 1) * 3 .+ (1:3)) for i = 1:4]

      # torque in body frame
      τ = transpose(MRP(r...)) * sum([cross(w[(i - 1) * 3 .+ (1:3)] - x, F_world[i]) for i = 1:4])

      SVector{12}([v;
                   0.25 * ((1.0 - r' * r) * ω - 2.0 * cross(ω, r) + 2.0*(ω' * r) * r);
                   model.gravity + (1.0 / model.mass) * sum(F_world);
                   model.inertia_inv * (τ - cross(ω, model.inertia * ω))])
end

n, m, d = 12, 12, 16


mass = 1.0
inertia = Diagonal(@SVector[1.0, 1.0, 1.0])
inertia_inv = inv(inertia)
body_height = 0.2

# a1
mass = 108.0 / 9.81
inertia = 10.0 * Diagonal(@SVector[0.017, 0.057, 0.064])
inertia_inv = inv(inertia)
# body_height = 0.24

# laikago
# mass = 200.0 / 9.81
# inertia = Diagonal(@SVector[0.07335, 0.25068, 0.25447])
# inertia_inv = inv(inertia)
# body_height = 0.45


model = QuadrupedFB{Midpoint, FixedTime}(n, m, d,
                  mass,
                  inertia,
                  inertia_inv,
                  @SVector[0.0, 0.0, -9.81])

# control projection problem
mf = 3
nz = mf + 1 + 1 + 1 + 1 + mf
nθ = mf + 1

idx = [3; 1; 2]
function residual(z, θ, κ)
  u = z[1:mf]
  s = z[mf .+ (1:1)]
  y = z[mf + 1 .+ (1:1)]
  w = z[mf + 1 + 1 .+ (1:1)]
  p = z[mf + 1 + 1 + 1 .+ (1:1)]
  v = z[mf + 1 + 1 + 1 + 1 .+ (1:mf)]

  ū = θ[1:mf]
  T = θ[mf .+ (1:1)]

  [
   u - ū - v - [0.0; 0.0; y[1] + p[1]];
   -y[1] - w[1];
   T[1] - u[3] - s[1];
   w .* s .- κ
   p .* u[3] .- κ
   second_order_cone_product(v[idx], u[idx]) .- κ .* [1.0; 0.0; 0.0]
  ]
end

@variables z[1:nz], θ[1:nθ], κ[1:1]

r = residual(z, θ, κ)
r .= simplify(r)
r_func = eval(Symbolics.build_function(r, z, θ, κ)[2])

rz = Symbolics.jacobian(r, z)
rz = simplify.(rz)
rz_func = eval(Symbolics.build_function(rz, z, θ)[2])

rθ = Symbolics.jacobian(r, θ)
rθ = simplify.(rθ)
rθ_func = eval(Symbolics.build_function(rθ, z, θ)[2])

idx_ineq = collect([3, 4, 6, 7])
idx_soc = [collect([3, 1, 2]), collect([10, 8, 9])]#collect([3, 1, 2]), collect([10, 8, 9])]

# ul = -1.0 * ones(m)
# uu = 1.0 * ones(m)
ū = [0.0, 0.0, 0.0]

z0 = 0.1 * ones(nz)
z0[1:mf] = copy(ū)
z0[3] += 1.0
z0[10] += 1.0

θ0 = [copy(ū); 1.0]

# solver
opts_con = InteriorPointOptions(
    κ_init = 1.0,
    κ_tol = 1.0e-4,
    r_tol = 1.0e-8,
    diff_sol = false)

ip_con = interior_point(z0, θ0,
    r! = r_func, rz! = rz_func,
    rz = similar(rz, Float64),
    rθ! = rθ_func,
    rθ = similar(rθ, Float64),
    idx_ineq = idx_ineq,
    idx_soc = idx_soc,
    opts = opts_con)

opts_jac = InteriorPointOptions(
	κ_init = 1.0,
	κ_tol = 1.0e-4,
	r_tol = 1.0e-8,
	diff_sol = true)

ip_jac = interior_point(z0, θ0,
	r! = r_func, rz! = rz_func,
	rz = similar(rz, Float64),
	rθ! = rθ_func,
	rθ = similar(rθ, Float64),
	idx_ineq = idx_ineq,
  idx_soc = idx_soc,
	opts = opts_jac)

interior_point_solve!(ip_con)
interior_point_solve!(ip_jac)

function soc_projection(x)
  proj = second_order_cone_projection(x[[3;1;2]])[1]
	ip_con.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
  ip_con.z[3] += max(1.0, norm(proj[2:3])) * 2.0
  ip_con.z[10] += 1.0
	ip_con.θ .= [x; 100.0]

	status = interior_point_solve!(ip_con)

  !status && (@warn "projection failure (res norm: $(norm(ip_con.r, Inf))) \n
		               z = $(ip_con.z), \n
					   θ = $(ip_con.θ)")

	return ip_con.z[1:mf]
end

soc_projection([100.0, 0.0, -1.0])

function soc_projection_jacobian(x)
  proj = second_order_cone_projection(x[[3;1;2]])[1]
	ip_con.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
  ip_con.z[3] += max(1.0, norm(proj[2:3])) * 2.0
  ip_con.z[10] += 1.0
	ip_con.θ .= [x; 100.0]

	interior_point_solve!(ip_jac)

	return ip_jac.δz[1:mf, 1:mf]
end

soc_projection_jacobian([10.0, 0.0, 1.0])

function fd(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)]) for i = 1:4]...)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	# x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
	return x⁺
	# return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

fd(model, ones(model.n), zeros(model.m), zeros(model.d), h, 1)
function fdx(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
    u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)]) for i = 1:4]...)
    rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
    # x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
    ∇rz = ForwardDiff.jacobian(rz, x⁺)
    rx(z) = fd(model, x⁺, z, u_proj, w, h, t) # implicit midpoint integration
    ∇rx = ForwardDiff.jacobian(rx, x)
    return -1.0 * ∇rz \ ∇rx
end

fdx(model, zeros(model.n), zeros(model.m), zeros(model.d), h, 1)

function fdu(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
    u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)]) for i = 1:4]...)
	u_proj_jac = cat([soc_projection_jacobian(u[(i - 1) * 3 .+ (1:3)]) for i = 1:4]..., dims=(1,2))

    rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
    # x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
    ∇rz = ForwardDiff.jacobian(rz, x⁺)
    ru(z) = fd(model, x⁺, x, z, w, h, t) # implicit midpoint integration
    ∇ru = ForwardDiff.jacobian(ru, u_proj)
    return -1.0 * ∇rz \ (∇ru * u_proj_jac)
    # return -1.0 * ∇rz \ ∇ru

	# f(z) = fd(model, x, z, w, h, t)
	# return ForwardDiff.jacobian(f, u)
end

fdu(model, zeros(model.n), zeros(model.m), zeros(model.d), h, 1)

##
x0 = [q_body; zeros(3); zeros(3)]
u_gravity = [0.0; 0.0; model.mass * 9.81 / 4.0]
u_stand = vcat([u_gravity for i = 1:4]...)
w_stand = [pf1_ref[1]; pf2_ref[2]; pf3_ref[3]; pf4_ref[4]; 1; 1; 1; 1]

soc_projection(u_gravity)
T_sim = 101
x_hist = [x0]
for t = 1:T_sim-1
    push!(x_hist, fd(model, x_hist[end], u_stand, w_stand, h, 1))
end

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x_hist,
    [pf1_ref[1] for t = 1:T_sim],
    [pf2_ref[1] for t = 1:T_sim],
    [pf3_ref[1] for t = 1:T_sim],
    [pf4_ref[1] for t = 1:T_sim],
    Δt = h)
