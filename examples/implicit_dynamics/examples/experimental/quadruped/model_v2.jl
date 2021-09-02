include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))
include_implicit_dynamics()

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

f_max = 100.0
f_min = 0.0
μ_friction = 0.5

# a1
# mass = 108.0 / 9.81
# inertia = 2.5 * Diagonal(@SVector[0.017, 0.057, 0.064])
# inertia_inv = inv(inertia)
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
nz = 3 + 2 + 1 + 1 + 3 + 3 + 3 + 2
nθ = 3 + 1 + 1 + 1

idx = [3; 1; 2]
function residual(z, θ, κ)
    m = 3

    f = z[1:m]
    s = z[m .+ (1:2)]
    y1 = z[m + 2 .+ (1:1)]
    y2 = z[m + 2 + 1 .+ (1:1)]
    y3 = z[m + 2 + 1 + 1 .+ (1:3)]
    β = z[m + 2 + 1 + 1 + 3 .+ (1:3)]
    η = z[m + 2 + 1 + 1 + 3 + 3 .+ (1:3)]
    z = z[m + 2 + 1 + 1 + 3 + 3 + 3 .+ (1:2)]

    f̄ = θ[1:m]
    f_min = θ[m .+ (1:1)]
    f_max = θ[m + 1 .+ (1:1)]
    μ = θ[m + 1 + 1 .+ (1:1)]

    A = [0.0 0.0 μ; 1.0 0.0 0.0; 0.0 1.0 0.0]

    return [f - f̄ + [0.0; 0.0; -y1[1] + y2[1]] + transpose(A) * y3;
            [-y1[1]; -y2[1]] - z;
            f_max[1] - f[3] - s[1];
            f[3] - f_min[1] - s[2];
            A * f - β;
            -y3 - η;
            second_order_cone_product(β, η) - [κ; 0.0; 0.0];
            s .* z .- κ]
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

idx_ineq = collect([4, 5, 17, 18])
idx_soc = [collect([11, 12, 13]), collect([14, 15, 16])]

# ul = -1.0 * ones(m)
# uu = 1.0 * ones(m)
ū = [0.0; 0.0; 0.0]
u_min = 0.0
u_max = 1.0
scaling = 1.0
z0 = [ū; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]

θ0 = [copy(ū); u_min; u_max; scaling]

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

function soc_projection(x, x_min, x_max, scaling)
    ip_con.z .= [x; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]
    ip_con.θ .= [x; x_min; x_max; scaling]

    status = interior_point_solve!(ip_con)

    !status && (@warn "projection failure (res norm: $(norm(ip_con.r, Inf))) \n
    	               z = $(ip_con.z), \n
    				   θ = $(ip_con.θ)")

	return ip_con.z[1:3]
end

soc_projection([100.0, 0.0, -1.0], 0.0, 1.0, 1.0)

function soc_projection_jacobian(x, x_min, x_max, scaling)
    ip_con.z .= [x; 1.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.1; 0.1; 1.0; 0.1; 0.1; 1.0; 1.0]
    ip_con.θ .= [x; x_min; x_max; scaling]

	interior_point_solve!(ip_jac)

	return ip_jac.δz[1:3, 1:3]
end

soc_projection_jacobian([0.0, 0.0, 1.0], 0.0, 10.0, 1.0)

function fd(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	# x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
	return x⁺
	# return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

fd(model, ones(model.n), zeros(model.m), zeros(model.d), h, 1)
function fdx(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
    u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...)
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
    u_proj = vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...)
	u_proj_jac = cat([soc_projection_jacobian(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]..., dims=(1,2))

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

# soc_projection(u_gravity)
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
