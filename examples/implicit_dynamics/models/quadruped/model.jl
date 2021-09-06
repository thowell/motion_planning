include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))
# include_implicit_dynamics()
include(joinpath(pwd(), "examples/implicit_dynamics/soc_projection_2.jl"))

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

f_max = 10.0
f_min = 0.0
μ_friction = 1.0

# a1
# mass = 108.0 / 9.81
# inertia = 10.0 * Diagonal(@SVector[0.017, 0.057, 0.064])
# inertia_inv = inv(inertia)
# body_height = 0.24

# laikago
# mass = 200.0 / 9.81
# inertia = Diagonal(@SVector[0.07335, 0.25068, 0.25447])
# inertia_inv = inv(inertia)
# body_height = 0.45

SOC_PROJ = true

model = QuadrupedFB{Midpoint, FixedTime}(n, m, d,
                  mass,
                  inertia,
                  inertia_inv,
                  @SVector[0.0, 0.0, -9.81])

function fd(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = SOC_PROJ ? vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...) : u
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	# x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
	return x⁺
end

fd(model, ones(model.n), zeros(model.m), zeros(model.d), h, 1)
function fdx(model::QuadrupedFB{Midpoint, FixedTime}, x, u, w, h, t)
    u_proj = SOC_PROJ ? vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...) : u
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
    u_proj = SOC_PROJ ? vcat([soc_projection(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]...) : u
	u_proj_jac = SOC_PROJ ? cat([soc_projection_jacobian(u[(i - 1) * 3 .+ (1:3)], f_min, f_max, μ_friction) for i = 1:4]..., dims=(1,2)) : I

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

fdu(model, ones(model.n), ones(model.m), ones(model.d), h, 1)

##
# x0 = [q_body; zeros(3); zeros(3)]
# u_gravity = [0.0; 0.0; model.mass * 9.81 / 4.0]
# u_stand = vcat([u_gravity for i = 1:4]...)
# w_stand = [pf1_ref[1]; pf2_ref[2]; pf3_ref[3]; pf4_ref[4]; 1; 1; 1; 1]
#
# # soc_projection(u_gravity)
# T_sim = 101
# x_hist = [x0]
# for t = 1:T_sim-1
#     push!(x_hist, fd(model, x_hist[end], u_stand, w_stand, h, 1))
# end
#
# include(joinpath(pwd(), "models/visualize.jl"))
# vis = Visualizer()
# render(vis)
# visualize!(vis, model, x_hist,
#     [pf1_ref[1] for t = 1:T_sim],
#     [pf2_ref[1] for t = 1:T_sim],
#     [pf3_ref[1] for t = 1:T_sim],
#     [pf4_ref[1] for t = 1:T_sim],
#     Δt = h)
#
# function foot_vis!(vis, i, t;
#     h = 0.01,
#     r = 0.05)
#     if convert(Bool, contact_modes[t][i])
#         setobject!(vis["f$(i)_$(j)_$(t)"],
#             Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, h), r),
#             MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, 0.01)))
#         settransform!(vis["f$(i)_$(j)_$(t)"], Translation(eval(Symbol("pf$(i)_ref"))[t]))
#     end
#     # setvisible!(vis["f$(i)_$(j)_$(t)"], convert(Bool, contact_modes[t][i]))
# end
#
# function cone_vis!(vis, i, t;
#     h = 0.05,
#     l = h * 1.0,
#     w = h * 1.45)
#
#     pyramid = Pyramid(Point3(0.0, 0.0, 0.0), l, w)
#
#     n = 50
#     if convert(Bool, contact_modes[t][i])
#         for p = 1:n
#             setobject!(vis["pyramid$(i)_$(p)_$(t)"], pyramid,
#                 MeshPhongMaterial(
#                 # color = RGBA(1,153/255,51/255, 1.0))
#                 color = RGBA(51/255,1,1, 0.01))
#                 )
#             settransform!(vis["pyramid$(i)_$(p)_$(t)"],
#                 compose(Translation(eval(Symbol("pf$(i)_ref"))[t][1], eval(Symbol("pf$(i)_ref"))[t][2], l),
#                     LinearMap(RotX(π) * RotZ(π * p / n))),
#                     )
#             setvisible!(vis["pyramid$(i)_$(p)_$(t)"], convert(Bool, contact_modes[t][i]))
#         end
#     end
# end
#
# for i = 1:4
#     for t = 1:T
#         foot_vis!(vis, i, t)
#         cone_vis!(vis, i, t)
#     end
# end
