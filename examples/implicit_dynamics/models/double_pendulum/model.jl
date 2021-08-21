include(joinpath(pwd(), "examples/implicit_dynamics/utils.jl"))

"""
    Double pendulum
"""

struct DoublePendulum{T}
    dim::Dimensions

    m1::T    # mass link 1
    J1::T    # inertia link 1
    l1::T    # length link 1
    lc1::T   # length to COM link 1

    m2::T    # mass link 2
    J2::T    # inertia link 2
    l2::T    # length link 2
    lc2::T   # length to COM link 2

    g::T     # gravity

    b1::T    # joint friction
    b2::T
end

lagrangian(model::DoublePendulum, q, q̇) = 0.0

function kinematics(model::DoublePendulum, x)
    @SVector [model.l1 * sin(x[1]) + model.l2 * sin(x[1] + x[2]),
              -1.0 * model.l1 * cos(x[1]) - model.l2 * cos(x[1] + x[2])]
end

function kinematics_elbow(model::DoublePendulum, x)
    @SVector [model.l1 * sin(x[1]),
              -1.0 * model.l1 * cos(x[1])]
end

function M_func(model::DoublePendulum, x)
    a = (model.J1 + model.J2 + model.m2 * model.l1 * model.l1
         + 2.0 * model.m2 * model.l1 * model.lc2 * cos(x[2]))

    b = model.J2 + model.m2 * model.l1 * model.lc2 * cos(x[2])

    c = model.J2

    @SMatrix [a b;
              b c]
end

function τ_func(model::DoublePendulum, x)
    a = (-1.0 * model.m1 * model.g * model.lc1 * sin(x[1])
         - model.m2 * model.g * (model.l1 * sin(x[1])
         + model.lc2 * sin(x[1] + x[2])))

    b = -1.0 * model.m2 * model.g * model.lc2 * sin(x[1] + x[2])

    @SVector [a,
              b]
end

function c_func(model::DoublePendulum, q, q̇)
    a = -2.0 * model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[2]
    b = -1.0 * model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[2]
    c = model.m2 * model.l1 * model.lc2 * sin(q[2]) * q̇[1]
    d = 0.0

    @SMatrix [a b;
              c d]
end

function B_func(model::DoublePendulum, x)
    @SMatrix [0.0;
              1.0]
end

function C_func(model::DoublePendulum, q, q̇)
    c_func(model, q, q̇) * q̇ - τ_func(model, q)
end

function ϕ_func(model, q)
    # SVector{model.dim.c}([q[1], 0.5 * π - q[2], q[2] + 0.5 * π])
    SVector{model.dim.c}([0.5 * π - q[2], q[2] + 0.5 * π])
end

function P_func(model, q)
    ϕ(z) = ϕ_func(model, z)
    ForwardDiff.jacobian(ϕ, q)
end

model = DoublePendulum(Dimensions(2, 1, 0, 2),
    1.0, 0.33, 1.0, 0.5, 1.0, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

model_no_impact = DoublePendulum(Dimensions(2, 1, 0, 0),
    1.0, 0.33, 1.0, 0.5, 1.0, 0.33, 1.0, 0.5, 9.81, 0.1, 0.1)

function lagrangian_derivatives(model::DoublePendulum, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function dynamics(model::DoublePendulum, h, q0, q1, u1, λ1, q2)
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * u1
        + transpose(P_func(model, q2)) * λ1
        - h[1] * 0.5 .* vm2) # damping
end

function dynamics_no_impact(model::DoublePendulum, h, q0, q1, u1, λ1, q2)
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * u1
        - h[1] * 0.5 .* vm2) # damping
end

function residual(model::DoublePendulum, z, θ, κ)
    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    λ1 = z[nq .+ (1:nc)]
    s1 = z[nq + nc .+ (1:nc)]

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
     s1 .- ϕ_func(model, q2);
     λ1 .* s1 .- κ;
    ]

end

function residual_no_impact(model::DoublePendulum, z, θ, κ)
    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    return dynamics_no_impact(model, h, q0, q1, u1, zeros(model.dim.c), q2)
end

# generate residual methods
# nq = model.dim.q
# nu = model.dim.u
# nc = model.dim.c
# nz = nq + nc + nc
# nz_no_impact = nq
# nθ = nq + nq + nu + 1
#
# # Declare variables
# @variables z[1:nz]
# @variables z_no_impact[1:nz_no_impact]
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
# @save joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array

# Residual
# r_no_impact = residual_no_impact(model, z_no_impact, θ, κ)
# r_no_impact = Symbolics.simplify.(r_no_impact)
# rz_no_impact = Symbolics.jacobian(r_no_impact, z_no_impact, simplify = true)
# rθ_no_impact = Symbolics.jacobian(r_no_impact, θ, simplify = true)
#
# # Build function
# r_no_impact_func = eval(build_function(r_no_impact, z_no_impact, θ, κ)[2])
# rz_no_impact_func = eval(build_function(rz_no_impact, z_no_impact, θ)[2])
# rθ_no_impact_func = eval(build_function(rθ_no_impact, z_no_impact, θ)[2])
#
# rz_no_impact_array = similar(rz_no_impact, Float64)
# rθ_no_impact_array = similar(rθ_no_impact, Float64)
#
# @save joinpath(@__DIR__, "dynamics/residual_no_impact.jl") r_no_impact_func rz_no_impact_func rθ_no_impact_func rz_no_impact_array rθ_no_impact_array
@load joinpath(@__DIR__, "dynamics/residual_no_impact.jl") r_no_impact_func rz_no_impact_func rθ_no_impact_func rz_no_impact_array rθ_no_impact_array

# q0 = [0.5 * π; 0.0]
# q1 = [0.5 * π; 0.0]
# u1 = [0.0]
# h = 0.1
#
# z0 = copy([q1; ones(nc + nc)])
# θ0 = [q0; q1; u1; h]
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
# idx_ineq = collect([nq .+ (1:(nc + nc))]...)
#
# # solver
# ip = interior_point(z0, θ0,
#    r! = r_func, rz! = rz_func,
#    rz = similar(rz, Float64),
#    rθ! = rθ_func,
#    rθ = similar(rθ, Float64),
#    idx_ineq = idx_ineq,
#    opts = opts)
#
# ip.methods.r! = r_func
# ip.methods.rz! = rz_func
# ip.methods.rθ! = rθ_func
#
# # simulate
# T = 50
# q_hist = [q0, q1]
#
# for t = 1:T
#     ip.z .= copy([q_hist[end]; 0.1 * ones(nc + nc)])
#     ip.θ .= copy([q_hist[end-1]; q_hist[end]; u1; h])
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
# include(joinpath(@__DIR__, "visuals.jl"))
# include(joinpath(pwd(), "models/visualize.jl"))
#
# vis = Visualizer()
# render(vis)
# default_background!(vis)
# settransform!(vis["/Cameras/default"],
#         compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
# setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 30)
#
# visualize!(vis, model, q_hist, Δt = h)
