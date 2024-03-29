include_implicit_dynamics()

"""
    Cartpole w/ internal friction
"""

struct Cartpole{T}
    dim::Dimensions

    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
end

function kinematics(model::Cartpole, q)
    [q[1] + model.l * sin(q[2]); -model.l * cos(q[2])]
end


lagrangian(model::Cartpole, q, q̇) = 0.0

function M_func(model::Cartpole, x)
    H = @SMatrix [model.mc + model.mp model.mp * model.l * cos(x[2]);
				  model.mp * model.l * cos(x[2]) model.mp * model.l^2.0]
    return H
end

function B_func(model::Cartpole, x)
    @SMatrix [1.0;
              0.0]
end

function P_func(model::Cartpole, x)
    @SMatrix [1.0 0.0;
              0.0 1.0]
end

function C_func(model::Cartpole, q, q̇)
    # c_func(model, q, q̇) * q̇ - τ_func(model, q)
    C = @SMatrix [0.0 -1.0 * model.mp * q̇[2] * model.l * sin(q[2]);
	 			  0.0 0.0]
    G = @SVector [0.0,
				  model.mp * model.g * model.l * sin(q[2])]

    return -C * q̇ + G
end

model = Cartpole(Dimensions(2, 1, 0, 4),
    1.0, 0.2, 0.5, 9.81)

model_no_friction = Cartpole(Dimensions(2, 1, 0, 4),
    1.0, 0.2, 0.5, 9.81)

function lagrangian_derivatives(model::Cartpole, q, v)
	D1L = -1.0 * C_func(model, q, v)
    D2L = M_func(model, q) * v
	return D1L, D2L
end

function dynamics(model::Cartpole, h, q0, q1, u1, λ1, q2)
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * u1
        + transpose(P_func(model, q2)) * λ1)
end

function dynamics_no_friction(model::Cartpole, h, q0, q1, u1, λ1, q2)
	# evalutate at midpoint
	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

	return (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
		+ B_func(model, qm2) * u1)
        # + transpose(P_func(model, q2)) * λ1)
end

function residual(model::Cartpole, z, θ, κ)

    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]
    μ_slider = θ[2nq + nu + 1 .+ (1:1)]
    μ_angle = θ[2nq + nu + 1 + 1 .+ (1:1)]

    q2 = z[1:nq]
    c1 = z[nq .+ (1:nc)]
    c2 = z[nq + nc .+ (1:nc)]

    vT1 = (q2[1] - q1[1]) / h
    vT2 = (q2[2] - q1[2]) / h

    β1 = c1[1:2]
    β2 = c2[1:2]

    η1 = c1[2 .+ (1:2)]
    η2 = c2[2 .+ (1:2)]

    λ1 = [β1[2]; β2[2]]

    [
     dynamics(model, h, q0, q1, u1, λ1, q2);
     η1[2] - vT1[1];
     # β1[1] - (μ_slider[1] * 1.0); # static normal
     β1[1] .- μ_slider[1] * (model.mp + model.mc) * model.g * h[1];
     second_order_cone_product(η1, β1) - κ .* [1.0; 0.0];
     η2[2] - vT2[1];
     # β2[1] - (μ_angle[1] * 1.0); # static "normal"
     β2[1] .- μ_angle[1] * (model.mp * model.g * model.l) * h[1];
     second_order_cone_product(η2, β2) - κ .* [1.0; 0.0];
    ]
end

function residual_no_friction(model::Cartpole, z, θ, κ)

    nq = model.dim.q
    nu = model.dim.u
    nc = model.dim.c

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]

    return dynamics_no_friction(model, h, q0, q1, u1, zeros(nc), q2);
end

nq = model.dim.q
nu = model.dim.u
nc = model.dim.c
nz = nq + nc + nc
nz_no_friction = nq
nθ = nq + nq + nu + 1 + 2

# Declare variables
@variables z[1:nz]
@variables z_no_friction[1:nz_no_friction]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r = vec(residual(model, z, θ, κ))
r = Symbolics.simplify.(r)
rz = Symbolics.jacobian(r, z, simplify = true)
rθ = Symbolics.jacobian(r, θ, simplify = true)

# Build function
r_func = eval(build_function(r, z, θ, κ)[2])
rz_func = eval(build_function(rz, z, θ)[2])
rθ_func = eval(build_function(rθ, z, θ)[2])

rz_array = similar(rz, Float64)
rθ_array = similar(rθ, Float64)

@save joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array
@load joinpath(@__DIR__, "dynamics/residual.jl") r_func rz_func rθ_func rz_array rθ_array

# Residual
r_no_friction = residual_no_friction(model, z_no_friction, θ, κ)
r_no_friction = Symbolics.simplify.(r_no_friction)
rz_no_friction = Symbolics.jacobian(r_no_friction, z_no_friction, simplify = true)
rθ_no_friction = Symbolics.jacobian(r_no_friction, θ, simplify = true)

# Build function
r_no_friction_func = eval(build_function(r_no_friction, z_no_friction, θ, κ)[2])
rz_no_friction_func = eval(build_function(rz_no_friction, z_no_friction, θ)[2])
rθ_no_friction_func = eval(build_function(rθ_no_friction, z_no_friction, θ)[2])

rz_no_friction_array = similar(rz_no_friction, Float64)
rθ_no_friction_array = similar(rθ_no_friction, Float64)

@save joinpath(@__DIR__, "dynamics/residual_no_friction.jl") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array
@load joinpath(@__DIR__, "dynamics/residual_no_friction.jl") r_no_friction_func rz_no_friction_func rθ_no_friction_func rz_no_friction_array rθ_no_friction_array

idx_ineq = collect(1:0)
idx_soc = [collect([nq .+ (1:2)]...), collect([nq + 2 .+ (1:2)]...), collect([nq + nc .+ (1:2)]...), collect([nq + nc + 2 .+ (1:2)]...)]

z_subset_init = [1.0; 0.1; 1.0; 0.1; 1.0; 0.1; 1.0; 0.1]

q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
u1 = [0.0]
h = 0.1


z0 = copy([q1; ones(nc + nc)])
θ0 = [q0; q1; u1; h; 0.1; 0.1]

# options
opts = InteriorPointOptions(
   κ_init = 0.1,
   κ_tol = 1.0e-4,
   r_tol = 1.0e-8,
   diff_sol = true)

# solver
ip = interior_point(z0, θ0,
   r! = r_func, rz! = rz_func,
   rz = similar(rz, Float64),
   rθ! = rθ_func,
   rθ = similar(rθ, Float64),
   # idx_ineq = idx_ineq,
   idx_soc = idx_soc,
   opts = opts)

# simulate
T = 100
q_hist = [q0, q1]

for t = 1:T-1
    u1 = [t == 1 ? 1.0 : t == 2 ? 0.0 : 0.0]
    ip.z .= copy([q_hist[end]; z_subset_init])
    ip.θ .= copy([q_hist[end-1]; q_hist[end]; u1; h; 0.1; 0.05])
    status = interior_point_solve!(ip)

    if status
        push!(q_hist, ip.z[1:nq])
    else
        println("dynamics failure t = $t")
        println("res norm = $(norm(ip.r, Inf))")
        # break
    end
end

include(joinpath(@__DIR__, "visuals.jl"))
include(joinpath(pwd(), "models/visualize.jl"))

vis = Visualizer()
# open(vis)
render(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 1)

visualize!(vis, model, q_hist, Δt = h)
