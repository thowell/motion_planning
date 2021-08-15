using Plots
include_ddp()

# function box_projection(x, l, u)
# 	max.(min.(x, u), l)
# end
#
# box_projection(-1.0 * ones(2), zeros(2), ones(2))
#
# function box_projection_jacobian(x, l, u)
# 	n = length(x)
# 	a = ones(n)
# 	upper = u - x
# 	lower = x - l
# 	for i = 1:n
# 		if upper[i] < 0.0 || lower[i] < 0.0
# 			a[i] = 0.0
# 		end
# 	end
# 	return Diagonal(a)
# end
#
# box_projection_jacobian(1.0 * ones(2), zeros(2), ones(2))

# Model
include_model("rocket3D")

n = model.n
m = model.m

ul = [-5.0; -5.0; 0.0]
uu = [5.0; 5.0; 15.0]

contact_control_path = "/home/taylor/Research/ContactControl.jl/src"

using Parameters

# Utilities
include(joinpath(contact_control_path, "utils.jl"))

# Solver
include(joinpath(contact_control_path, "solver/cones.jl"))
include(joinpath(contact_control_path, "solver/interior_point.jl"))
# include(joinpath(contact_control_path, "solver/lu.jl"))

m = 3
nz = m * 7
nθ = m * 3

function residual(z, θ, κ)
    u = z[1:m]
    s1 = z[m .+ (1:m)]
    s2 = z[m + m .+ (1:m)]
    y1 = z[m + m + m .+ (1:m)]
    y2 = z[m + m + m + m .+ (1:m)]
    z1 = z[m + m + m + m + m .+ (1:m)]
    z2 = z[m + m + m + m + m + m .+ (1:m)]

    ū = θ[1:m]
    ul = θ[m .+ (1:m)]
    uu = θ[m + m .+ (1:m)]

    [
     (u - ū) - y1 + y2;
     -y1 - z1;
     -y2 - z2;
     (uu - u) - s1;
     (u - ul) - s2;
     s1 .* z1 .- κ;
     s2 .* z2 .- κ;
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

idx_ineq = collect([(m .+ (1:(m + m)))..., (m + m + m + m + m .+ (1:(m + m)))...])


# ul = -1.0 * ones(m)
# uu = 1.0 * ones(m)
ū = [0.0, 0.0, 0.0]

z0 = 0.1 * ones(nz)
z0[1:m] = copy(ū)

θ0 = [ū; ul; uu]


# solver
opts_con = InteriorPointOptions(
  κ_init = 0.1,
  κ_tol = 1.0e-4,
  r_tol = 1.0e-8,
  diff_sol = false)

ip_con = interior_point(z0, θ0,
  r! = r_func, rz! = rz_func,
  rz = similar(rz, Float64),
  rθ! = rθ_func,
  rθ = similar(rθ, Float64),
  idx_ineq = idx_ineq,
  opts = opts_con)

opts_jac = InteriorPointOptions(
	κ_init = 0.1,
	κ_tol = 1.0e-2,
	r_tol = 1.0e-8,
	diff_sol = true)

ip_jac = interior_point(z0, θ0,
	r! = r_func, rz! = rz_func,
	rz = similar(rz, Float64),
	rθ! = rθ_func,
	rθ = similar(rθ, Float64),
	idx_ineq = idx_ineq,
	opts = opts_jac)

interior_point_solve!(ip_con)
interior_point_solve!(ip_jac)

ip_con.z[1:m]

function box_projection(x, l, u)
	ip_con.z .= [x; 0.1 * ones(m * 6)]
	ip_con.θ .= [x; l; u]

	interior_point_solve!(ip_con)

	return ip_con.z[1:m]
end

box_projection(-1.0 * ones(m), zeros(m), ones(m))

function box_projection_jacobian(x, l, u)
	ip_jac.z .= [x; 0.1 * ones(m * 6)]
	ip_jac.θ .= [x; l; u]

	interior_point_solve!(ip_jac)

	return ip_jac.δz[1:m, 1:m]
end

box_projection_jacobian(-1.0 * ones(m), zeros(m), ones(m))

function fd(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = box_projection(u, ul, uu)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	x⁺ = newton(rz, copy(x))
	return x⁺
	# return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

fd(model, ones(model.n), zeros(model.m), zeros(model.d), h, 1)


function fdx(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = box_projection(u, ul, uu)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	x⁺ = newton(rz, copy(x))
	∇rz = ForwardDiff.jacobian(rz, x⁺)
	rx(z) = fd(model, x⁺, z, u_proj, w, h, t) # implicit midpoint integration
	∇rx = ForwardDiff.jacobian(rx, x)
	return -1.0 * ∇rz \ ∇rx
	# f(z) = fd(model, z, u, w, h, t)
	# return ForwardDiff.jacobian(f, x)
end

fdx(model, zeros(model.n), zeros(model.m), zeros(model.d), h, 1)


function fdu(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = box_projection(u, ul, uu)
	u_proj_jac = box_projection_jacobian(u, ul, uu)

	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	x⁺ = newton(rz, copy(x))
	∇rz = ForwardDiff.jacobian(rz, x⁺)
	ru(z) = fd(model, x⁺, x, z, w, h, t) # implicit midpoint integration
	∇ru = ForwardDiff.jacobian(ru, u_proj)
	return -1.0 * ∇rz \ (∇ru * u_proj_jac)
	# f(z) = fd(model, x, z, w, h, t)
	# return ForwardDiff.jacobian(f, u)
end

fdu(model, zeros(model.n), zeros(model.m), zeros(model.d), h, 1)

# Time
T = 201
h = 0.01

# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[1] = 2.5
x1[2] = 0.0
x1[3] = 10.0
mrp = MRP(RotY(-0.5 * π) * RotX(0.0 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]
x1[9] = -5.0

# visualize!(vis, model, [x1], Δt = h)

xT = zeros(model.n)
# xT[1] = 2.5
# xT[2] = 0.0
xT[3] = model.length

u_ref = [0.0; 0.0; 0.0]#model.mass * 9.81]
ū = [u_ref + [1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = h * [(t < T ? 1.0 * Diagonal([1.0e-1 * ones(3); 0.0 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)])
        : 0.0 * Diagonal(0.0 * ones(model.n))) for t = 1:T]
q = h * [-2.0 * Q[t] * xT for t = 1:T]

R = h * [Diagonal([10000.0; 10000.0; 100.0]) for t = 1:T-1]
r = h * [-2.0 * R[t] * u_ref  for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
p = [t < T ? 2 * m : n for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		# c .= [ul - u; u - uu]
	elseif t == T
		xT = cons.con[T].info[:xT]
		c .= x - xT
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Trajectories
maximum(hcat(ū...))
plot(hcat(ū...)', linetype = :steppost)
plot(hcat(x̄...)[1:3, :]', linetype = :steppost)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x̄, Δt = h)

# ū_fixed_time = ū

# Visualize
obj_rocket = joinpath(pwd(), "models/starship/Starship.obj")
mtl_rocket = joinpath(pwd(), "models/starship/Starship.mtl")
ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale=1.0)
setobject!(vis["rocket"]["starship"], ctm)

settransform!(vis["rocket"]["starship"],
	compose(Translation(0.0, 0.0, -model.length),
		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))

default_background!(vis)
settransform!(vis["rocket"],
	compose(Translation(0.0, 0.0, 0.0),
	LinearMap(RotY(0.0))))
