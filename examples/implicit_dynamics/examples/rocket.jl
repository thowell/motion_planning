using Plots
Random.seed!(0)

include_implicit_dynamics()
include_ddp()
include_model("rocket3D")

h = 0.05
n = model.n
m = model.m

# control limits
ul = [-5.0; -5.0; 0.0]
uu = [5.0; 5.0; 12.5]

# control projection problem
m = 3
nz = m + 1 + 1 + 1 + 1 + m
nθ = m + 1

idx = [3; 1; 2]
function residual(z, θ, κ)
    u = z[1:m]
    s = z[m .+ (1:1)]
    y = z[m + 1 .+ (1:1)]
    w = z[m + 1 + 1 .+ (1:1)]
    p = z[m + 1 + 1 + 1 .+ (1:1)]
    v = z[m + 1 + 1 + 1 + 1 .+ (1:m)]

    ū = θ[1:m]
    T = θ[m .+ (1:1)]

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
z0[1:m] = copy(ū)
z0[3] += 1.0
z0[10] += 1.0

θ0 = [copy(ū); uu[3]]

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
	κ_tol = 1.0e-2,
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
	ip_con.θ .= [x; uu[3]]

	status = interior_point_solve!(ip_con)

    !status && (@warn "projection failure (res norm: $(norm(ip_con.r, Inf))) \n
		               z = $(ip_con.z), \n
					   θ = $(ip_con.θ)")

	return ip_con.z[1:m]
end

soc_projection([100.0, 0.0, -1.0])

function soc_projection_jacobian(x)
    proj = second_order_cone_projection(x[[3;1;2]])[1]
	ip_con.z .= [proj[2:3]; proj[1]; 0.1 * ones(7)]
    ip_con.z[3] += max(1.0, norm(proj[2:3])) * 2.0
    ip_con.z[10] += 1.0
	ip_con.θ .= [x; uu[3]]

	interior_point_solve!(ip_jac)

	return ip_jac.δz[1:m, 1:m]
end

soc_projection_jacobian([10.0, 0.0, 1.0])

function fd(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = soc_projection(u)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
	# x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
	return x⁺
	# return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

fd(model, ones(model.n), zeros(model.m), zeros(model.d), h, 1)

function fdx(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = soc_projection(u)
	rz(z) = fd(model, z, x, u_proj, w, h, t) # implicit midpoint integration
    # x⁺ = newton(rz, copy(x))
    x⁺ = levenberg_marquardt(rz, copy(x))
	∇rz = ForwardDiff.jacobian(rz, x⁺)
	rx(z) = fd(model, x⁺, z, u_proj, w, h, t) # implicit midpoint integration
	∇rx = ForwardDiff.jacobian(rx, x)
	return -1.0 * ∇rz \ ∇rx
end

fdx(model, zeros(model.n), zeros(model.m), zeros(model.d), h, 1)


function fdu(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	u_proj = soc_projection(u)
	u_proj_jac = soc_projection_jacobian(u)

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

# Time
T = 101
# h = 0.05

# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[1] = 2.5
x1[2] = 2.5
x1[3] = 10.0
mrp = MRP(RotZ(0.25 * π) * RotY(-0.5 * π))
x1[4:6] = [mrp.x; mrp.y; mrp.z]
x1[9] = -1.0

# visualize!(vis, model, [x1], Δt = h)

xT = zeros(model.n)
# xT[1] = 2.5
# xT[2] = 0.0
xT[3] = model.length
mrpT = MRP(RotZ(0.25 * π) * RotY(0.0))
xT[4:6] = [mrpT.x; mrpT.y; mrpT.z]

u_ref = [0.0; 0.0; 0.0]#model.mass * 9.81]
ū = [u_ref + [1.0e-3; 1.0e-3; 1.0e-3] .* randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# visualize!(vis, model, x̄, Δt = h)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = h * [(t < T ? 1.0 * Diagonal([1.0e-1 * ones(3); 0.0 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)])
        : 1000.0 * Diagonal(1.0 * ones(model.n))) for t = 1:T]
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
x_con = [-0.5; 0.5]
y_con = [-0.75; 0.75]
p = [t < T ? 2 * m + 1 : n - 2 + 4 for t = 1:T]
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m + 1))
info_T = Dict(:xT => xT, :inequality => (1:4))
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]
idx_T = collect([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		# c[1:2 * m] .= [ul - u; u - uu]
		c[2 * m + 1] = model.length - x[3]
	elseif t == T
		c[1] = x_con[1] - x[1]
		c[2] = x[1] - x_con[2]
		c[3] = y_con[1] - x[2]
		c[4] = x[2] - y_con[2]
		xT = cons.con[T].info[:xT]
		c[4 .+ (1:(n - 2))] .= (x - xT)[idx_T]
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

u_ref = [0.0; 0.0; 0.0] # model.mass * 9.81]
ū = [u_ref + [1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m) for t = 1:T-1]

# Solve
@time constrained_ddp_solve!(prob,
    linesearch = :armijo,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 0.001,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

x̄_soc = x̄
ū_soc = ū

# @save "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/rocket_landing_soc.jld2" x̄_soc ū_soc
# @load "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/rocket_landing_soc.jld2"

all([second_order_cone_projection(ū[t][idx])[2] for t = 1:T-1])
[second_order_cone_projection(ū[t][idx])[2] for t = 1:T-1]
u_proj = [soc_projection(ū[t]) for t = 1:T-1]

t = 3
fd(model, x̄[t], ū[t], zeros(model.d), h, t) - x̄[t+1]
soc_projection(ū[t])
norm(ū[t][1:2]) - ū[t][3]

soc_projection(ū[t])
second_order_cone_projection(ū[t][idx])

# Trajectories
maximum(hcat(ū...))
maximum(hcat(u_proj...))

t = range(0, stop = h * (T-1), length = T)
# plt = plot(t, hcat(ū..., ū[end])',
plt = plot(t, hcat(u_proj..., u_proj[end])[1:3, :]',
	width = 2.0,
	color = [:magenta :orange :cyan],
	title = "rocket soft landing",
	xlabel = "time (s)",
	ylabel = "control",
	labels = ["f1" "f2" "f3"],
	legend = :right,
	linetype = :steppost)

# # savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/rocket_control.png")
#
plot(hcat(x̄...)[1:3, :]', linetype = :steppost)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
x_anim = [[x̄[1] for i = 1:100]..., x̄..., [x̄[end] for i = 1:100]...]
visualize!(vis, model,
	x_anim,
	Δt = h, T_off = length(x_anim)-125)
obj_platform = joinpath(pwd(), "models/rocket/space_x_platform.obj")
mtl_platform = joinpath(pwd(), "models/rocket/space_x_platform.mtl")
ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)
#
# setobject!(vis["platform"],ctm_platform)
# settransform!(vis["platform"], compose(Translation(0.0,2.0,-0.6), LinearMap(0.75 * RotZ(pi)*RotX(pi/2))))
#
# settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
# 	LinearMap(RotZ(1.5 * π + 0.0 * π))))
# setvisible!(vis["/Grid"], false)
# setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 1.0)

# line_mat = LineBasicMaterial(color=color=RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=10.0)
# points = Vector{Point{3,Float64}}()
# for xt in x̄
# 	push!(points, Point(xt[1:3]...))
# end
# setobject!(vis[:traj], MeshCat.Line(points, line_mat))
#
# t = 35
#
# setobject!(vis["rocket2"]["starship"], ctm)
#
# settransform!(vis["rocket2"]["starship"],
# 	compose(Translation(0.0, 0.0, -model.length),
# 		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))
#
# body = Cylinder(Point3f0(0.0, 0.0, -1.25),
#   Point3f0(0.0, 0.0, 0.5),
#   convert(Float32, 0.125))
#
# setobject!(vis["rocket2"]["body"], body,
#   MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
#
# settransform!(vis["rocket2"],
# 	  compose(Translation(x̄[t][1:3]),
# 			LinearMap(MRP(x̄[t][4:6]...) * RotX(0.0))))
#
# t = 85
#
# setobject!(vis["rocket3"]["starship"], ctm)
#
# settransform!(vis["rocket3"]["starship"],
# 	compose(Translation(0.0, 0.0, -model.length),
# 		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))
#
# body = Cylinder(Point3f0(0.0, 0.0, -1.25),
#   Point3f0(0.0, 0.0, 0.5),
#   convert(Float32, 0.125))
#
# setobject!(vis["rocket3"]["body"], body,
#   MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
#
# settransform!(vis["rocket3"],
# 	  compose(Translation(x̄[t][1:3]),
# 			LinearMap(MRP(x̄[t][4:6]...) * RotX(0.0))))
#
#
# t = T
#
# setobject!(vis["rocket4"]["starship"], ctm)
#
# settransform!(vis["rocket4"]["starship"],
# 	compose(Translation(0.0, 0.0, -model.length),
# 		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))
#
# body = Cylinder(Point3f0(0.0, 0.0, -1.25),
#   Point3f0(0.0, 0.0, 0.5),
#   convert(Float32, 0.125))
#
# setobject!(vis["rocket4"]["body"], body,
#   MeshPhongMaterial(color = RGBA(1.0, 0.0, 0.0, 1.0)))
#
# settransform!(vis["rocket4"],
# 	  compose(Translation(x̄[t][1:3]),
# 			LinearMap(MRP(x̄[t][4:6]...) * RotX(0.0))))
#
#
# # # Visualize
# # obj_rocket = joinpath(pwd(), "models/starship/Starship.obj")
# # mtl_rocket = joinpath(pwd(), "models/starship/Starship.mtl")
# # ctm = ModifiedMeshFileObject(obj_rocket, mtl_rocket, scale=1.0)
# # setobject!(vis["rocket"]["starship"], ctm)
# #
# # settransform!(vis["rocket"]["starship"],
# # 	compose(Translation(0.0, 0.0, -model.length),
# # 		LinearMap(0.25 * RotY(0.0) * RotZ(0.5 * π) * RotX(0.5 * π))))
# #
# # default_background!(vis)
# # settransform!(vis["rocket"],
# # 	compose(Translation(0.0, 0.0, 0.0),
# # 	LinearMap(RotY(0.0))))
#
#
# using PGFPlots
# const PGF = PGFPlots
#
# plt_F1_smooth = PGF.Plots.Linear(t, hcat(ū_nominal..., ū_nominal[end])[3,:],
# 	mark="none",style="const plot, color=cyan, line width = 2pt, dashed",legendentry="F1")
#
# plt_F2_smooth = PGF.Plots.Linear(t, hcat(ū_nominal..., ū_nominal[end])[1,:],
# 	mark="none",style="const plot, color=orange, line width = 2pt, dashed",legendentry="F2")
#
# plt_F3_smooth = PGF.Plots.Linear(t, hcat(ū_nominal..., ū_nominal[end])[2,:],
# 	mark="none",style="const plot, color=magenta, line width = 2pt, dashed",legendentry="F2")
#
# plt_F1_soc = PGF.Plots.Linear(t, hcat(ū_soc..., ū_soc[end])[3,:],
# 	mark="none",style="const plot, color=cyan, line width = 2pt",legendentry="F1 (soc)")
#
# plt_F2_soc = PGF.Plots.Linear(t, hcat(ū_soc..., ū_soc[end])[1,:],
# 	mark="none",style="const plot, color=orange, line width = 2pt",legendentry="F2 (soc)")
#
# plt_F3_soc = PGF.Plots.Linear(t, hcat(ū_soc..., ū_soc[end])[2,:],
# 	mark="none",style="const plot, color=magenta, line width = 2pt",legendentry="F2 (soc)")
#
# a = Axis([plt_F1_soc; plt_F2_soc; plt_F3_soc; plt_F1_smooth; plt_F2_smooth; plt_F3_smooth],
#     axisEqualImage=false,
#     hideAxis=false,
# 	ylabel="control",
# 	xlabel="time (s)",
# 	# xlims=(0.0, 2.0),
# 	legendStyle="{at={(0.5,0.5)},anchor=west}")
#
# PGF.save("/home/taylor/Research/implicit_dynamics_manuscript/figures/rocket_control.tikz", a, include_preamble=false)
