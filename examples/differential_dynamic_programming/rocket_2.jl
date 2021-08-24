using Plots
include_ddp()

# Model
include_model("rocket3D")

function fd(model::Rocket3D{Midpoint, FixedTime}, x, u, w, h, t)
	return view(x, 1:model.n) + h * f(model, view(x, 1:model.n) + 0.5 * h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w)
end

n = model.n
m = model.m

# Time
T = 101
h = 0.05

# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[1] = 2.5
x1[2] = 2.5
x1[3] = 10.0
# mrp = MRP(RotY(-0.45 * π) * RotX(-0.25 * π))
mrp = MRP(RotZ(0.25 * π) * RotY(-0.45 * π))

x1[4:6] = [mrp.x; mrp.y; mrp.z]
x1[9] = -5.0

visualize!(vis, model, [x1], Δt = h)

# visualize!(vis, model, [x1], Δt = h)

xT = zeros(model.n)
# xT[1] = 2.5
# xT[2] = 0.0
xT[3] = model.length
mrpT = MRP(RotZ(0.25 * π) * RotY(0.0))
xT[4:6] = [mrpT.x; mrpT.y; mrpT.z]
visualize!(vis, model, [xT], Δt = h)


u_ref = [0.0; 0.0; 0.0]#model.mass * 9.81]
ū = [u_ref + [1.0e-2; 1.0e-2; 1.0e-2] .* randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = h * [(t < T ? 1.0 * Diagonal([1.0e-1 * ones(3); 0.0 * ones(3); 1.0e-1 * ones(3); 1000.0 * ones(3)])
        : 100.0 * Diagonal(1.0 * ones(model.n))) for t = 1:T]
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
info_t = Dict(:ul => [-5.0; -5.0; 0.0], :uu => [5.0; 5.0; 15.0], :inequality => (1:2 * m + 1))
info_T = Dict(:xT => xT, :inequality => (1:4))
con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]
idx_T = collect([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
model.length
function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c[1:2 * m] .= [ul - u; u - uu]
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

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

x̄_nominal = x̄
ū_nominal = ū
#
# @save "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/rocket.jld2" x̄_nominal ū_nominal
# @load "/home/taylor/Research/motion_planning/examples/differential_dynamic_programming/implicit_dynamics/rocket.jld2"

# Trajectories
plot(hcat(ū...)', linetype = :steppost)
plot(hcat(x̄...)[1:3, :]', linetype = :steppost)
plot(hcat(x̄...)[4:6, :]', linetype = :steppost)

# t = 1
# idx = [3; 1; 2]
# second_order_cone_projection(ū[t][idx])
# ū[t]
# # Visualize
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

setobject!(vis["platform"],ctm_platform)
settransform!(vis["platform"], compose(Translation(0.0,2.0,-0.6), LinearMap(0.75 * RotZ(pi)*RotX(pi/2))))

settransform!(vis["/Cameras/default"], compose(Translation(0.0, 0.0, 0.0),
	LinearMap(RotZ(1.5 * π + 0.0 * π))))
setvisible!(vis["/Grid"], false)
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 1.0)
