using Plots
using Random
Random.seed!(1)

include_ddp()

contact_control_path = "/home/taylor/Research/ContactControl.jl/src"

using Parameters
# Utilities
include(joinpath(contact_control_path, "utils.jl"))

# Solver
include(joinpath(contact_control_path, "solver/cones.jl"))
include(joinpath(contact_control_path, "solver/interior_point.jl"))
include(joinpath(contact_control_path, "solver/lu.jl"))

# Environment
include(joinpath(contact_control_path, "simulator/environment.jl"))

# Dynamics
include(joinpath(contact_control_path, "dynamics/model.jl"))

# Simulator
include(joinpath(contact_control_path, "simulation/contact_methods.jl"))
include(joinpath(contact_control_path, "simulation/simulation.jl"))
include(joinpath(contact_control_path, "simulator/trajectory.jl"))

include(joinpath(contact_control_path, "dynamics/code_gen_dynamics.jl"))
include(joinpath(contact_control_path, "dynamics/fast_methods_dynamics.jl"))

# Models
include(joinpath(contact_control_path, "dynamics/quaternions.jl"))
include(joinpath(contact_control_path, "dynamics/mrp.jl"))
include(joinpath(contact_control_path, "dynamics/euler.jl"))

# include("dynamics/particle_2D/model.jl")
# include("dynamics/particle/model.jl")
include(joinpath(contact_control_path, "dynamics/hopper_2D/model.jl"))
# include("dynamics/hopper_3D/model.jl")
# include("dynamics/hopper_3D_quaternion/model.jl")
# include("dynamics/quadruped/model.jl")
# include("dynamics/quadruped_simple/model.jl")
# include("dynamics/biped/model.jl")
# include("dynamics/flamingo/model.jl")
# include("dynamics/pushbot/model.jl")
# include("dynamics/planarpush/model.jl")
# include("dynamics/planarpush_2D/model.jl")
# include("dynamics/rigidbody/model.jl")
# include("dynamics/box/model.jl")

# Simulation
include(joinpath(contact_control_path, "simulation/environments/flat.jl"))
# include("simulation/environments/piecewise.jl")
# include("simulation/environments/quadratic.jl")
# include("simulation/environments/slope.jl")
# include("simulation/environments/sinusoidal.jl")
# include("simulation/environments/stairs.jl")

include(joinpath(contact_control_path, "simulation/residual_approx.jl"))
include(joinpath(contact_control_path, "simulation/code_gen_simulation.jl"))

# Visuals
using MeshCatMechanisms
include(joinpath(contact_control_path, "dynamics/visuals.jl"))
include(joinpath(contact_control_path, "dynamics/visual_utils.jl"))

s = get_simulation("hopper_2D", "flat_2D_lc", "flat")

nq = s.model.dim.q
m = s.model.dim.u

T = 11
h = 0.1

q0 = [0.0; 0.5; 0.0; 0.5]
q1 = [0.0; 0.5; 0.0; 0.5]
qT = [0.0; 0.5; 0.0; 0.5]
q_ref = [0.0; 0.5; 0.0; 0.5]
u0 = zeros(m)

struct Dynamics{T}
	s::Simulation
	ip_dyn::InteriorPoint
	ip_jac::InteriorPoint
	h::T
end

function gen_dynamics(s::Simulation, h;
		dyn_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true),
		jac_opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 0.1,
						diff_sol = true))

	z = zeros(num_var(s.model, s.env))
	θ = zeros(num_data(s.model))

	ip_dyn = interior_point(z, θ,
		idx_ineq = inequality_indices(s.model, s.env),
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = dyn_opts)

	ip_dyn.opts.diff_sol = false

	ip_jac = interior_point(z, θ,
		idx_ineq = inequality_indices(s.model, s.env),
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = jac_opts)

	ip_jac.opts.diff_sol = true

	Dynamics(s, ip_dyn, ip_jac, h)
end

d = gen_dynamics(s, h,
	dyn_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-4, κ_init = 0.1),
	jac_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-2, κ_init = 0.1))

function f!(d::Dynamics, q0, q1, u1, mode = :dynamics)
	s = d.s
	ip = (mode == :dynamics ? d.ip_dyn : d.ip_jac)
	h = d.h

	z_initialize!(ip.z, s.model, s.env, copy(q1))
	θ_initialize!(ip.θ, s.model, copy(q0), copy(q1), copy(u1), zeros(s.model.dim.w), s.model.μ_world, h)

	status = interior_point_solve!(ip)

	!status && (@warn "dynamics failure")
end

function f(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :dynamics)
	return copy(d.ip_dyn.z[1:d.s.model.dim.q])
end

f(d, q0, q1, zeros(m))

function fq0(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 1:d.s.model.dim.q])
end

fq0(d, q0, q1, zeros(m))

function fq1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, d.s.model.dim.q .+ (1:d.s.model.dim.q)])
end

fq1(d, q0, q1, zeros(m))

function fx1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 1:(2 * d.s.model.dim.q)])
end

fx1(d, q0, q1, zeros(m))

function fu1(d::Dynamics, q0, q1, u1)
	f!(d, q0, q1, u1, :jacobian)
	return copy(d.ip_jac.δz[1:d.s.model.dim.q, 2 * d.s.model.dim.q .+ (1:d.s.model.dim.u)])
end

fu1(d, q0, q1, zeros(m))

struct HopperCI{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::Dynamics
end

model = HopperCI{Midpoint, FixedTime}(2 * s.model.dim.q, s.model.dim.u, 0, d)

np = model.n
nq = model.dynamics.s.model.dim.q
function fd(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	nu = model.dynamics.s.model.dim.u

	if t == 1
		θ = u[nu .+ (1:np)]
		q0 = θ[1:nq]
		q1 = θ[nq .+ (1:nq)]
	else
		θ = x[2 * nq .+ (1:np)]
		q0 = x[1:nq]
		q1 = x[nq .+ (1:nq)]
	end

	u1 = u[1:nu]

	q2 = f(model.dynamics, q0, q1, u1)

	return [q1; q2; θ]
end

x0 = [q0; q1]
p0 = [q0; q1]
fd(model, x0, [u0; p0], zeros(0), h, 1)
fd(model, [x0; p0], u0, zeros(0), h, 2)

function fdx(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	nu = model.dynamics.s.model.dim.u

	if t == 1
		θ = u[nu .+ (1:np)]
		q0 = θ[1:nq]
		q1 = θ[nq .+ (1:nq)]
	else
		θ = x[2 * nq .+ (1:np)]
		q0 = x[1:nq]
		q1 = x[nq .+ (1:nq)]
	end

	u1 = u[1:nu]

	dq2dx1 = fx1(model.dynamics, q0, q1, u1)

	if t == 1
		return zeros(4 * nq, 2 * nq)
	else
		return [zeros(nq, nq) I zeros(nq, 2 * nq);
				dq2dx1 zeros(nq, 2 * nq);
				zeros(2 * nq, 2 * nq) I]
	end
end

fdx(model, x0, [u0; p0], zeros(0), h, 1)
fdx(model, [x0; p0], u0, zeros(0), h, 2)

function fdu(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	nu = model.dynamics.s.model.dim.u

	if t == 1
		θ = u[nu .+ (1:np)]
		q0 = θ[1:nq]
		q1 = θ[nq .+ (1:nq)]
	else
		θ = x[2 * nq .+ (1:np)]
		q0 = x[1:nq]
		q1 = x[nq .+ (1:nq)]
	end

	u1 = u[1:nu]

	dq2dx1 = fx1(model.dynamics, q0, q1, u1)
	dq2du1 = fu1(model.dynamics, q0, q1, u1)

	if t == 1
		return [zeros(nq, nu + nq) I;
		        dq2du1 dq2dx1;
				zeros(2 * nq, nu) I]
	else
		return [zeros(nq, nu);
				dq2du1;
				zeros(2 * nq, nu)]
	end

	return [zeros(nq, model.m); dq2du1]
end

fdu(model, x0, [u0; p0], zeros(0), h, 1)
fdu(model, [x0; p0], u0, zeros(0), h, 2)


# Time
T = 21
h = 0.05

n = [t == 1 ? model.n : 2 * model.n for t = 1:T]
m = [t == 1 ? model.m + model.n : model.m for t = 1:T-1]


# Initial conditions, controls, disturbances
q0 = [0.0; 0.5; 0.0; 0.5]
q1 = [0.0; 0.5; 0.0; 0.5]
qM = [0.5; 0.5; 0.0; 0.5]
qT = [1.0; 0.5; 0.0; 0.5]
q_ref = [0.5; 0.75; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

ū = [t == 1 ? [0.0; s.model.g * (s.model.mb + s.model.ml) * 0.5 * h; x1] : [0.0; s.model.g * (s.model.mb + s.model.ml) * 0.5 * h] for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t == 1 ? 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(model.n)])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(model.n)])) for t = 1:T]
q = [t == 1 ? -2.0 * Q[t] * (t < 6 ? x_ref : x_ref) : -2.0 * Q[t] * [(t < 6 ? x_ref : x_ref); zeros(model.n)] for t = 1:T]
R = [t == 1 ? Diagonal([1.0e-3 * ones(model.m); 1.0e-1 * ones(nq); 1.0e-5 * ones(nq)]) : Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1]
r = [t == 1 ? [zeros(model.m); -2.0 * R[t][1:nq, 1:nq] * x1[1:nq]; zeros(nq)] : zeros(model.m) for t = 1:T-1]

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
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
p = [t < T ? (t == 1 ? 2 * model.m + nq + 2 * 2 : (t == 6 ? 2 * model.m + 0 * model.n : 2 * model.m)) : (model.n - 2) + 2 for t = 1:T]
info_1 = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model.m))
# info_M = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model.m))
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model.m))
info_T = Dict(:xT => xT, :inequality => (1:2))
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : (t == -1 ? info_M : info_t)) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		u1 = u[1:model.m]
		c[1:(2 * model.m)] .= [ul - u1; u1 - uu]

		if t == 1
			# c[2 * model.m .+ (1:model.n)] .= u[model.m .+ (1:model.n)] - x1
			c[2 * model.m .+ (1:nq)] .= u[model.m .+ (1:nq)] - x1[1:nq]
			c[2 * model.m + nq .+ (1:2)] = kinematics(model.dynamics.s.model, u[model.m .+ (1:nq)]) - kinematics(model.dynamics.s.model, x1[1:nq])
			c[2 * model.m + nq + 2 .+ (1:2)] = kinematics(model.dynamics.s.model, u[model.m + nq .+ (1:nq)]) - kinematics(model.dynamics.s.model, x1[nq .+ (1:nq)])
		end

		# if t == 6
		# 	c[2 * model.m .+ (1:model.n)] .= x[1:model.n] - xM
		# end
	end
	if t == T
		x_travel = 0.5
		θ = x[model.n .+ (1:model.n)]

		c[1] = x_travel - (x[1] - θ[1])
		c[2] = x_travel - (x[nq + 1] - θ[nq + 1])
		# c[2 .+ (1:model.n)] .= x[1:model.n] - cons.con[T].info[:xT]
		c[2 .+ (1:3)] = x[1:nq][collect([2, 3, 4])] - θ[1:nq][collect([2, 3, 4])]
		c[2 + 3 .+ (1:3)] = x[nq .+ (1:nq)][collect([2, 3, 4])] - θ[nq .+ (1:nq)][collect([2, 3, 4])]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
 	n = n, m = m,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
q̄[1] = ū[1][model.m .+ (1:nq)]
q̄[2] = ū[1][model.m + nq .+ (1:nq)]
vis = Visualizer()
render(vis)
# visualize!(vis, s.model, q̄, Δt = h)

 # x̄[T][1:model.n] - ū[1][model.m .+ (1:model.n)]

function mirror_gait(q, T; n = 5)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]

	stride = zero(qm[1])
	strd = q[T+1][1] - q[2][1]
	@show stride[1] += strd
	@show 0.5 * stride

	for i = 1:n-1
		for t = 1:T-1
			push!(qm, q[t+2] + stride)
			push!(um, u[t])
		end
		stride[1] += strd
	end
	len = qm[end][1]

	# center
	for t = 1:length(qm)
		qm[t][1] -= 0.5 * len
	end

	return qm, um
end

qm, um = mirror_gait(q̄, T)
visualize!(vis, s.model, qm, Δt = h)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)

body_points = Vector{Point{3,Float64}}()
foot_points = Vector{Point{3,Float64}}()
body_points_opt = Vector{Point{3,Float64}}()
foot_points_opt = Vector{Point{3,Float64}}()
for q in qm
	push!(body_points, Point(q[1], -0.025, q[2]))
	push!(body_points_opt, Point(q[1], -0.05, q[2]))

	k = kinematics(s.model, q)
	push!(foot_points, Point(k[1], -0.025, k[2]))
	push!(foot_points_opt, Point(k[1], -0.05, k[2]))

end
line_opt_mat = LineBasicMaterial(color=color=RGBA(1.0, 0.0, 0.0, 1.0), linewidth=10.0)
line_mat = LineBasicMaterial(color=color=RGBA(0.0, 0.0, 0.0, 1.0), linewidth=5.0)

setobject!(vis[:body_traj], MeshCat.Line(body_points, line_mat))
setobject!(vis[:foot_traj], MeshCat.Line(foot_points, line_mat))
setobject!(vis[:body_traj_opt], MeshCat.Line(body_points_opt[1:T+1], line_opt_mat))
setobject!(vis[:foot_traj_opt], MeshCat.Line(foot_points_opt[1:T+1], line_opt_mat))

open(vis)
function visualize!(vis, model, q;
		Δt = 0.1, scenario = :vertical)

    r_foot = 0.05
    r_leg = 0.5 * r_foot

	default_background!(vis)

    setobject!(vis["body"], Sphere(Point3f0(0),
        convert(Float32, 0.1)),
        MeshPhongMaterial(color = RGBA(0, 1, 0, 1.0)))

    setobject!(vis["foot"], Sphere(Point3f0(0),
        convert(Float32, r_foot)),
        MeshPhongMaterial(color = RGBA(1.0, 165.0 / 255.0, 0, 1.0)))

    n_leg = 100
    for i = 1:n_leg
        setobject!(vis["leg$i"], Sphere(Point3f0(0),
            convert(Float32, r_leg)),
            MeshPhongMaterial(color = RGBA(0, 0, 0, 1.0)))
    end

    p_leg = [zeros(3) for i = 1:n_leg]
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        p_body = [q[t][1], 0.0, q[t][2]]
        p_foot = [kinematics(model, q[t])[1], 0.0, kinematics(model, q[t])[2]]

        q_tmp = Array(copy(q[t]))
        r_range = range(0, stop = q[t][4], length = n_leg)
        for i = 1:n_leg
            q_tmp[4] = r_range[i]
            p_leg[i] = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]
        end
        q_tmp[4] = q[t][4]
        p_foot = [kinematics(model, q_tmp)[1], 0.0, kinematics(model, q_tmp)[2]]

        z_shift = [0.0; 0.0; r_foot]

        MeshCat.atframe(anim, t) do
            settransform!(vis["body"], Translation(p_body + z_shift))
            settransform!(vis["foot"], Translation(p_foot + z_shift))

            for i = 1:n_leg
                settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
            end
        end
    end

	if scenario == :vertical
		settransform!(vis["/Cameras/default"],
			compose(Translation(0.0, 0.5, -1.0),LinearMap(RotZ(-pi / 2.0))))
	end

    MeshCat.setanimation!(vis, anim)
end
