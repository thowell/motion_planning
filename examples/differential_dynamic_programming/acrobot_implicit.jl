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
include(joinpath(contact_control_path, "simulation/environments/flat.jl"))

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
include(joinpath(contact_control_path, "dynamics/double_pendulum/model.jl"))
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
#
# # Visuals
# using MeshCatMechanisms
# include(joinpath(contact_control_path, "dynamics/visuals.jl"))
# include(joinpath(contact_control_path, "dynamics/visual_utils.jl"))

# s = get_simulation("hopper_2D", "flat_2D_lc", "flat")

nq = s.model.dim.q
m = s.model.dim.u

T = 51
h = 0.1

q0 = [0.0; 0.0]
q1 = [0.0; 0.0]
qT = [π; 0.0]
q_ref = [π; 0.0]

x1 = [q0; q1]
xT = [qT; qT]

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

	z = zeros(nz)
	θ = zeros(nθ)

	ip_dyn = interior_point(z, θ,
		idx_ineq = idx_ineq,
		r! = s.res.r!,
		rz! = s.res.rz!,
		rθ! = s.res.rθ!,
		rz = s.rz,
		rθ = s.rθ,
		opts = dyn_opts)

	ip_dyn.opts.diff_sol = false

	ip_jac = interior_point(z, θ,
		idx_ineq = idx_ineq,
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

	ip.z .= copy([q1; 0.1 * ones(nc + nc)])
	ip.θ .= copy([q0; q1; u1; h])

	status = interior_point_solve!(ip)

	!status && (@warn "dynamics failure (res norm: $(norm(ip.r, Inf)))")
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

struct DoublePendulumI{I, T} <: Model{I, T}
    n::Int
    m::Int
    d::Int
	dynamics::Dynamics
end

model = DoublePendulumI{Midpoint, FixedTime}(2 * s.model.dim.q, s.model.dim.u, 0, d)

function fd(model::DoublePendulumI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]

	q2 = f(model.dynamics, q0, q1, u)

	return [q1; q2]
end

fd(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)

function fdx(model::DoublePendulumI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2dx1 = fx1(model.dynamics, q0, q1, u)

	return [zeros(nq, nq) I; dq2dx1]
end

fdx(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)


function fdu(model::DoublePendulumI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2du1 = fu1(model.dynamics, q0, q1, u)
	return [zeros(nq, model.m); dq2du1]
end

fdu(model, [q0; q1], zeros(model.m), zeros(model.d), h, 1)

n = model.n
m = model.m

ū = [1.0e-2 * randn(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
V = Diagonal(ones(s.model.dim.q))
_Q = [V -V; -V V] ./ h^2.0
Q = [t < T ? _Q : _Q for t = 1:T]
q = [-2.0 * Q[t] * zeros(n) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]
r = [zeros(model.m) for t = 1:T-1]

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
ul = [-2.0]
uu = [2.0]
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
		c .= [ul - u; u - uu]
	else
		c .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 0.005)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)
q2l = -0.5 * π * ones(length(q̄))
q2u = 0.5 * π * ones(length(q̄))
t = range(0, stop = h * (length(q̄) - 1), length = length(q̄))
plt = plot();
plt = plot!(t, q2l, color = :black, width = 2.0, label = "q2 limit lower")
plt = plot!(t, q2u, color = :black, width = 2.0, label = "q2 limit upper")
plt = plot!(t, hcat(q̄...)', width = 2.0,
	color = [:magenta :orange],
	labels = ["q1" "q2"],
	legend = :topleft,
	xlabel = "time (s)",
	ylabel = "configuration",
	title = "acrobot (w/o joint limits)")

	# title = "acrobot (w/ joint limits)")

# show(plt)
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/acrobot_joint_limits.png")
# savefig(plt, "/home/taylor/Research/implicit_dynamics_manuscript/figures/acrobot_no_joint_limits.png")

plot(hcat(ū..., ū[end])', linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
open(vis)
default_background!(vis)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 30)

visualize!(vis, s.model, q̄, Δt = h)

# visualization
function visualize!(vis, model::DoublePendulum, x;
        color=RGBA(0.0, 0.0, 0.0, 1.0),
        r = 0.1, Δt = 0.1)

    i = 1
    l1 = Cylinder(Point3f0(0.0, 0.0, 0.0), Point3f0(0.0, 0.0, model.l1),
        convert(Float32, 0.025))
    setobject!(vis["l1"], l1, MeshPhongMaterial(color = color))
    l2 = Cylinder(Point3f0(0.0,0.0,0.0), Point3f0(0.0, 0.0, model.l2),
        convert(Float32, 0.025))
    setobject!(vis["l2"], l2, MeshPhongMaterial(color = color))

    setobject!(vis["elbow_nominal"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))
	setobject!(vis["elbow_limit"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 1.0, 0.0, 1.0)))
    setobject!(vis["ee"], Sphere(Point3f0(0.0),
        convert(Float32, 0.05)),
        MeshPhongMaterial(color = RGBA(0.0, 0.0, 0.0, 1.0)))

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	ϵ = 1.0e-1
    T = length(x)
    for t = 1:T

        MeshCat.atframe(anim,t) do
            p_mid = [kinematics_elbow(model, x[t])[1], 0.0, kinematics_elbow(model, x[t])[2]]
            p_ee = [kinematics(model, x[t])[1], 0.0, kinematics(model, x[t])[2]]

            settransform!(vis["l1"], cable_transform(zeros(3), p_mid))
            settransform!(vis["l2"], cable_transform(p_mid, p_ee))

            settransform!(vis["elbow_nominal"], Translation(p_mid))
			settransform!(vis["elbow_limit"], Translation(p_mid))
            settransform!(vis["ee"], Translation(p_ee))

			if x[t][2] <= -0.5 * π + ϵ || x[t][2] >= 0.5 * π - ϵ
				setvisible!(vis["elbow_nominal"], false)
				setvisible!(vis["elbow_limit"], true)
			else
				setvisible!(vis["elbow_nominal"], true)
				setvisible!(vis["elbow_limit"], false)
			end
        end
    end

    # settransform!(vis["/Cameras/default"],
    #    compose(Translation(0.0 , 0.0 , 0.0), LinearMap(RotZ(pi / 2.0))))

    MeshCat.setanimation!(vis, anim)
end
