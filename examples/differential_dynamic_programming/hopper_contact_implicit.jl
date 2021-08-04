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

n = s.model.dim.q
m = s.model.dim.u

T = 11
h = 0.1

q0 = [0.0; 0.5; 0.0; 0.5]
q1 = [0.0; 0.5; 0.0; 0.5]
qT = [0.0; 0.5; 0.0; 0.5]
q_ref = [0.0; 0.5; 0.0; 0.5]

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
	jac_opts = InteriorPointOptions{Float64}(κ_tol = 1.0e-4, κ_init = 0.1))

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

function fd(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]

	q2 = f(model.dynamics, q0, q1, u)

	return [q1; q2]
end

fd(model, x1, ū[1], w[1], h, 1)

function fdx(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2dx1 = fx1(model.dynamics, q0, q1, u)

	return [zeros(nq, nq) I; dq2dx1]
end

fdx(model, x1, ū[1], w[1], h, 1)


function fdu(model::HopperCI{Midpoint, FixedTime}, x, u, w, h, t)
	nq = model.dynamics.s.model.dim.q
	q0 = x[1:nq]
	q1 = x[nq .+ (1:nq)]
	dq2du1 = fu1(model.dynamics, q0, q1, u)
	return [zeros(nq, model.m); dq2du1]
end

fdu(model, x1, ū[1], w[1], h, 1)

n = model.n
m = model.m

# Time
T = 11
h = 0.1

# Initial conditions, controls, disturbances
q0 = [0.0; 0.5; 0.0; 0.5]
q1 = [0.0; 0.5; 0.0; 0.5]
qT = [1.0; 0.5; 0.0; 0.5]
q_ref = [0.0; 0.5; 0.0; 0.5]

x1 = [q1; q1]
xT = [qT; qT]

ū = [[0.0; s.model.g * (s.model.mb + s.model.ml) * 0.5 * h] for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [t < T ? 0.1 * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) : Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]
R = [Diagonal(1.0e-3 * ones(model.m)) for t = 1:T-1]
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
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
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
	max_iter = 1000, max_al_iter = 5,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

q̄ = state_to_configuration(x̄)

vis = Visualizer()
render(vis)
visualize!(vis, s.model, q̄, Δt = h)

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



# # Model
# include_model("double_integrator")
#
# function f(model::DoubleIntegratorContinuous, x, u, w)
#     [x[2]; (1.0 + w[1]) * u[1]]
# end
#
# function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
# 	x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
# end
#
# model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
# n = model.n
# m = model.m
#
# # Time
# T = 11
# h = 0.1
#
# # Initial conditions, controls, disturbances
# x1 = [1.0; 0.0]
# x_ref = [[0.0; 0.0] for t = 1:T]
# xT = [0.0; 0.0]
# ū = [1.0 * randn(model.m) for t = 1:T-1]
# u_ref = [zeros(model.m) for t = 1:T-1]
# w = [zeros(model.d) for t = 1:T-1]
#
# # Rollout
# x̄ = rollout(model, x1, ū, w, h, T)
#
# # Objective
# Q = [(t < T ? h : 1.0) * (t < T ?
# 	 Diagonal([1.0; 1.0])
# 		: Diagonal([1.0; 1.0])) for t = 1:T]
# q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
# R = h * [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
# r = [zeros(model.m) for t = 1:T-1]
#
# obj = StageCosts([QuadraticCost(Q[t], q[t],
# 	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)
#
# function g(obj::StageCosts, x, u, t)
# 	T = obj.T
#     if t < T
# 		Q = obj.cost[t].Q
# 		q = obj.cost[t].q
# 	    R = obj.cost[t].R
# 		r = obj.cost[t].r
#         return x' * Q * x + q' * x + u' * R * u + r' * u
#     elseif t == T
# 		Q = obj.cost[T].Q
# 		q = obj.cost[T].q
#         return x' * Q * x + q' * x
#     else
#         return 0.0
#     end
# end
#
# # Constraints
# ul = [-5.0]
# uu = [5.0]
# p = [t < T ? 2 * m : n for t = 1:T]
# info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * m))
# info_T = Dict(:xT => xT)
# con_set = [StageConstraint(p[t], t < T ? info_t : info_T) for t = 1:T]
#
# function c!(c, cons::StageConstraints, x, u, t)
# 	T = cons.T
# 	p = cons.con[t].p
#
# 	if t < T
# 		ul = cons.con[t].info[:ul]
# 		uu = cons.con[t].info[:uu]
# 		c .= [ul - u; u - uu]
# 	else
# 		c .= x - cons.con[T].info[:xT]
# 	end
# end
#
# prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T)
#
# # Solve
# @time constrained_ddp_solve!(prob,
# 	max_iter = 1000, max_al_iter = 10,
# 	ρ_init = 1.0, ρ_scale = 10.0,
# 	con_tol = 1.0e-5)
#
# x, u = current_trajectory(prob)
# x̄, ū = nominal_trajectory(prob)
#
# # Visualize
# using Plots
# # plot(hcat([[0.0; 0.0] for t = 1:T]...)',
# #     width = 2.0, color = :black, label = "")
# plt = plot(hcat(x...)',
# 	width = 2.0,
# 	color = [:cyan :orange],
# 	label = ["x" "ẋ"],
# 	xlabel = "time step",
# 	ylabel = "state")
#
# savefig(plt,
# 	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_base_state.png"))
#
# plt = plot(hcat(u..., u[end])',
# 	width = 2.0,
# 	color = :magenta,
# 	linetype = :steppost,
# 	xlabel = "time step",
# 	ylabel = "control",
# 	label = "")
#
# savefig(plt,
# 	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_base_control.png"))
#
# # Simulate policy
# include(joinpath(@__DIR__, "simulate.jl"))
#
# # Model
# model_sim = DoubleIntegratorContinuous{RK3, FixedTime}(model.n, model.m, model.d)
# x1_sim = copy(x1)
# T_sim = 10 * T
#
# # Time
# tf = h * (T - 1)
# t = range(0, stop = tf, length = T)
# t_sim = range(0, stop = tf, length = T_sim)
# dt_sim = tf / (T_sim - 1)
#
# # Policy
# K = [K for K in prob.p_data.K]
# plot(vcat(K...))
# K = [prob.p_data.K[t] for t = 1:T-1]
# # K, _ = tvlqr(model, x̄, ū, h, Q, R)
# # # K = [-k for k in K]
# # K = [-K[1] for t = 1:T-1]
# # plot(vcat(K...))
#
# # Simulate
# N_sim = 1
# x_sim = []
# u_sim = []
# J_sim = []
# Random.seed!(1)
# for k = 1:N_sim
# 	wi_sim = 0.0 * min(0.1, max(-0.1, 1.0e-1 * randn(1)[1]))
# 	w_sim = [wi_sim for t = 1:T-1]
# 	println("sim: $k - w = $(wi_sim[1])")
#
# 	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate_linear_feedback(
# 		model_sim,
# 		K,
# 	    x̄, ū,
# 		x_ref, u_ref,
# 		Q, R,
# 		T_sim, h,
# 		x1_sim,
# 		w_sim,
# 		ul = ul,
# 		uu = uu)
#
# 	push!(x_sim, x_ddp)
# 	push!(u_sim, u_ddp)
# 	push!(J_sim, J_ddp)
# end
#
# # Visualize
# idx = (1:2)
# plt = plot(t, hcat(x̄...)[idx, :]',
# 	width = 2.0, color = :black, label = "",
# 	xlabel = "time (s)", ylabel = "state",
# 	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")
#
# for xs in x_sim
# 	plt = plot!(t_sim, hcat(xs...)[idx, :]',
# 	    width = 1.0, color = :magenta, label = "")
# end
# display(plt)
#
# plt = plot(t, hcat(ū..., ū[end])',
# 	width = 2.0, color = :black, label = "",
# 	xlabel = "time (s)", ylabel = "control",
# 	linetype = :steppost,
# 	title = "double integrator (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")
#
# for us in u_sim
# 	plt = plot!(t_sim, hcat(us..., us[end])',
# 		width = 1.0, color = :magenta, label = "",
# 		linetype = :steppost)
# end
# display(plt)
# # u_sim
# # plot(vcat(K...))
