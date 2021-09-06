using Plots
Random.seed!(0)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/quadruped/model.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/quadruped/gait.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/quadruped/visuals.jl"))

# Horizon
T = 51
T2 = 2 * T - 1
Tm = 26

# Time step
tf = 1.0
tf2 = 2 * tf
h = tf2 / (T - 1)

# Generate gait
l_torso = 0.367
l_thigh = 0.2
l_leg = 0.2
w_torso = 0.267

θ1 = 0.2 * π
θ2 = θ1 + 0.25 * π
zh = l_thigh * cos(θ1) + l_leg * cos(θ2 - θ1)
θ3 = 0.275 * π
θ4 = acos((zh - l_thigh * cos(θ3)) / l_leg) + θ3

q_body_ref, pf1_ref, pf2_ref, pf3_ref, pf4_ref, contact_modes = generate_gait(θ1, θ2, θ3, θ4, T, Tm, h,
    l_torso = l_torso, l_thigh = l_thigh, l_leg = l_leg, w_torso = w_torso)
# v_body_ref = zero((q_body_ref[end] - q_body_ref[1]) / tf2)
v_body_ref = (q_body_ref[end] - q_body_ref[1]) / tf2

# Initial conditions, controls, disturbances
x1 = [q_body_ref[1]; v_body_ref[1:3]; zeros(3)]
xT = [q_body_ref[end]; v_body_ref[1:3]; zeros(3)]
u_gravity = [0.0; 0.0; model.mass * 9.81 / 4.0]
u_stand = [vcat([u_gravity for i = 1:4]...) for t = 1:T2-1]
# w_stand = [[pf1_ref[1]; pf2_ref[1]; pf3_ref[1]; pf4_ref[1]; 1; 1; 1; 1] for t = 1:T2-1]
w_stand = [[pf1_ref[t]; pf2_ref[t]; pf3_ref[t]; pf4_ref[t]; contact_modes[t]] for t = 1:T2-1]

# Rollout
x̄ = rollout(model, x1, u_stand, w_stand, h, T2)
visualize!(vis, model, x̄,
    [pf1_ref[t] for t = 1:T2],
    [pf2_ref[t] for t = 1:T2],
    [pf3_ref[t] for t = 1:T2],
    [pf4_ref[t] for t = 1:T2],
    Δt = h)

x_ref_2 = [[q_body_ref[t]; v_body_ref[1:3]; zeros(3)] for t = 1:T2]

visualize!(vis, model, x_ref_2,
    [pf1_ref[t] for t = 1:T2],
    [pf2_ref[t] for t = 1:T2],
    [pf3_ref[t] for t = 1:T2],
    [pf4_ref[t] for t = 1:T2],
    Δt = h)


# visualize!(vis, model, x̄, Δt = h)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = [(t < T2 ? Diagonal([10.0; 10.0; 10.0; 1.0e-1 * ones(3); 1.0 * ones(3); 1.0e-1 * ones(3)])
        : Diagonal([10.0; 10.0; 10.0; 1.0e-1 * ones(3); 1.0 * ones(3); 1.0e-1 * ones(3)])) for t = 1:T2]
q = [-2.0 * Q[t] * x_ref_2[t] for t = 1:T2]

R = [0.1 * Diagonal([1.0; 1.0; 1.0e-2; 1.0; 1.0; 1.0e-2; 1.0; 1.0; 1.0e-2; 1.0; 1.0; 1.0e-2]) for t = 1:T2-1]
r = [-2.0 * R[t] * zeros(model.m)  for t = 1:T2-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T2 ? R[t] : nothing, t < T2 ? r[t] : nothing) for t = 1:T2], T2)

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

prob = problem_data(model, obj, copy(x̄),
    copy(u_stand),
    # [0.001 * randn(model.m) for t = 1:T2-1],
    copy(w_stand),
    h,
    T2,
	analytical_dynamics_derivatives = true)

# Solve
@time ddp_solve!(prob,
    verbose = true,
    linesearch = :armijo,
    grad_tol = 1.0e-3,
    max_iter = 1000)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

vis = Visualizer()
render(vis)
visualize!(vis, model, x̄,
    [pf1_ref[t] for t = 1:T2],
    [pf2_ref[t] for t = 1:T2],
    [pf3_ref[t] for t = 1:T2],
    [pf4_ref[t] for t = 1:T2],
    q_body_ref = x_ref_2,
    Δt = h)

default_background!(vis)

cone_vis_scale = 0.35

for i = 1:4
    for t in [1, Tm, T, T + Tm, T2]
        foot_vis!(vis, i, t, contact_modes, tl = 0.5, r = 0.05)
    end
    for t in [1, Tm, T, T + Tm, T2-1]
        if t == T2-1
            uit = u[T2-1][(i - 1) * 3 .+ (1:3)]

            cone_vis!(vis, i, t, contact_modes, n = 25, tl = 0.1,
                h = cone_vis_scale * (uit[3] / f_max))#f_max * μ_friction * cone_vis_scale)
            force_vis!(vis, u, i, T2-1,
                f_max = f_max, μ_friction = μ_friction, cone_vis_scale = cone_vis_scale)
        end
    end
end
offset = 0.025
settransform!(vis[:world], Translation(x̄[end-1][1], 0.0, x̄[end-1][3] + offset))
default_background!(vis)

plot(tr2, hcat(ū..., ū[end])', label = "")
tr2 = range(0, stop = tf2, length = T2)
p1 = plot(tr2, hcat([[soc_projection(u[1:3], f_min, f_max, μ_friction) for u in ū]..., soc_projection(ū[end][1:3], f_min, f_max, μ_friction)]...)', linetype = :steppost, labels = "")
p2 = plot(tr2, hcat([[soc_projection(u[3 .+ (1:3)], f_min, f_max, μ_friction) for u in ū]..., soc_projection(ū[end][3 .+ (1:3)], f_min, f_max, μ_friction)]...)', linetype = :steppost, labels = "")
p3 = plot(tr2, hcat([[soc_projection(u[6 .+ (1:3)], f_min, f_max, μ_friction) for u in ū]..., soc_projection(ū[end][6 .+ (1:3)], f_min, f_max, μ_friction)]...)', linetype = :steppost, labels = "")
p4 = plot(tr2, hcat([[soc_projection(u[9 .+ (1:3)], f_min, f_max, μ_friction) for u in ū]..., soc_projection(ū[end][9 .+ (1:3)], f_min, f_max, μ_friction)]...)', linetype = :steppost, labels = "")

plot(p1, p2, p3, p4, layout = (4, 1))
