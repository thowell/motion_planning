using Plots
Random.seed!(0)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/quadruped/model.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/quadruped/gait.jl"))


# Initial conditions, controls, disturbances
x1 = [q_body; v_body_ref[1:3]; zeros(3)]
xT = [qT_body; v_body_ref[1:3]; zeros(3)]
u_gravity = [0.0; 0.0; model.mass * 9.81 / 4.0]
u_stand = [vcat([u_gravity for i = 1:4]...) for t = 1:T2-1]
# w_stand = [[pf1_ref[1]; pf2_ref[1]; pf3_ref[1]; pf4_ref[1]; 1; 1; 1; 1] for t = 1:T-1]
w_stand = [[pf1_ref_2[t]; pf2_ref_2[t]; pf3_ref_2[t]; pf4_ref_2[t]; contact_modes_2[t]] for t = 1:T2-1]

# Rollout
x̄ = rollout(model, x1, u_stand, w_stand, h, T2)
visualize!(vis, model, x_ref,
    [pf1_ref_2[1] for t = 1:T2],
    [pf2_ref_2[1] for t = 1:T2],
    [pf3_ref_2[1] for t = 1:T2],
    [pf4_ref_2[1] for t = 1:T2],
    Δt = h)

x_ref_2 = [[q_body_ref_2[t]; v_body_ref[1:3]; zeros(3)] for t = 1:T2]

visualize!(vis, model, x_ref_2,
    [pf1_ref_2[t] for t = 1:T2],
    [pf2_ref_2[t] for t = 1:T2],
    [pf3_ref_2[t] for t = 1:T2],
    [pf4_ref_2[t] for t = 1:T2],
    Δt = h)


# visualize!(vis, model, x̄, Δt = h)
# x̄ = linear_interpolation(x1, xT, T)
# plot(hcat(x̄...)')

# Objective
Q = [(t < T2 ? Diagonal([10.0; 10.0; 10.0; 1.0e-1 * ones(3); 1.0 * ones(3); 1.0e-1 * ones(3)])
        : Diagonal([10.0; 10.0; 10.0; 1.0e-1 * ones(3); 1.0 * ones(3); 1.0e-1 * ones(3)])) for t = 1:T2]
q = [-2.0 * Q[t] * x_ref_2[t] for t = 1:T2]

R = [0.01 * Diagonal([1.0; 1.0; 1.0e-1; 1.0; 1.0; 1.0e-1; 1.0; 1.0; 1.0e-1; 1.0; 1.0; 1.0e-1]) for t = 1:T2-1]
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

# Constraints
p = [t < T2 ? 0 : 0 for t = 1:T2]
info_t = Dict()#:ul => ul, :uu => uu, :inequality => (1:2 * m + 1))
info_T = Dict()#:xT => xT, :inequality => (1:4))
con_set = [StageConstraint(p[t], t < T2 ? info_t : info_T) for t = 1:T2]
# idx_T = collect([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		# ul = cons.con[t].info[:ul]
		# uu = cons.con[t].info[:uu]
		# # c[1:2 * m] .= [ul - u; u - uu]
		# c[2 * m + 1] = model.length - x[3]
	elseif t == T
		# c[1] = x_con[1] - x[1]
		# c[2] = x[1] - x_con[2]
		# c[3] = y_con[1] - x[2]
		# c[4] = x[2] - y_con[2]
		# xT = cons.con[T].info[:xT]
		# c[4 .+ (1:(n - 2))] .= (x - xT)[idx_T]
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(u_stand), copy(w_stand), h, T2,
	analytical_dynamics_derivatives = true)

# Solve
@time constrained_ddp_solve!(prob,
    linesearch = :armijo,
    max_iter = 100, max_al_iter = 1,
	con_tol = 0.1,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

visualize!(vis, model, x̄,
    [pf1_ref_2[t] for t = 1:T2],
    [pf2_ref_2[t] for t = 1:T2],
    [pf3_ref_2[t] for t = 1:T2],
    [pf4_ref_2[t] for t = 1:T2],
    q_body_ref = x_ref_2,
    Δt = h)

p1 = plot(tr2, hcat([[soc_projection(u[1:3]) for u in ū]..., soc_projection(ū[end][1:3])]...)', linetype = :steppost, labels = "")
p2 = plot(tr2, hcat([[soc_projection(u[3 .+ (1:3)]) for u in ū]..., soc_projection(ū[end][3 .+ (1:3)])]...)', linetype = :steppost, labels = "")
p3 = plot(tr2, hcat([[soc_projection(u[6 .+ (1:3)]) for u in ū]..., soc_projection(ū[end][6 .+ (1:3)])]...)', linetype = :steppost, labels = "")
p4 = plot(tr2, hcat([[soc_projection(u[9 .+ (1:3)]) for u in ū]..., soc_projection(ū[end][9 .+ (1:3)])]...)', linetype = :steppost, labels = "")

plot(p1, p2, p3, p4, layout = (4, 1))
