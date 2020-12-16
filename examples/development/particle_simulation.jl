# Model
include_model("particle")

# Horizon
T = 101

# Time step
tf = 1.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= Inf
_ul = zeros(model.m)
_ul[model.idx_u] .= -Inf
ul, uu = control_bounds(model, T, _ul, _uu)

# Circle
θ = range(0, stop = 2 * π, length = T)
x = range(0.0, stop = 1.0, length = T) #cos.(θ)
y = range(0.0, stop = 0.0, length = T) #sin.(θ)
# plot(x, y, ratio = :equal)

# Initial and final states
q_ref = [[x[t]; y[t]; 1.0] for t = 1:T]
x_ref = configuration_to_state(q_ref)

xl, xu = state_bounds(model, T,
    x1 = [q_ref[1]; q_ref[1]],
    xT = [q_ref[T]; q_ref[T]])

# Objective
# obj_penalty = PenaltyObjective(1.0e5, model.m)
obj_track = quadratic_tracking_objective(
    [Diagonal(1000.0 * ones(model.n)) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu); zeros(model.m - model.nu)]) for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [zeros(model.m) for t = 1:T])
obj = MultiObjective([obj_penalty, obj_track])

# Constraints
include_constraints("contact")
con_contact = contact_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact)

# Trajectory initialization
x0 = deepcopy(x_ref) #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄ = solve(prob, copy(z0), tol = 1.0e-4, c_tol = 1.0e-4, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

Q = [Diagonal(ones(model.n)) for t = 1:T]
R = [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
x_proj, u_proj = lqr_projection(model, x̄, ū, h, Q, R)
q_proj = state_to_configuration(x_proj)

using Plots
plot(hcat(ū...)[1:3, :]',
    linetype = :steppost,
    label = "",
    color = :red,
    width = 2.0)

plot!(hcat(u_proj...)[1:3, :]',
    linetype = :steppost,
    label = "", color = :black)

plot(hcat(state_to_configuration(x̄)...)',
    color = :red,
    width = 2.0,
	label = "")

plot!(hcat(state_to_configuration(x_proj)...)',
    color = :black,
	label = "")

q̄ = state_to_configuration(x̄)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

ee_positions = hcat(q_ref...)
points = collect(eachcol(ee_positions))
material = LineBasicMaterial(color=colorant"orange", linewidth=6.0)
setobject!(vis["ee path"], Object(PointCloud(points), material, "Line"))


# Simulator
function dynamics(model::Particle, v1, q1, q2, q3, u, λ, b, w, h)
	(M_func(model, q1) * v1
	- M_func(model, q2) * (SVector{3}(q3) - SVector{3}(q2)) / h
	+ h * (transpose(B_func(model, q3)) * SVector{3}(u)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{4}(b)
	- G_func(model, q2)))
end

function maximum_dissipation(model::Particle, q2, q3, ψ, η, h)
	ψ_stack = ψ[1] * ones(model.nb)
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

function no_slip(model::Particle, q2, q3, λ, h)
	return (λ' * _P_func(model, q3) * (q3 - q2) / h)[1]
end

function friction_cone(model::Particle, λ, b)
	return @SVector [model.μ * λ[1] - sum(b)]
end

mutable struct MOISimulator <: MOI.AbstractNLPEvaluator
    num_var::Int                 # number of decision variables
    num_con::Int                 # number of constraints
    primal_bounds
    constraint_bounds

	model
	slack_penalty
	v1
	q1
	q2
	u
	w
	h
end

function simulator_problem(model, v1, q1, q2, u, w, h; slack_penalty = 1.0e5)
	num_var = model.nq + model.nc + model.nb + 0 * model.nc + 0 *model.nb + model.ns
	num_con = model.nq + model.nc + 0 * model.nc + model.nb +  1 #3

	zl = zeros(num_var)
	zl[1:model.nq] .= -Inf
	zu = Inf * ones(num_var)

	cl = zeros(num_con)
	cu = zeros(num_con)
	cu[model.nq .+ (1:model.nc)] .= Inf
	cu[model.nq + model.nc .+ (1:model.nc)] .= Inf
	cu[model.nq + model.nc + model.nc + model.nb + 1] = Inf
	cu[model.nq + model.nc + model.nc + model.nb + 2] = Inf
	cu[model.nq + model.nc + model.nc + model.nb + 3] = Inf

	MOISimulator(num_var,
		num_con,
		(zl, zu),
		(cl, cu),
		model,
		slack_penalty,
		v1,
		q1,
		q2,
		u,
		w,
		h)
end
prob_sim = simulator_problem(model, ones(nq), ones(nq), ones(nq), zeros(nu), ones(nq), h)
z0 = rand(prob_sim.num_var)

function MOI.eval_objective(prob::MOISimulator, x)
    return prob.slack_penalty * x[prob.num_var]
end

MOI.eval_objective(prob_sim, z0)

function MOI.eval_objective_gradient(prob::MOISimulator, grad_f, x)
    grad_f .= 0.0
	grad_f[prob.num_var] = prob.slack_penalty
	return nothing
end

∇obj = zeros(prob_sim.num_var)
MOI.eval_objective_gradient(prob_sim, ∇obj, z0)

function MOI.eval_constraint(prob::MOISimulator, g, x)
	model = prob.model
	q3 = view(x, 1:model.nq)
	λ = view(x, model.nq .+ (1:model.nc))
	b = view(x, model.nq + model.nc .+ (1:model.nb))
	ψ = view(x, model.nq + model.nc + model.nb .+ (1:model.nc))
	η = view(x, model.nq + model.nc + model.nb + model.nc .+ (1:model.nb))
	s = view(x, model.nq + model.nc + model.nb + model.nc + model.nb .+ (1:model.ns))
	# λ = 0
	# b = 0
    g[1:model.nq] = dynamics(model,
		prob.v1, prob.q1, prob.q2, q3, prob.u, λ, b, prob.w, prob.h)
	g[model.nq .+ (1:model.nc)] = ϕ_func(model, q3)
	g[model.nq + model.nc .+ (1:model.nc)] = friction_cone(model, λ, b)
	g[model.nq + model.nc + model.nc .+ (1:model.nb)] = maximum_dissipation(model,
		prob.q2, q3, ψ, η, prob.h)
	g[model.nq + model.nc + model.nc + model.nb + 1] = s[1] - (λ' * ϕ_func(model, q3))[1]
	g[model.nq + model.nc + model.nc + model.nb + 2] = s[1] - (ψ' * friction_cone(model, λ, b))[1]
	g[model.nq + model.nc + model.nc + model.nb + 3] = s[1] - (η' * b)[1]

    return nothing
end
prob_sim.h
c0 = zeros(prob_sim.num_con)
MOI.eval_constraint(prob_sim, c0, z0)

function MOI.eval_constraint_jacobian(prob::MOISimulator, jac, x)
    con!(g, z) = MOI.eval_constraint(prob, g, z)
	ForwardDiff.jacobian!(reshape(jac, prob.num_var, prob.num_con), con!, zeros(prob.num_con), x)
    return nothing
end

∇c0 = vec(zeros(prob_sim.num_con, prob_sim.num_var))
MOI.eval_constraint_jacobian(prob_sim, ∇c0, z0)

function sparsity_jacobian(prob::MOISimulator)
    row = []
    col = []

    r = (1:prob.num_con)
    c = (1:prob.num_var)

    row_col!(row, col, r, c)

    return collect(zip(row, col))
end

sparsity_jacobian(prob_sim)

@time solve(prob_sim, z0)

function step_contact(model, v1, q1, q2, u, w, h)
    prob = simulator_problem(model, v1, q1, q2, u, w, h)
    z0 = [copy(q2); 1.0e-5 * ones(model.nc + model.nb + model.nc + model.nb + model.ns)]
    @time z = solve(prob, copy(z0), tol = 1.0e-6, c_tol = 1.0e-6, mapl = 0)

    @assert z[end] < 1.0e-5

    return z[1:model.nq]
end



q_sim = [q_proj[1], q_proj[2]]
v_sim = [(q_proj[2] - q_proj[1]) / h]

for t = 1:T-1
	# 2x rate
	_q_sim = [q_sim[end-1], q_sim[end]]
	_v_sim = [v_sim[end]]

	d = 4
	for i = 1:d
		_h = h / convert(Float64, d)
		_q = step_contact(model,
			_v_sim[end], _q_sim[end-1], _q_sim[end], u_proj[t][model.idx_u], zeros(model.d), _h)
		_v = (_q - _q_sim[end]) / _h

		push!(_q_sim, _q)
		push!(_v_sim, _v)
	end

	push!(q_sim, _q_sim[end])
	push!(v_sim, _v_sim[end])

	# same rate
	# push!(q_sim, step_contact(model,
	# 	v_sim[end], q_sim[end-1], q_sim[end], ū[t][model.idx_u], zeros(model.d), h))
	# push!(v_sim, (q_sim[end] - q_sim[end-1]) / h)
end

plot(hcat(q_proj...)',
    color = :red,
    width = 2.0,
	label = "")
plot!(hcat(q_sim...)',
    color = :black,
    width = 1.0,
	label = "")

plot(hcat(v_sim...)',
    color = :black,
    width = 1.0,
	label = "")
