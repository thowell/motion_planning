include_dpo()

# Model
include_model("quadrotor2D")
model = additive_noise_model(model)

# Horizon
T = 51

# Time step
h = 0.1

# Bounds
ul, uu = control_bounds(model, T, 0.0, Inf)

# Initial and final states
x1_nom = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xT_nom = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1_nom, xT = xT_nom)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(ones(model.n)) : Diagonal(10.0 * ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT_nom for t = 1:T], [zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu)

# Trajectory initialization
x0 = linear_interpolation(x1_nom, xT_nom, T) # linear interpolation on state
u0 = [0.1 * ones(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve for nominal trajectory
z̄, info = solve(prob, copy(z0))
x̄, ū = unpack(z̄, prob)

using Plots
plot(hcat(x̄...)')
plot(hcat(ū...)', linetype = :steppost)

# DPO

# Linear model
A, B = jacobians(model, x̄, ū, h)

function fd(model::Quadrotor2D{Midpoint, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - A[t] * x - B[t] * u - w
end

# DPO
β = 1.0
δ = 1.0e-1

x1 = resample(zeros(model.n), Diagonal(ones(model.n)), β)

# Problems
prob_nom = trajectory_optimization(
			model,
			EmptyObjective(),
			T,
			xl = state_bounds(model, T, zeros(model.n), zeros(model.n))[1],
			xu = state_bounds(model, T, zeros(model.n), zeros(model.n))[2],
			ul = control_bounds(model, T, 0.0, 0.0)[1],
			uu = control_bounds(model, T, 0.0, 0.0)[2])

# Mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false)

# Sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false,
				xl = state_bounds(model, T, x1 = x1[i])[1],
				xu = state_bounds(model, T, x1 = x1[i])[2]) for i = 1:2 * model.n]

# Sample objective
Q = [(t < T ? Diagonal(10.0 * ones(model.n))
 		: Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m)
dist = disturbances([Diagonal(δ * ones(model.d)) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)

z0 = ones(prob_dpo.num_var)

# Solve
if true
	include_snopt()
	z, info = solve(prob_dpo, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-7, c_tol = 1.0e-7,
		time_limit = 60 * 2)
	@save joinpath(@__DIR__, "sol_dpo.jld2") z
else
	println("Loading solution...")
    @load joinpath(@__DIR__, "sol_dpo.jld2") z
end

# DPO policy
θ = get_policy(z, prob_dpo)

# TVLQR policy
K, P = tvlqr(A, B, Q, R)

# Policy difference
policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
println("policy difference (inf. norm): $(norm(policy_diff, Inf))")
