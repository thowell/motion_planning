include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))
include(joinpath(pwd(), "src/models/quadrotor2D.jl"))

# Additive noise model
model = additive_noise_model(model)

function fd(model::Quadrotor2D, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

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
           uu = uu,
           )

# Trajectory initialization
X0 = linear_interp(x1_nom, xT_nom, T) # linear interpolation on state
U0 = [0.1 * ones(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

Z̄ = solve(prob, copy(Z0))
X̄, Ū = unpack(Z̄, prob)

using Plots
plot(hcat(X̄...)')
plot(hcat(Ū...)', linetype = :steppost)

# DPO
# Linear model
A, B = jacobians(model, X̄, Ū, h)

function fd(model::Quadrotor2D, x⁺, x, u, w, h, t)
    x⁺ - A[t] * x - B[t] * u - w
end

# DPO
N = 2 * model.n
D = 2 * model.d

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
			uu = control_bounds(model, T, 0.0, 0.0)[2],
			)

# mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false
				)

# sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				dynamics = false,
				xl = state_bounds(model, T, x1 = x1[i])[1],
				xu = state_bounds(model, T, x1 = x1[i])[2]
				) for i = 1:N]

# sample objective
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
include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

z_sol = solve(prob_dpo, copy(z0),
	nlp = :SNOPT7,
	tol = 1.0e-7, c_tol = 1.0e-7,
	time_limit = 60 * 20)

# TVLQR policy
K = tvlqr(A, B, Q, R)

# DPO policy
θ = [reshape(z_sol[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]],
	model.m, model.n) for t = 1:T-1]

# Policy difference
policy_diff = [norm(vec(θ[t] - K[t])) / norm(vec(K[t])) for t = 1:T-1]
println("policy difference (inf. norm): $(norm(policy_diff, Inf))")
