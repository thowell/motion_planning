include(joinpath(pwd(), "src/models/car.jl"))
include(joinpath(pwd(), "src/constraints/obstacles.jl"))

optimize = true

# Horizon
T = 51

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T, -3.0, 3.0)

# Initial and final states
x1 = [0.0; 0.0; 0.0]
xT = [1.0; 1.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(ones(model.n)) : Diagonal(10.0 * ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T], [zeros(model.m) for t = 1:T])

# Constraints
circles = [(0.85, 0.3, 0.1),
           (0.375, 0.75, 0.1),
           (0.25, 0.2, 0.1),
           (0.75, 0.8, 0.1)]

# Constraints
function obstacles!(c, x)
    c[1] = circle_obs(x[1], x[2], circles[1][1], circles[1][2], circles[1][3])
    c[2] = circle_obs(x[1], x[2], circles[2][1], circles[2][2], circles[2][3])
    c[3] = circle_obs(x[1], x[2], circles[3][1], circles[3][2], circles[3][3])
    c[4] = circle_obs(x[1], x[2], circles[4][1], circles[4][2], circles[4][3])
    nothing
end

n_stage = 4
n_con = n_stage * T
con_obstacles = ObstacleConstraints(n_con, (1:n_con), n_stage)

# Problem
prob = trajectory_optimization_problem(model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu,
           con = con_obstacles
           )

# Trajectory initialization
X0 = linear_interp(x1, xT, T) # linear interpolation on state
U0 = random_controls(model, T, 0.001) # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

Z̄ = solve(prob, copy(Z0))

include(joinpath(pwd(), "src/direct_policy_optimization/dpo.jl"))

# Additive noise model
model = Car(3, 2, 3)

function fd(model::Car, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, h) - w
end

# Nominal solution
X̄, Ū = unpack(Z̄, prob)
prob_nom = prob.prob

# DPO
N = 2 * model.n
D = 2 * model.d

β = 1.0
δ = 1.0e-3

# initial samples
x1_sample = resample(x1, Diagonal([1.0; 1.0; 0.1]), 1.0e-1)

# mean problem
prob_mean = trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				dynamics = false
				)

# sample problems
prob_sample = [trajectory_optimization(
				model,
				EmptyObjective(),
				T,
				h = h,
				xl = state_bounds(model, T, x1 = x1_sample[i])[1],
				xu = state_bounds(model, T, x1 = x1_sample[i])[2],
				ul = ul,
				uu = uu,
				dynamics = false,
				con = con_obstacles
				) for i = 1:N]

# sample objective
Q = [(t < T ? Diagonal([10.0; 10.0; 1.0])
	: Diagonal(100.0 * ones(model.n))) for t = 1:T]
R = [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1]

obj_sample = sample_objective(Q, R)
policy = linear_feedback(model.n, model.m)
dist = disturbances([Diagonal(δ * [1.0; 1.0; 0.1]) for t = 1:T-1])
sample = sample_params(β, T)

prob_dpo = dpo_problem(
	prob_nom, prob_mean, prob_sample,
	obj_sample,
	policy,
	dist,
	sample)



# TVLQR policy
K = tvlqr(model, X̄, Ū, Q, R, h)

z0_dpo = zeros(prob_dpo.num_var)
z0_dpo[prob_dpo.prob.idx.nom] = pack(X̄, Ū, prob_nom)
z0_dpo[prob_dpo.prob.idx.mean] = pack(X̄, Ū, prob_nom)
for i = 1:N
	z0_dpo[prob_dpo.prob.idx.sample[i]] = pack(X̄, Ū, prob_nom)
end
for t = 1:T-1
	z0_dpo[prob_dpo.prob.idx.policy[prob_dpo.prob.idx.θ[t]]] = vec(copy(K[t]))
end

# Solve
# Z = solve(prob_dpo, copy(z0_dpo),
# 	tol = 1.0e-3, c_tol = 1.0e-3,
# 	max_iter = 1000)

function sample_dynamics(model, xt, ut, μ, ν, w, h, t, β)
	N = 2 * model.n
	D = 2 * model.d
	M = N + D

	w0 = zeros(model.d)

	# propagate samples
	s = zeros(model.n * M)

	for j = 1:M
		if j <= N
			xi = view(xt, (j - 1) * model.n .+ (1:model.n))
			ui = view(ut, (j - 1) * model.m .+ (1:model.m))
			s[(j - 1) * model.n .+ (1:model.n)] = propagate_dynamics(model,
				xi, ui, w0, h, t)
		else
			k = j - N
			s[(j - 1) * model.n .+ (1:model.n)] = propagate_dynamics(model,
				μ, ν,
				β[t] * w[t][k], h, t)
		end
	end

	# resample
	xt⁺ = resample_vec(s, model.n, M, β[t + 1])

	return xt⁺, s
end

function sample_dynamics_jacobian(model, xt, ut, μ, ν, w, h, t, β)
	N = 2 * model.n
	D = 2 * model.d
	M = N + D

	w0 = zeros(model.d)

	dx⁺dxt = zeros(model.n * M, model.n * N)
	dx⁺dut = zeros(model.n * M, model.m * N)
	dx⁺dμ = zeros(model.n * M, model.n)
	dx⁺dν = zeros(model.n * M, model.m)

	dsdxt = zeros(model.n * M, model.n * N)
	dsdut = zeros(model.n * M, model.m * N)
	dsdμ = zeros(model.n * M, model.n)
	dsdν = zeros(model.n * M, model.m)

	xt⁺, s = sample_dynamics(model, xt, ut, μ, ν, w, h, t, β)
	r(y) = resample_vec(y, model.n, M, β[t + 1])
	dx⁺ds = real.(FiniteDiff.finite_difference_jacobian(r, s))

	# s = []
	# A = []
	# B = []

	for j = 1:M
		if j <= N
			xi = view(xt, (j - 1) * model.n .+ (1:model.n))
			ui = view(ut, (j - 1) * model.m .+ (1:model.m))
			_, _A, _B = propagate_dynamics_jacobian(model, xi, ui,
				w0, h, t)
			# push!(s, _s)
			# push!(A, _A)
			# push!(B, _B)

			dsdxt[(j - 1) * model.n .+ (1:model.n),
				(j - 1) * model.n .+ (1:model.n)] = _A
			dsdut[(j - 1) * model.n .+ (1:model.n),
				(j - 1) * model.m .+ (1:model.m)] = _B
		else
			k = j - N
			_, _A, _B = propagate_dynamics_jacobian(model, μ, ν,
				β[t] * w[t][k], h, t)

			# push!(s, _s)
			# push!(A, _A)
			# push!(B, _B)

			dsdμ[(j - 1) * model.n .+ (1:model.n), :] = _A
			dsdν[(j - 1) * model.n .+ (1:model.n), :] = _B
		end
	end

	dx⁺dxt = dx⁺ds * dsdxt
	dx⁺dut = dx⁺ds * dsdut
	dx⁺dμ = dx⁺ds * dsdμ
	dx⁺dν = dx⁺ds * dsdν

	return dx⁺dxt, dx⁺dut, dx⁺dμ, dx⁺dν
end

model = model_ft
t = 2
xt = view(z0_dpo, prob_dpo.prob.idx.xt[t])
ut = view(z0_dpo, prob_dpo.prob.idx.ut[t])
μ = view(z0_dpo, prob_dpo.prob.idx.mean[prob_dpo.prob.prob.mean.idx.x[t]])
ν = view(z0_dpo, prob_dpo.prob.idx.mean[prob_dpo.prob.prob.mean.idx.u[t]])

propagate_dynamics(model, rand(model.n), rand(model.m), rand(model.d), h, t)
propagate_dynamics_jacobian(model, rand(model.n), rand(model.m), rand(model.d), h, t)

sample_dynamics(model, xt, ut, μ, ν, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)

a1, a2, a3, a4 = sample_dynamics_jacobian(model, xt, ut, μ, ν, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)

sdx(y) = sample_dynamics(model, y, ut, μ, ν, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)[1]
sdx(xt)

sdu(y) = sample_dynamics(model, xt, y, μ, ν, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)[1]
sdu(ut)

sdμ(y) = sample_dynamics(model, xt, ut, y, ν, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)[1]
sdμ(μ)

sdν(y) = sample_dynamics(model, xt, ut, μ, y, prob_dpo.prob.dist.w, h, t,
	prob_dpo.prob.sample.β)[1]
sdν(ν)

norm(FiniteDiff.finite_difference_jacobian(sdx, xt) - a1)
norm(FiniteDiff.finite_difference_jacobian(sdu, ut) - a2)
norm(FiniteDiff.finite_difference_jacobian(sdμ, μ) - a3)
norm(FiniteDiff.finite_difference_jacobian(sdν, ν) - a4)
