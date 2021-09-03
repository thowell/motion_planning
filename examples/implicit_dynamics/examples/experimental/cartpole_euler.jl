using Random, Plots
include_ddp()

# Model
include_model("cartpole")
n, m, d = 4, 1, 4
model = Cartpole{Euler, FixedTime}(n, m, d, 1.0, 0.2, 0.5, 9.81)

function fd(model::Cartpole{Euler, FixedTime}, x⁺, x, u, w, h, t)
    x⁺ - (x + h * f(model, x⁺, u, w))
end

function fd(model::Cartpole{Euler, FixedTime}, x, u, w, h, t)
    rz(z) = fd(model, z, x, u, w, h, t) # implicit Euler integration
	x⁺ = newton(rz, copy(x))
    # x⁺ = levenberg_marquardt(rz, copy(x))
	return x⁺
end

fd(model, zeros(model.n), 0.001 * ones(model.m), zeros(model.d), 0.1, 1)

function fdx(model::Cartpole{Euler, FixedTime}, x, u, w, h, t)
	rz(z) = fd(model, z, x, u, w, h, t) # implicit Euler integration
    x⁺ = newton(rz, copy(x))
    # x⁺ = levenberg_marquardt(rz, copy(x))
	∇rz = ForwardDiff.jacobian(rz, x⁺)
	rx(z) = fd(model, x⁺, z, u, w, h, t) # implicit Euler integration
	∇rx = ForwardDiff.jacobian(rx, x)
	return -1.0 * ∇rz \ ∇rx
end

fdx(model, ones(model.n), 0.01 * ones(model.m), zeros(model.d), 0.1, 1)

function fdu(model::Cartpole{Euler, FixedTime}, x, u, w, h, t)
	rz(z) = fd(model, z, x, u, w, h, t) # implicit Euler integration
    x⁺ = newton(rz, copy(x))
    # x⁺ = levenberg_marquardt(rz, copy(x))
	∇rz = ForwardDiff.jacobian(rz, x⁺)
	ru(z) = fd(model, x⁺, x, z, w, h, t) # implicit Euler integration
	∇ru = ForwardDiff.jacobian(ru, u)
	return -1.0 * ∇rz \ ∇ru
end

fdu(model, zeros(model.n), zeros(model.m), zeros(model.d), 0.1, 1)

# function fd(model::Cartpole{Euler, FixedTime}, x, u, w, h, t)
#     x + h * f(model, x, u, w)
# end

# Time
T = 101
tf = 2.5
h = tf / (T - 1)

# Initial conditions, controls, disturbances
x1 = [0.0, 0.0, 0.0, 0.0]
xT = [0.0, π, 0.0, 0.0] # goal state
ū = [(t == 1 ? 1.0 : 0.0) * ones(m) for t = 1:T-1] # initialize no friciton model this direction (not sure why it goes the oppostive direction...)
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)
# x̄ = linear_interpolation(x1, xT, T)

# Objective
Q = [(t < T ? Diagonal(h * [1.0; 1.0; 1.0; 1.0])
        : Diagonal(1.0 * ones(model.n))) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [h * Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
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
p_con = [t == T ? model.n : 2 * model.m for t = 1:T]

uL = -10.0 * ones(model.m)
uU = 10.0 * ones(model.m)
info_t = Dict(:uL => uL, :uU => uU, :inequality => collect(1:2 * model.m))
info_T = Dict(:xT => xT)

con_set = [StageConstraint(p_con[t], t < T ? info_t : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	n = model.n
	m = model.m

	if t < T
		c[1:m] = cons.con[t].info[:uL] - u
		# c[m .+ (1:m)] = u - cons.con[t].info[:uU]
	end

	if t == T
		c[1:n] .= x - cons.con[T].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T,
    analytical_dynamics_derivatives = true)

# Solve
@time stats = constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-3,
	cache = false)

@show sum([s[:iters] for s in stats[:stats_ilqr]])

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Visualize
plot(π * ones(T),
    width = 2.0, color = :black, linestyle = :dash)
plot!(hcat(x̄...)', width = 2.0, label = "",
	ylims = (-4.0, 8.0),
    color = :cyan)
plot(hcat(ū..., ū[end])',
    width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)

# Simulate policy
include(joinpath(@__DIR__, "simulate.jl"))

# Model
model_sim = model
x1_sim = copy(x1)
T_sim = 10 * T

# Time
tf = h * (T - 1)
t = range(0, stop = tf, length = T)
t_sim = range(0, stop = tf, length = T_sim)
dt_sim = tf / (T_sim - 1)

# Policy
K = [K for K in prob.p_data.K]
# plot(vcat(K...))
# K = [K[end] for k in K]
# K = [prob.p_data.K[t] for t = 1:T-1]
# K, _ = tvlqr(model, x̄, ū, h, Q, R)
# # # K = [-k for k in K]
# K = [-K[t] for t = 1:T-1]
# plot(vcat(K...))

# Simulate
N_sim = 1
x_sim = []
u_sim = []
J_sim = []
Random.seed!(1)
for k = 1:N_sim
	wi_sim = 0.0 * rand(range(-1.0, stop = 1.0, length = 1000))

	println("sim: $k")#- w = $(wi_sim[1])")

	x_ddp, u_ddp, J_ddp, Jx_ddp, Ju_ddp = simulate_linear_feedback(
		model_sim,
		K,
	    x̄, ū,
		[xT for t = 1:T], [zeros(model.m) for t = 1:T-1],
		Q, R,
		T_sim, h,
		[wi_sim; 0.0; 0.0; 0.0],
		[zeros(model.d) for t = 1:T-1])

	push!(x_sim, x_ddp)
	push!(u_sim, u_ddp)
	push!(J_sim, J_ddp)
end

# Visualize
idx = (1:2)
plt = plot(t, hcat(x̄...)[idx, :]',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "state",
	title = "LQR cartpole (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for xs in x_sim
	plt = plot!(t_sim, hcat(xs...)[idx, :]',
	    width = 2.0, color = :cyan, label = "")
end
display(plt)

plt = plot(t, hcat(ū..., ū[end])',
	width = 2.0, color = :black, label = "",
	xlabel = "time (s)", ylabel = "control",
	title = "cartpole (J_avg = $(round(mean(J_sim), digits = 3)), N_sim = $N_sim)")

for us in u_sim
	plt = plot!(t_sim, hcat(us..., us[end])',
		width = 1.0, color = :magenta, label = "",
		linetype = :steppost)
end
display(plt)
