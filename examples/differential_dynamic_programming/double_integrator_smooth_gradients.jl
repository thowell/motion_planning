using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; (1.0 + w[1]) * u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
end

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 1)
n = model.n
m = model.m

# Time
T = 11
h = 0.1

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
x_ref = [[0.0; 0.0] for t = 1:T]
xT = [0.0; 0.0]
ū = [1.0 * randn(model.m) for t = 1:T-1]
u_ref = [zeros(model.m) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ? h : 1.0) * (t < T ?
	 Diagonal([1.0; 1.0])
		: Diagonal([1.0; 1.0])) for t = 1:T]
q = [-2.0 * Q[t] * x_ref[t] for t = 1:T]
R = h * [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
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
ul = [-5.0]
uu = [5.0]
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
	analytical_dynamics_derivatives = false)

# Solve
@time constrained_ddp_solve!(prob,
    linesearch = :armijo,
	max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0,
	con_tol = 1.0e-5)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

n = [model.n for t = 1:T]
m = [model.m for t = 1:T-1]

# Visualize
using Plots
plt = plot(hcat(x...)',
	width = 2.0,
	color = [:cyan :orange],
	label = ["x" "ẋ"],
	xlabel = "time step",
	ylabel = "state")
