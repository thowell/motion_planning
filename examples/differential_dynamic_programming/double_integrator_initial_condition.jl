using Plots
using Random
Random.seed!(1)

include_ddp()

# Model
include_model("double_integrator")

function f(model::DoubleIntegratorContinuous, x, u, w)
    [x[2]; u[1]]
end

function fd(model::DoubleIntegratorContinuous{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		x1 = u[2:3]
    	return x1 + h * f(model, x1 + 0.5 * h * f(model, x1, u, w), u, w)
	else
		return x + h * f(model, x + 0.5 * h * f(model, x, u, w), u, w)
	end
end

# Time
T = 11
h = 0.1

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 0)
n = [model.n for t = 1:T]
m = [t == 1 ? model.m + model.n : model.m for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
xT = [0.0; 0.0]
ū = [1.0 * randn(m[t]) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ? h : 1.0) * Diagonal([1.0; 1.0]) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]
R = [(t < T ? h : 1.0) * Diagonal(1.0 * ones(m[t])) for t = 1:T-1]
r = [zeros(m[t]) for t = 1:T-1]
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
p = [t == 1 ? 2 * n[1] + 2 * m[1] : (t == T ? n[T] : 2 * m[t]) for t = 1:T]
ul = [-5.0]
uu = [5.0]
info_1 = Dict(:x1l => [0.5; -0.1], :x1u => [1.5; 0.1],
	:ul => ul, :uu => uu, :inequality => (1:2 * n[1] + 2 * model.m))
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:(2 * model.m)))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	if t == 1
		c[1:1] = view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] = cons.con[t].info[:ul] - view(u, 1:1)
		c[3:4] .= view(u, 2:3) - cons.con[t].info[:x1u]
		c[5:6] = cons.con[t].info[:x1l] - view(u, 2:3)
	elseif t < T
		c[1:1] = view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] = cons.con[t].info[:ul] - view(u, 1:1)
	end
	if t == T
		c[1:2] .= view(x, 1:2) - cons.con[t].info[:xT]
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T, n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 6,
	ρ_init = 1.0, ρ_scale = 10.0)

# # Solve
# @time ddp_solve!(prob,
#     max_iter = 100, verbose = true)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

x[1] = u[1][2:3]
x̄[1] = ū[1][2:3]

plt = plot(hcat([xt[1:2] for xt in x]...)',
	width = 2.0,
	color = [:cyan :orange],
	label = ["x" "ẋ"],
	xlabel = "time step",
	ylabel = "state")

savefig(plt,
	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_initial_state.png"))

plt = plot(hcat([ut[1] for ut in u]..., u[end][1])',
	width = 2.0,
	color = :magenta,
	linetype = :steppost,
	xlabel = "time step",
	ylabel = "control",
	label = "")

savefig(plt,
	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_initial_control.png"))

@show u[1][2]
