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
    	return [x1 + h * f(model, x1 + 0.5 * h * f(model, x1, u, w), u, w); x1]
	else
		x1 = x[3:4]
		return [view(x, 1:2) + h * f(model, view(x, 1:2) + 0.5 * h * f(model, view(x, 1:2), u, w), u, w); x1]
	end
end

# Time
T = 11
Tm = 6
h = 0.1

model = DoubleIntegratorContinuous{Midpoint, FixedTime}(2, 1, 0)
n = [t == 1 ? model.n : 2 * model.n for t = 1:T]
m = [t == 1 ? model.m + model.n : model.m for t = 1:T]

# Initial conditions, controls, disturbances
x1 = [1.0; 0.0]
xM = [0.0; 0.0]
xT = [1.0; 0.0]
ū = [t == 1 ? [1.0 * randn(1); randn(2)] : 1.0 * randn(1) for t = 1:T-1]
w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū, w, h, T)

# Objective
Q = [(t < T ? h : 1.0) * (t > 1 ? Diagonal([ones(2); zeros(2)]) : Diagonal(ones(2))) for t = 1:T]
q = [(t > 1 ? -2.0 * Q[t] * [xT; zeros(2)] : -2.0 * Q[t] * xT) for t = 1:T]
R = h * [t > 1 ? Diagonal(1.0 * ones(1)) : Diagonal([1.0; 1.0e-5 * ones(model.n)]) for t = 1:T-1]
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
		Q = o bj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

g(obj, x̄[1], ū[1], 1)
g(obj, x̄[2], ū[2], 2)
g(obj, x̄[T], nothing, T)
objective(obj, x̄, ū)

# Constraints
p = [t == 1 ? 2 * model.n + 2 * model.m : (t == T ? model.n : (t == Tm ? model.n + 2 * model.m : 2 * model.m)) for t = 1:T]
ul = [-5.0]
uu = [5.0]
info_1 = Dict(:x1l => [0.1; -1.0], :x1u => [10.0; 1.0],
	:ul => ul, :uu => uu,
	:inequality => (1:2 * model.m + 2 * model.n))
info_M = Dict(:xM => xM,
	:ul => ul, :uu => uu, :inequality => (1:2 * model.m))
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model.m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : (t == Tm ? info_M : info_t)) : info_T) for t = 1:T]

info_t
function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	if t == 1
		c[1:1] .= view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] .= cons.con[t].info[:ul] - view(u, 1:1)
		c[3:4] .= view(u, 2:3) - cons.con[t].info[:x1u]
		c[5:6] .= cons.con[t].info[:x1l] - view(u, 2:3)
	end
	if t > 1 && t < T
		c[1:1] .= view(u, 1:1) - cons.con[t].info[:uu]
		c[2:2] .= cons.con[t].info[:ul] - view(u, 1:1)
		if t == Tm
			c[3:4] .= view(x, 1:2) - cons.con[t].info[:xM]
		end
	end
	if t == T
		c[1:2] .= view(x, 1:2) - view(x, 3:4)
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū), w, h, T, n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

x[1] = u[1][2:3]
x̄[1] = ū[1][2:3]
norm(x[1] - x[T][1:2])

plt = plot(hcat([xt[1:2] for xt in x]...)',
	width = 2.0,
	color = [:cyan :orange],
	label = ["x" "ẋ"],
	xlabel = "time step",
	ylabel = "state")

savefig(plt,
	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_gait_state.png"))

plt = plot(hcat([ut[1] for ut in u]..., u[end][1])',
	width = 2.0,
	color = :magenta,
	linetype = :steppost,
	xlabel = "time step",
	ylabel = "control",
	label = "")

savefig(plt,
	joinpath("/home/taylor/Research/parameter_optimization_manuscript/figures/di_gait_control.png"))
