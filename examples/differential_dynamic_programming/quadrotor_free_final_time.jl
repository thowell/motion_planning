using Plots
include_ddp()

# Model
include_model("quadrotor")

# Time
T = 101
h = 0.1

# sigmoid(z) = 1.0 / (1.0 + exp(-z))
# sigmoid_inv(z) = log(z / (1.0 - z))

function fd(model::Quadrotor{Midpoint, FixedTime}, x, u, w, h, t)
	if t == 1
		_h = u[5]
	else
		_h = x[13]
	end
	return [view(x, 1:model.n) + _h * f(model, view(x, 1:model.n) + 0.5 * _h * f(model, view(x, 1:model.n), view(u, 1:model.m), w), view(u, 1:model.m), w);
	        _h]
end

n = [t == 1 ? model.n : model.n + 1 for t = 1:T]
m = [t == 1 ? model.m + 1 : model.m for t = 1:T]

fd(model, rand(model.n), rand(model.m + 1), zeros(model.d), 1.0, 1)
fd(model, rand(model.n + 1), rand(model.m), zeros(model.d), 1.0, 2)



# Initial conditions, controls, disturbances
x1 = zeros(model.n)
x1[3] = 1.0

xT = copy(x1)
# xT[1] = 5.0
# xT[2] = 2.5
xT[3] = 5.0

u_ref = -1.0 * model.mass * model.g[3] / 4.0 * ones(model.m)
# ū = [t == 1 ? [u_ref; h] : u_ref for t = 1:T-1]
ū_alt = [t == 1 ? [ū[t]; h] : ū[t] for t = 1:T-1]

w = [zeros(model.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model, x1, ū_alt, w, h, T)

# Objective
# Q = [(t == 1 ? 0.0 * Diagonal(1.0e-3 * ones(model.n))
#         : (t == T ? 0.0 * Diagonal([1.0 * ones(model.n); 0.0])
# 		: 0.0 * Diagonal([1.0e-3 * ones(model.n); 0.0]))) for t = 1:T]
# q = [(t == 1 ? -2.0 * Q[t] * xT
#  	 : -2.0 * Q[t] * [xT; 0.0]) for t = 1:T]
#
# R = [(t == 1 ? Diagonal([1.0 * ones(model.m); 1.0e-3])
# 	 : Diagonal(1.0 * ones(model.m))) for t = 1:T-1]
# r = [(t == 1 ? [-2.0 * R[t][1:model.m, 1:model.m] * u_ref; 1.0]
# 	 : -2.0 * R[t] * u_ref)  for t = 1:T-1]

# obj = StageCosts([QuadraticCost(Q[t], q[t],
# 	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

# function g(obj::StageCosts, x, u, t)
# 	T = obj.T
#     if t < T
# 		Q = obj.cost[t].Q
# 		q = obj.cost[t].q
# 	    R = obj.cost[t].R
# 		r = obj.cost[t].r
#         return x' * Q * x + q' * x + u' * R * u + r' * u
#     elseif t == T
# 		Q = obj.cost[T].Q
# 		q = obj.cost[T].q
#         return x' * Q * x + q' * x
#     else
#         return 0.0
#     end
# end

obj = StageCosts([NonlinearCost() for t = 1:T], T)

Q = [(t < T ? Diagonal([1.0 * ones(6); 1.0 * ones(6)])
     : Diagonal([0.0 * ones(6); 0.0 * ones(6)])) for t = 1:T]
q = [-2.0 * Q[t] * xT for t = 1:T]

R = [Diagonal(1.0 * ones(model.m)) for t = 1:T-1]
r = [-2.0 * R[t] * u_ref for t = 1:T-1]

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		t == 1 ? (_h = u[end]) : (_h = x[end])
        J = h * (view(x, 1:model.n)' * Q[t] * view(x, 1:model.n) + q[t]' * view(x, 1:model.n) + view(u, 1:model.m)' * R[t] * view(u, 1:model.m) + r[t]' * view(u, 1:model.m))
		t == 1 && (J += 1.0e-5 * _h^2.0)
		return J
	elseif t == T
        return view(x, 1:model.n)' * Q[T] * view(x, 1:model.n) + q[T]' * view(x, 1:model.n)
    else
        return 0.0
    end
end

# Constraints
p = [t < T ? (t == 1 ? 2 * model.m + 2 : 2 * model.m) : model.n for t = 1:T]
info_1 = Dict(:ul => [zeros(model.m); 0.0], :uu => [10.0 * ones(model.m); 2.0 * h], :inequality => (1:2 * model.m + 2))
info_t = Dict(:ul => zeros(model.m), :uu => 10.0 * ones(model.m), :inequality => (1:2 * model.m))
info_T = Dict(:xT => xT)
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		c .= [ul - u; u - uu]
	elseif t == T
		xT = cons.con[T].info[:xT]
		c .= view(x, 1:model.n) - xT
	else
		nothing
	end
end

prob = problem_data(model, obj, con_set, copy(x̄), copy(ū_alt), w, h, T,
	n = n, m = m)

# Solve
@time constrained_ddp_solve!(prob,
    max_iter = 1000, max_al_iter = 10,
	con_tol = 1.0e-3,
	ρ_init = 1.0, ρ_scale = 10.0)

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

# Time step
@show u[1][end]


# Trajectories
plot(hcat([ut[1:model.m] for ut in u]...)', linetype = :steppost)
plot(hcat([xt[1:model.n] for xt in x]...)[1:3, :]', linetype = :steppost)

# Visualize
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
# open(vis)
visualize!(vis, model, x, Δt = u[1][5])
