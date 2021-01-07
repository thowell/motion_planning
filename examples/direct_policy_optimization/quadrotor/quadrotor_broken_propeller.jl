# Model
include_model("quadrotor")
model = free_time_model(additive_noise_model(model))

function fd(model::Quadrotor{Midpoint, FreeTime}, x⁺, x, u, w, h, t)
	h = u[end]
    x⁺ - (x + h * f(model, 0.5 * (x + x⁺), u, w) + w)
end

# Horizon
T = 31

# Time step
tf0 = 5.0
h0 = tf0 / (T - 1)

# ul <= u <= uu
_uu = 5.0 * ones(model.m)
_uu[end] = h0

uu_nom = copy(_uu)
uu1 = copy(_uu)
uu1[1] *= 0.5
uu2 = copy(_uu)
uu2[2] *= 0.5
uu3 = copy(_uu)
uu3[3] *= 0.5
uu4 = copy(_uu)
uu4[4] *= 0.5

_ul = zeros(model.m)
_ul[end] = 0.0
ul_nom = copy(_ul)
@assert sum(_uu) > -1.0 * model.mass * model.g[3]

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
x1 = zeros(model.n)
x1[3] = 1.0
xT = copy(x1)
xT[1] = 3.0
xT[2] = 3.0

_xl = -Inf * ones(model.n)
_xl[3] = 0.0

_xu = Inf * ones(model.n)

xl, xu = state_bounds(model, T, _xl, _xu, x1 = x1, xT = xT)

u_ref = [-1.0 * model.mass * model.g[3] / 4.0 * ones(4); 0.0]

# Objective
obj = quadratic_time_tracking_objective(
	[(t < T ? Diagonal(ones(model.n))
		: Diagonal(1.0 * ones(model.n))) for t = 1:T],
	[Diagonal([1.0e-1 * ones(4); 0.0]) for t = 1:T-1],
    [copy(xT) for t = 1:T],
	[copy(u_ref) for t = 1:T-1],
	1.0)

# Time step constraints
include_constraints("free_time")
con_free_time = free_time_constraints(T)

# Problem
prob = trajectory_optimization_problem(model,
			   obj,
			   T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_free_time)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = [[copy(u_ref[1:4]); h0] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
if false
    @time z̄, info = solve(prob, copy(z0))
    @save joinpath(@__DIR__, "sol_to.jld2") z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_to.jld2") z̄
end
