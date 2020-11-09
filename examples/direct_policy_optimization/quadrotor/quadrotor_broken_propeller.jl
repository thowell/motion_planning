include(joinpath(pwd(),"src/models/quadrotor.jl"))
include(joinpath(pwd(),"src/constraints/free_time.jl"))

optimize = true

# Free-time model
model_ft = free_time_model(model)

function fd(model::Quadrotor, x⁺, x, u, w, h, t)
	midpoint_implicit(model, x⁺, x, u, w, u[end])
end

# Horizon
T = 31

# Time step
tf0 = 5.0
h0 = tf0 / (T-1)

# ul <= u <= uu
_uu = 5.0 * ones(model_ft.m)
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

_ul = zeros(model_ft.m)
_ul[end] = 0.0
ul_nom = copy(_ul)
@assert sum(_uu) > -1.0 * model_ft.mass * model.g[3]

ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
x1 = zeros(model_ft.n)
x1[3] = 1.0
xT = copy(x1)
xT[1] = 3.0
xT[2] = 3.0

_xl = -Inf * ones(model_ft.n)
_xl[3] = 0.0

_xu = Inf * ones(model_ft.n)

xl, xu = state_bounds(model_ft, T, _xl, _xu, x1 = x1, xT = xT)

u_ref = [-1.0 * model_ft.mass * model_ft.g[3] / 4.0 * ones(4); 0.0]

# Objective
obj = quadratic_time_tracking_objective(
	[(t < T ? Diagonal(ones(model_ft.n))
		: Diagonal(1.0 * ones(model_ft.n))) for t = 1:T],
	[Diagonal([1.0e-1 * ones(4); 0.0]) for t = 1:T-1],
    [copy(xT) for t = 1:T],
	[copy(u_ref) for t = 1:T-1],
	1.0)

# Time step constraints
con_free_time = free_time_constraints(T)

# Problem
prob = trajectory_optimization_problem(model_ft,
			   obj,
			   T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_free_time
               )

# Trajectory initialization
X0 = linear_interp(x1, xT, T) # linear interpolation on state
U0 = [[copy(u_ref[1:4]); h0] for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
if optimize
    @time Z̄ = solve(prob, copy(Z0))
    @save joinpath(@__DIR__, "sol.jld2") Z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol.jld2") Z̄
end
