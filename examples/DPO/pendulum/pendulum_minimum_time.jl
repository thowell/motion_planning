include(joinpath(pwd(), "src/models/pendulum.jl"))
include(joinpath(pwd(), "src/constraints/free_time.jl"))

optimize = true

# Free-time model with additive noise
model_ft = Pendulum(2, 2, 1, 1.0, 0.1, 0.5, 0.25, 9.81)

function fd(model::Pendulum, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end]) - w[1] * ones(model.n)
end

# Horizon
T = 51

# Bounds
tf = 2.0
h0 = tf / (T-1) # timestep

ul, uu = control_bounds(model_ft, T, [-3.0; 0.0], [3.0; h0])

# Initial and final states
x1 = [0.0; 0.0]
xT = [π; 0.0]

xl, xu = state_bounds(model_ft, T, x1 = x1, xT = xT)

# Objective (minimum time)
obj = PenaltyObjective(1.0, model_ft.m)

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

# Initialization
X0 = linear_interp(x1, xT, T) # linear interpolation for states
U0 = [ones(model_ft.m) for t = 1:T-1]

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
