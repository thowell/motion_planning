include(joinpath(pwd(), "src/models/pendulum.jl"))
include(joinpath(pwd(), "src/constraints/free_time.jl"))

# Free-time model with additive noise
model = free_time_model(additive_noise_model(model))

function fd(model::Pendulum, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end]) - w
end

# Horizon
T = 51

# Bounds
tf = 2.0
h0 = tf / (T-1) # timestep

ul, uu = control_bounds(model, T, [-3.0; 0.0], [3.0; h0])

# Initial and final states
x1 = [0.0; 0.0]
xT = [π; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective (minimum time)
obj = PenaltyObjective(1.0, model.m)

# Time step constraints
con_free_time = free_time_constraints(T)

# Problem
prob = trajectory_optimization_problem(model,
			   obj,
			   T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
			   con = con_free_time
               )

# Initialization
x0 = linear_interp(x1, xT, T) # linear interpolation for states
u0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
optimize = true

if optimize
    @time z̄ = solve(prob, copy(z0))
    @save joinpath(@__DIR__, "sol_to.jld2") z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_to.jld2") z̄
end
