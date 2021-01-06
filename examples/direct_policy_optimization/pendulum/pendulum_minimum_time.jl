# Model
include_model("pendulum")
model = free_time_model(additive_noise_model(model))

# Horizon
T = 51

# Bounds
tf = 2.0
h0 = tf / (T - 1) # timestep

ul, uu = control_bounds(model, T, [-3.0; 0.0], [3.0; h0])

# Initial and final states
x1 = [0.0; 0.0]
xT = [π; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective (minimum time)
obj = PenaltyObjective(1.0, model.m)

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

# Initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation for states
u0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
if true
    @time z̄, info = solve(prob, copy(z0))
    @save joinpath(@__DIR__, "sol_to.jld2") z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_to.jld2") z̄
end
