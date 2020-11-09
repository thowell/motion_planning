include(joinpath(pwd(), "src/models/double_integrator.jl"))

# Horizon
T = 11

# Bounds
tf = 1.0
h0 = tf / (T-1) # timestep

# Initial and final states
x1 = [1.0; 1.0]
xT = [0.0; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(ones(model.n)) : Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T], [zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
			   obj,
			   T,
               xl = xl,
               xu = xu,
               )

# Initialization
X0 = linear_interp(x1, xT, T) # linear interpolation for states
U0 = [ones(model.m) for t = 1:T-1]

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

# Solve
@time Z̄ = solve(prob, copy(Z0))

# Visualize
using Plots
X̄, Ū = unpack(Z̄, prob)
plot(hcat(X̄...)', width = 2.0)
plot(hcat(Ū...)', width = 2.0, linetype = :steppost)
