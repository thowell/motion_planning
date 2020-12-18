# Model
include_model("acrobot")

# Horizon
T = 101

# Time step
tf = 5.0
h = tf / (T - 1)

# ul <= u <= uu
ul, uu = control_bounds(model, T, -10.0, 10.0)

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]
xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
        [t < T ? Diagonal(1.0 * ones(model.n)) : Diagonal(ones(model.n)) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [xT for t = 1:T],
        [zeros(model.m) for t = 1:T])

# Problem
prob = trajectory_optimization_problem(model,
           obj,
           T,
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu)

# Trajectory initialization
x0 = linear_interpolation(x1, xT, T) # linear interpolation on state
u0 = random_controls(model, T, 0.001) # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z = solve(prob, copy(z0))

# Visualize
using Plots
x, u = unpack(z, prob)
plot(hcat(x...)', width = 2.0)
plot(hcat(u...)', width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)
