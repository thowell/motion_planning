include(joinpath(pwd(), "src/models/acrobot.jl"))

# Horizon
T = 101

# Time step
tf = 5.0
h = tf / (T - 1)

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T, -10.0, 10.0)

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]
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
           h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu,
           )

# Trajectory initialization
X0 = linear_interp(x1, xT, T) # linear interpolation on state
U0 = random_controls(model, T, 0.001) # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

# Solve
@time Z̄ = solve(prob, copy(Z0))

# Visualize
using Plots
X̄, Ū = unpack(Z̄, prob)
plot(hcat(X̄...)', width = 2.0)
plot(hcat(Ū...)', width = 2.0, linetype = :steppost)
