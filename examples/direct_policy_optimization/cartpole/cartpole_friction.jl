include(joinpath(pwd(),"src/models/cartpole.jl"))
include(joinpath(pwd(),"src/constraints/friction.jl"))

optimize = true

# Model
μ0 = 0.1 # coefficient of friction

model_nominal = CartpoleFriction(n, m, d, 1.0, 0.2, 0.5, 9.81, 0.0)
model_friction = CartpoleFriction(n, m, d, 1.0, 0.2, 0.5, 9.81, μ0)

# Horizon
T = 51
tf = 5.0
h = tf / (T-1)

# Bounds
ul_friction = zeros(7)
ul_friction[1] = -10.0
uu_friction = Inf * ones(7)
uu_friction[1] = 10.0

ul, uu = control_bounds(model, T, ul_friction, uu_friction)

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [0.0; π; 0.0; 0.0]

xl, xu = state_bounds(model, T, x1 = x1, xT = xT)

# Objective
obj = quadratic_tracking_objective(
    [(t < T ? Diagonal(ones(model_nominal.n))
        : Diagonal(zeros(model_nominal.n))) for t = 1:T],
    [Diagonal([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for t = 1:T-1],
    [xT for t = 1:T],
    [zeros(model_nominal.m) for t = 1:T])

penalty_obj = PenaltyObjective(1000.0, 7)

multi_obj = MultiObjective([obj, penalty_obj])

# Constraints
n_stage = 5

con_friction = FrictionConstraints(n_stage * (T - 1),
    vcat([(t - 1) * n_stage .+ (3:5) for t = 1:T-1]...),
    n_stage)

# Problem
prob_nominal = trajectory_optimization_problem(model_nominal,
                    multi_obj,
                    T,
                    xl = xl,
                    xu = xu,
                    ul = ul,
                    uu = uu,
                    h = h,
                    con = con_friction)

prob_friction = trajectory_optimization_problem(model_friction,
                    multi_obj,
                    T,
                    xl = xl,
                    xu = xu,
                    ul = ul,
                    uu = uu,
                    h = h,
                    con = con_friction)

# Trajectory initialization
X0 = linear_interp(x1, xT, T) # linear interpolation on state
U0 = [ones(model_nominal.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob_nominal)

# Solve nominal problem
if optimize
    @time Z̄_nominal = solve(prob_nominal, copy(Z0), tol = 1.0e-5, c_tol = 1.0e-5)
    @time Z̄_friction = solve(prob_friction, copy(Z0), tol = 1.0e-5, c_tol = 1.0e-5)

    @save joinpath(@__DIR__, "sol.jld2") Z̄_nominal Z̄_friction
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol.jld2") Z̄_nominal Z̄_friction
end
