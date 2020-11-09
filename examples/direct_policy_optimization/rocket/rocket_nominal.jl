include(joinpath(pwd(),"src/models/rocket.jl"))
include(joinpath(pwd(),"src/constraints/free_time.jl"))

optimize = true

# Free-time model
model_nom = free_time_model(model_nominal)

function fd(model::RocketNominal, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end])
end

# Horizon
T = 41

# Bounds
ul, uu = control_bounds(model_nom, T, [-5.0; 0.0; 0.01], [5.0; 100.0; 1.0])

# Initial and final states
x1 = [15.0;
	  model_nom.l1 + 10.0;
	  -45.0 * pi / 180.0;
	  -10.0;
	  -10.0;
	  -1.0 * pi / 180.0]

xT = [0.0;
	  model_nom.l1;
	  0.0;
	  0.0;
	  0.0;
	  0.0]

# Radius of landing pad
r_pad = 0.25

# xl <= x <= xl
_xl = -Inf * ones(model_nom.n)
_xl[2] = model_nom.l1
_xu = Inf * ones(model_nom.n)

xl, xu = state_bounds(model_nom, T, _xl, _xu, x1 = x1, xT = xT)

xl[T][1] = -1.0 * r_pad
xu[T][1] = r_pad
xl[T][2] = xT[2] - 0.001
xu[T][2] = xT[2] + 0.001
xl[T][3] = -1.0 * pi / 180.0
xu[T][3] = 1.0 * pi / 180.0
xl[T][4] = -0.001
xu[T][4] = 0.001
xl[T][5] = -0.001
xu[T][5] = 0.001
xl[T][6] = -0.01 * pi / 180.0
xu[T][6] = 0.01 * pi / 180.0

# Objective
obj = quadratic_time_tracking_objective(
	[(t != T ? Diagonal([1.0; 10.0; 1.0; 1.0; 10.0; 1.0])
		: Diagonal([10.0; 100.0; 10.0; 10.0; 100.0; 10.0])) for t = 1:T],
	[Diagonal([1.0e-1*ones(2); 0.0]) for t = 1:T-1],
    [xT for t = 1:T],
	[zeros(model_nom.m) for t = 1:T],
	1.0)

# Time step constraints
con_free_time = free_time_constraints(T)

# Problem
prob_nominal = trajectory_optimization_problem(model_nom,
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
U0 = [ones(model_nom.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob_nominal)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
if optimize
    @time Z̄ = solve(prob_nominal, copy(Z0))
    @save joinpath(@__DIR__, "sol.jld2") Z̄
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol.jld2") Z̄
end

X, U = unpack(Z̄, prob_nominal.prob)
@show sum([U[t][end] for t = 1:T-1])
