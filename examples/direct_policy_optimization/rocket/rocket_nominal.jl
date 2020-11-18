include(joinpath(pwd(),"models/rocket.jl"))
include(joinpath(pwd(),"src/constraints/free_time.jl"))

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
	[zeros(model_nom.m) for t = 1:T-1],
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
                    con = con_free_time)

# Trajectory initialization
x0 = linear_interp(x1, xT, T) # linear interpolation on state
u0 = [ones(model_nom.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob_nominal)


# Solve
optimize = true

if optimize
	include_snopt()
    @time z̄_nom = solve(prob_nominal, copy(z0),
		nlp = :SNOPT7,
		time_limit = 60 * 10)
    @save joinpath(@__DIR__, "sol_nom.jld2") z̄_nom
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_nom.jld2") z̄_nom
end

x̄_nom, ū_nom = unpack(z̄_nom, prob_nominal.prob)
@show sum([ū_nom[t][end] for t = 1:T-1]) # ~2.72
