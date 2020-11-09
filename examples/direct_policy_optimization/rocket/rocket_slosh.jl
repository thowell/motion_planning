include(joinpath(pwd(),"src/models/rocket.jl"))
include(joinpath(pwd(),"src/constraints/free_time.jl"))

optimize = true

# Free-time model
model_slosh = free_time_model(additive_noise_model(model_slosh))

function fd(model::RocketSlosh, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end]) - w
end

# Horizon
T = 41

# Bounds
ul, uu = control_bounds(model_slosh, T, [-5.0; 0.0; 0.01], [5.0; 100.0; 1.0])

# Initial and final states
x1_slosh = [15.0;
	  model_slosh.l1 + 10.0;
	  -45.0 * pi / 180.0;
	  0.0;
	  -10.0;
	  -10.0;
	  -1.0 * pi / 180.0;
	  0.0]

xT_slosh = [0.0;
	  model_slosh.l1;
	  0.0;
	  0.0;
	  0.0;
	  0.0;
	  0.0;
	  0.0]

# Radius of landing pad
r_pad = 0.25

# xl <= x <= xl
_xl_slosh = -Inf * ones(model_slosh.n)
_xl_slosh[2] = model_slosh.l1
_xu_slosh = Inf * ones(model_slosh.n)

xl_slosh, xu_slosh = state_bounds(model_slosh, T, _xl_slosh, _xu_slosh,
 	x1 = x1_slosh, xT = xT_slosh)

xl_slosh[T][1] = -1.0 * r_pad
xu_slosh[T][1] = r_pad
xl_slosh[T][2] = xT_slosh[2] - 0.001
xu_slosh[T][2] = xT_slosh[2] + 0.001
xl_slosh[T][3] = -1.0 * pi / 180.0
xu_slosh[T][3] = 1.0 * pi / 180.0
xl_slosh[T][4] = -0.001
xu_slosh[T][4] = 0.001
xl_slosh[T][5] = -0.001
xu_slosh[T][5] = 0.001
xl_slosh[T][6] = -0.01 * pi / 180.0
xu_slosh[T][6] = 0.01 * pi / 180.0

# Objective
obj_slosh = quadratic_time_tracking_objective(
	[(t != T ? Diagonal([1.0; 10.0; 1.0; 0.1; 1.0; 10.0; 1.0; 0.1])
		: Diagonal([10.0; 100.0; 10.0; 0.1; 10.0; 100.0; 10.0; 0.1])) for t = 1:T],
	[Diagonal([1.0e-1*ones(2); 0.0]) for t = 1:T-1],
    [xT_slosh for t = 1:T],
	[zeros(model_slosh.m) for t = 1:T],
	1.0)

# Time step constraints
con_free_time = free_time_constraints(T)

# Problem
prob_slosh = trajectory_optimization_problem(model_slosh,
					obj_slosh,
					T,
                    xl = xl_slosh,
                    xu = xu_slosh,
                    ul = ul,
                    uu = uu,
                    con = con_free_time
                    )

# Trajectory initialization
X0_slosh = linear_interp(x1_slosh, xT_slosh, T) # linear interpolation on state
U0 = [ones(model_slosh.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0_slosh = pack(X0_slosh, U0, prob_slosh)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
if optimize
    @time Z̄_slosh = solve(prob_slosh, copy(Z0_slosh))
    @save joinpath(@__DIR__, "sol_slosh.jld2") Z̄_slosh
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_slosh.jld2") Z̄_slosh
end

X̄_slosh, Ū_slosh = unpack(Z̄_slosh, prob_slosh)
@show sum([U[t][end] for t = 1:T-1])
