# Model
include_model("rocket")
model_sl = free_time_model(additive_noise_model(model_slosh))

# Horizon
T = 41

# Bounds
ul, uu = control_bounds(model_sl, T, [-5.0; 0.0; 0.01], [5.0; 100.0; 1.0])

# Initial and final states
x1_slosh = [15.0;
	  model_sl.l1 + 10.0;
	  -45.0 * pi / 180.0;
	  0.0;
	  -10.0;
	  -10.0;
	  -1.0 * pi / 180.0;
	  0.0]

xT_slosh = [0.0;
	  model_sl.l1;
	  0.0;
	  0.0;
	  0.0;
	  0.0;
	  0.0;
	  0.0]

# Radius of landing pad
r_pad = 0.25

# xl <= x <= xl
_xl_slosh = -Inf * ones(model_sl.n)
_xl_slosh[2] = model_sl.l1
_xu_slosh = Inf * ones(model_sl.n)

xl_slosh, xu_slosh = state_bounds(model_sl, T, _xl_slosh, _xu_slosh,
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
	[Diagonal([1.0e-1 * ones(2); 0.0]) for t = 1:T-1],
    [xT_slosh for t = 1:T],
	[zeros(model_sl.m) for t = 1:T],
	1.0)

# Time step constraints
con_free_time = free_time_constraints(T)

# Problem
prob_slosh = trajectory_optimization_problem(model_sl,
					obj_slosh,
					T,
                    xl = xl_slosh,
                    xu = xu_slosh,
                    ul = ul,
                    uu = uu,
                    con = con_free_time)

# Trajectory initialization
x0_slosh = linear_interpolation(x1_slosh, xT_slosh, T) # linear interpolation on state
u0 = [ones(model_sl.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0_slosh = pack(x0_slosh, u0, prob_slosh)

# Solve
optimize = true

if optimize
	include_snopt()
    @time z̄_slosh , info = solve(prob_slosh, copy(z0_slosh),
		nlp = :SNOPT7,
		time_limit = 60 * 10)
    @save joinpath(@__DIR__, "sol_slosh.jld2") z̄_slosh
else
    println("Loading solution...")
    @load joinpath(@__DIR__, "sol_slosh.jld2") z̄_slosh
end

x̄_slosh, ū_slosh = unpack(z̄_slosh, prob_slosh)
@show sum([ū_slosh[t][end] for t = 1:T-1]) # ~2.74
