# Model
include_model("hopper")
model_ft = free_time_model(model)

# Horizon
T = 101

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[model_ft.idx_u] .= 10.0
_uu[end] = 2.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -10.0
_ul[end] = 0.5 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.25
q1 = [0.0, 0.5 + z_h, 0.0, 0.25]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; Inf * ones(model.nq)])

# Objective
include_objective("velocity")
obj_tracking = quadratic_time_tracking_objective(
    [Diagonal(zeros(model_ft.n)) for t = 1:T],
    [Diagonal([1.0e-1, 1.0e-1, zeros(model_ft.m - model_ft.nu)...]) for t = 1:T-1],
    [zeros(model_ft.n) for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)
obj_contact_penalty = PenaltyObjective(1.0e5, model_ft.m - 1)
obj_velocity = velocity_objective(
    [Diagonal(10.0 * ones(model_ft.nq)) for t = 1:T-1],
    model_ft.nq,
    h = h,
    idx_angle = collect([3]))
obj = MultiObjective([obj_tracking, obj_contact_penalty, obj_velocity])

# Constraints
include_constraints(["free_time", "contact", "loop"])
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)
con_loop = loop_constraints(model, 1, T)
con = multiple_constraints([con_free_time, con_contact, con_loop])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# Trajectory initialization
x0 = [[q1; q1] for t = 1:T] # linear interpolation on state
u0 = [[1.0e-3 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include_snopt()

optimize = true
if optimize
	@time z̄ = solve(prob, copy(z0),
		nlp = :SNOPT7,
		tol = 1.0e-5, c_tol = 1.0e-5, mapl = 5,
		time_limit = 60)
	@show check_slack(z̄, prob)
	x̄, ū = unpack(z̄, prob)
	_ū = [ū[t][1:5] for t = 1:T-1]
	tf, t, h̄ = get_time(ū)
	@show tf
	@show h̄[1]
	@save joinpath(@__DIR__, "hopper_vertical_gait.jld2") x̄ _ū h̄
else
	@load joinpath(@__DIR__, "hopper_vertical_gait.jld2") x̄ _ū h̄
end

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model_ft, state_to_configuration(x̄), Δt = ū[1][end])

plot(hcat(_ū...)[1:2, :]', linetype = :steppost)
