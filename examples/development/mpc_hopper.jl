import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()

using TrajectoryOptimization
using Altro
using RobotDynamics
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Random
const TO = TrajectoryOptimization
const RD = RobotDynamics

using JLD2
@load joinpath(@__DIR__, "hopper_vertical_gait.jld2") x̄ _ū h̄

function Altro.initialize!(solver::Altro.iLQRSolver)
	Altro.reset!(solver)
    Altro.set_verbosity!(solver)
    Altro.clear_cache!(solver)

    solver.ρ[1] , info = solver.opts.bp_reg_initial
    solver.dρ[1] = 0.0

    # Initial rollout
    rollout!(solver)
	# @warn "skip initial rollout!"
    TO.cost!(solver.obj, solver.Z)
end
# Model and discretization
include(joinpath(@__DIR__, "hopper.jl"))

n, m = size(model)

# Horizon
# _T = length(x̄)
T = length(x̄) #2 * length(x̄) - 1

# Time step
tf = sum(h̄)
h = h̄[1]

# Objective
q1 = x̄[1][1:model.nq]
x1 = [q1; q1; 0.0]
xT = [q1; q1; 0.0]
# X0 = [[[x̄[t]; 0.0] for t = 1:_T]..., [[x̄[t]; 0.0] for t = 2:_T]...]
# U0 = [deepcopy(_ū)..., deepcopy(_ū)...]
X0 = [[x̄[t]; 0.0] for t = 1:T]
U0 = deepcopy(_ū)

Q = Diagonal(10.0 * @SVector ones(n))
R = Diagonal(0.1 * @SVector ones(m))
obj = LQRObjective(Q, R, 1.0 * Q, xT, T)

# Constraints
include(joinpath(@__DIR__, "contact_constraints.jl"))
cons = ConstraintList(n, m, T)

add_constraint!(cons, GoalConstraint(xT, (1:2 * model.nq)), T)
add_constraint!(cons, BoundConstraint(n, m,
    x_min = [model.qL; model.qL; 0.0],
    x_max = [model.qU; model.qU; Inf],
    u_min = [-10.0 * ones(model.nu); zeros(m - model.nu)],
    u_max = [10.0 * ones(model.nu); Inf * ones(m - model.nu)]), 1:T-1)
add_constraint!(cons, SD(n, model.nc, model), 1:T)
add_constraint!(cons, IC(n, model.nc, model), 2:T)
add_constraint!(cons, FC(m, model.nc, model), 1:T-1)
add_constraint!(cons, NS(n, model.nc, model, h), 2:T)

# Create and solve problem
opts , info = solverOptions(
    cost_tolerance_intermediate = 1.0e-2,
    penalty_scaling = 10.0,
    penalty_initial = 1.0,
    projected_newton = false,
    constraint_tolerance = 1.0e-3,
    iterations = 500,
    iterations_inner = 100,
    iterations_linesearch = 50,
    iterations_outer = 20)

prob = Problem(model, obj, xT, tf,
    U0 = U0,
    X0 = X0,
    dt = h,
    x0 = x1, constraints = cons, integration = PassThrough)

solver = ALTROSolver(prob, opts, verbose = 2)
cost(solver)           # initial cost
@time solve!(solver)   # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)

using Plots
plot(hcat(X...)')
plot(hcat(U...)', linetype=:steppost)

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(X), Δt = h)

# ## Update reference trajectory
prob = Problem(model, obj, xT, tf,
    U0 = U,
    X0 = X,
    dt = h,
    x0 = x1, constraints = cons, integration = PassThrough)

solver = ALTROSolver(prob, opts, verbose = 2)
cost(solver)           # initial cost
@time solve!(solver)   # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)

Z_shift = deepcopy(TO.get_trajectory(solver))
Z_shift_init = deepcopy(TO.get_trajectory(solver))

x_shift = 0.1
for t = 1:T
    if t > 1
        RD.set_state!(Z_shift_init[t], states(Z_shift_init)[t] + [x_shift; 0.0; 0.0; 0.0; x_shift; 0.0; 0.0; 0.0; 0.0])
    end
    RD.set_state!(Z_shift[t], states(Z_shift)[t] + [x_shift; 0.0; 0.0; 0.0; x_shift; 0.0; 0.0; 0.0; 0.0])
end

Z_track = TO.Traj([TO.get_trajectory(solver)[1:end]...,
    TO.get_trajectory(solver)[2:end]...,
    TO.get_trajectory(solver)[2:end]...,
    TO.get_trajectory(solver)[2:end]...,
    TO.get_trajectory(solver)[2:end]...])

# Z_track = TO.Traj([TO.get_trajectory(solver)[1:end]...,
#     TO.get_trajectory(solver)[2:end]...,
#     Z_shift_init[2:end]...,
#     Z_shift[2:end]...,
#     Z_shift[2:end]...])
X_track = states(Z_track)
U_track = controls(Z_track)

using Plots
plot(hcat(X_track[1:4:end]...)[1:model.nq, :]',
    labels=["x" "z" "t" "r"],
    width = 2.0, legend = :right)
plot(hcat(U_track[1:4:end]...)[1:model.nu, :]',
    linetype=:steppost,
    labels = ["orientation torque" "length force"],
    width = 2.0)

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(X_track), Δt = h)

## Model-predictive control tracking problem
include("../mpc.jl")

function run_hopper_MPC(prob_mpc, opts_mpc, Z_track,
                            num_iters = length(Z_track) - prob_mpc.N)
    solver_mpc = ALTROSolver(prob_mpc, opts_mpc, verbose = 2)

    # Solve initial iteration
    Altro.solve!(solver_mpc)

    iters = zeros(Int, num_iters,2)
    times = zeros(num_iters,2)

    # Get the problem state size and control size
    n, m = size(prob_mpc)
    dt = Z_track[1].dt
    t0 = 0
    k_mpc = 1
    x0 = SVector(prob_mpc.x0)
    X_traj = [copy(x0) for k = 1:num_iters+1]

    # Begin the MPC LOOP
    for i = 1:num_iters
        # Update initial time
		global model_dist = 0.0 * randn(model.nq)
        t0 += dt
        k_mpc += 1
        TO.set_initial_time!(prob_mpc, t0)

        # Update initial state by using 1st control, and adding some noise
        x0 = discrete_dynamics(TO.integration(prob_mpc),
                                    prob_mpc.model, prob_mpc.Z[1])

        # Update the initial state after the dynamics are propogated.
        TO.set_initial_state!(prob_mpc, x0)
        X_traj[i+1] = x0

        # Update tracking cost
        TO.update_trajectory!(prob_mpc.obj, Z_track, k_mpc)

        # Shift the initial trajectory
        RD.shift_fill!(prob_mpc.Z)

        # Shift the multipliers and penalties
        Altro.shift_fill!(TO.get_constraints(solver_mpc))

        # Solve the updated problem
        Altro.solve!(solver_mpc)

        if Altro.status(solver_mpc) != Altro.SOLVE_SUCCEEDED
            println(Altro.status(solver_mpc))
            @warn ("Solve not succeeded at iteration $i")
            return X_traj, Dict(:time=>times, :iter=>iters)
        end

        # Log the results and performance
        iters[i,1] = iterations(solver_mpc)

        times[i,1] , info = solver_mpc.stats.tsolve
    end

    @warn "solve success"
    return X_traj, Dict(:time=>times, :iter=>iters)
end

Random.seed!(1)
opts_mpc , info = solverOptions(
    cost_tolerance = 1.0e-4,
    cost_tolerance_intermediate = 1.0e-4,
    constraint_tolerance = 1.0e-2,
    reset_duals = false,
    penalty_initial = 1000.0,
    penalty_scaling = 10.0,
    projected_newton = false,
    iterations = 50)

T_mpc = 101
prob_mpc = gen_tracking_problem(prob, T_mpc,
    Q = Diagonal(SVector{n}([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])),
    R = Diagonal(SVector{m}([1.0, 1.0, 1.0, 1.0, 1.0])),
    Qf = Diagonal(SVector{n}([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0])))
Z_track
N_mpc = 301
X_traj, res = run_hopper_MPC(prob_mpc, opts_mpc, Z_track, N_mpc)

plot(hcat(X_track[1:2:N_mpc]...)[model.nq .+ (1:model.nq), :]',
    labels = "", legend = :bottomleft,
    width = 2.0, color = ["red" "green" "blue" "orange"])
plot!(hcat(X_traj[1:2:N_mpc]...)[model.nq .+ (1:model.nq), :]',
    labels = "", legend = :bottom,
    width = 1.0, color = ["red" "green" "blue" "orange"], linestyle = :dash)

vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(X_traj), Δt = h)
