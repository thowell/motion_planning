# Model
include_model("double_pendulum")
include_constraints("free_time")

model = free_time_model(model)

function fd(model::DoublePendulum, x⁺, x, u, w, h, t)
    midpoint_implicit(model, x⁺, x, u, w, u[end])
end

# Horizon
T = 50
Tm = convert(Int, floor(T / 2))

# Time step
tf = 5.0
h = tf / (T - 1)

# Bounds

# ul <= u <= uu
ul, uu = control_bounds(model, T,
        [-50.0 * ones(model.m - 1); 0.01],
        [50.0 * ones(model.m - 1); 2.0 * h])

# Initial and final states
θ1 = pi / 3.0
θM = pi / 6.0
x1 = [pi / 2.0 + θ1; -2.0 * θ1; 0.0; 0.0]
xM = [pi / 2.0 + θM; -2.0 * θM; 0.0; 0.0]
xl, xu = state_bounds(model, T,
        x1 = [x1[1:2]; Inf * ones(2)])
xl[Tm][1:2] = copy(xM[1:2])
xu[Tm][1:2] = copy(xM[1:2])

# Trajectory initialization
x0 = [linear_interpolation(x1, xM, Tm)..., linear_interpolation(xM, x1, Tm)...] # linear interpolation on state
u0 = [[0.001 * rand(model.m - 1); h] for t = 1:T-1] # random controls

# Objective
include_objective("nonlinear_stage")
function lz_stage(x, u, t)
    if t > 1
        return 100.0 * (kinematics_ee(model, x)[2])^2.0
    else
        return 0.0
    end
end
lz_terminal(x) = 0.0
obj_z = nonlinear_stage_objective(lz_stage, lz_terminal)

obj_track = quadratic_time_tracking_objective(
        [Diagonal([1.0 * ones(2); zeros(2)]) for t = 1:T],
        [Diagonal(1.0e-1 * ones(model.m)) for t = 1:T-1],
        [x0[t] for t = 1:T],
        [zeros(model.m) for t = 1:T],
        1.0)
obj = MultiObjective([obj_z, obj_track])

# Constraints
include(joinpath(pwd(), "src/constraints/loop.jl"))
con_loop = loop_constraints(model, 1, T)
con_free_time = free_time_constraints(T)
con = multiple_constraints([con_free_time, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
           obj,
           T,
           # h = h,
           xl = xl,
           xu = xu,
           ul = ul,
           uu = uu,
           con = con)

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

# Solve
@time z = solve(prob, copy(z0))

# Visualize
using Plots
x, u = unpack(z, prob)
_u = [u[t][1:2] for t = 1:T-1]
tf, t, _h = get_time(u)

plot(hcat(x...)', width = 2.0)
plot(hcat(u...)', width = 2.0, linetype = :steppost)

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, x, Δt = h)

# LQR
Q = [Diagonal(ones(model.n)) for t = 1:T]
R = [Diagonal(0.1 * ones(model.m)) for t = 1:T-1]
K, P = tvlqr(model, x, u, Q, R, 0.0)

plot(hcat([vec(K[t]) for t = 1:T-1]...)',
    linetype = :steppost,
    width = 2.0,
    labels = "",
    title = "LQR gains",
    xlabel = "time step t")

plot(hcat([vec(P[t]) for t = 1:T-1]...)',
    linetype = :steppost,
    width = 2.0,
    labels = "",
    title = "LQR cost-to-go",
    xlabel = "time step t")

N = 5
TN = N * T - (N - 1)
xN = deepcopy(x)
uN = deepcopy(u)
for i = 1:(N - 1)
    xN = [xN..., x[2:end]...]
    uN = [uN..., u...]
end
QN = [Diagonal(ones(model.n)) for t = 1:TN]
RN = [Diagonal(0.1 * ones(model.m)) for t = 1:TN-1]
KN, PN = tvlqr(model, xN, uN, QN, RN, 0.0)

plot(hcat([vec(KN[t]) for t = 1:(TN - 1)]...)',
    linetype = :steppost,
    width = 2.0,
    title = "LQR gains ($N cycles)",
    xlabel = "time step t",
    labels = "")

plot(hcat([vec(PN[t]) for t = 1:(TN - 1)]...)',
    linetype = :steppost,
    width = 2.0,
    title = "LQR cost-to-go ($N cycles)",
    xlabel = "time step t",
    labels = "")

ctg = PN[1:T]
@save joinpath(@__DIR__, "double_pendulum_limit_cycle.jld2") x _u _h ctg
