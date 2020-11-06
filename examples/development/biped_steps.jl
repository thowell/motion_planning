include(joinpath(pwd(), "src/models/biped.jl"))
include(joinpath(pwd(), "src/objectives/velocity.jl"))
include(joinpath(pwd(), "src/objectives/acceleration.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to thigh 1)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to thigh 2)
θ = pi / 10.0
# q1 = initial_configuration(model, θ) # generate initial config from θ
# qT = copy(q1)
# qT[1] += 1.0
# q1[3] -= pi / 30.0
# q1[4] += pi / 20.0
# q1[5] -= pi / 10.0
q1, qT = loop_configurations(model, θ)
qT[1] += 1.0
visualize!(vis, model, [qT])

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 100.0

_ul = zeros(model.m)
_ul[model.idx_u] .= -100.0
ul, uu = control_bounds(model, T, _ul, _uu)

xl, xu = state_bounds(model, T, x1 = [q1; q1])

# Objective
q_ref = linear_interp(q1, qT, T)
X0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e3, model.m)

Qq = 1.0 * Diagonal(ones(model.nq)) # configuration penalty
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2)) # state penalty
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model.nq)), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...]) # no penalty on "contact" controls

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [X0[t] for t = 1:T],
    [zeros(model.m) for t = 1:T]
    )
# quadratic velocity penalty
# Σ v' Q v
obj_velocity = velocity_objective(
    [Diagonal(1.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h)
# quadratic acceleration penalty
# Σ a' Q a
# obj_acceleration = acceleration_objective(
#     [Diagonal(1.0e-1 * ones(model.nq)) for t = 1:T-1],
#     model.nq,
#     h = h)
obj = MultiObjective([obj_tracking,
                      obj_penalty,
                      obj_velocity])#,
                      # obj_acceleration])

# Constraints
con_contact = contact_constraints(model, T)
# con_pinned = pinned_foot_constraint(model, q1, T)
con = multiple_constraints([con_contact])#, con_pinned])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# trajectory initialization
U0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z̄ = solve(prob, copy(Z0),
    nlp = :ipopt,
    tol = 1.0e-3, c_tol = 1.0e-3)#, mapl = 5)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

# using Plots
# plot(hcat(Ū...)[1:model.nu,:]', linetype = :steppost)

# Visualize
visualize!(vis, model, state_to_configuration(X̄), Δt = h)
