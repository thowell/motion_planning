include(joinpath(pwd(), "src/models/miniature_golf.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))
include(joinpath(pwd(), "src/constraints/visualize.jl"))

# Horizon
T = 41

tf = 2.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_ul = -Inf * ones(model.m)
_ul[model.nu .+ (1:m_contact)] .= 0.0

ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
px_init = [0.66; 0.0; softplus(0.66)]
px_goal = [0.66; -3.0; softplus(0.66)]

q1 = [q_res1; px_init]
qT = [q_res2; px_goal]

x1 = [q1; q1]

xl, xu = state_bounds(model, T, [model.qL; model.qL], [model.qU; model.qU],
	x1 = x1)

# Objective
q_ref = linear_interp(q1, qT, T)
x_ref = configuration_to_state(q_ref)
set_configuration!(mvis, q_init)
set_configuration!(state, q_init)
u_ref = [h * Array(RigidBodyDynamics.dynamics_bias(state));
	zeros(model.m - model.nu)]

Qq = Diagonal([10.0 * ones(7); 1.0 * ones(3)])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, Diagonal([10.0 * ones(7); 1000.0 * ones(3)]), dims = (1, 2))
R = Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...])

obj_tracking = quadratic_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [x_ref[t] for t = 1:T],
    [u_ref for t = 1:T]
    )
obj_penalty = PenaltyObjective(1000.0, model.m)
obj = MultiObjective([obj_tracking, obj_penalty])

# Constraints
con_contact = contact_constraints(model, T)

# Problem
prob = problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con_contact
               )


# Trajectory initialization
X0 = deepcopy(x_ref)
U0 = [[u_ref[1:model.nu]; 1.0e-5 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

@time Z̄ = solve(prob, copy(Z0),
	c_tol = 1.0e-3, tol = 1.0e-3, max_iter = 1000)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

# [ϕ_func(model,X̄[t]) for t = 1:T]
# x_nom = [X̄[t][1] for t = 1:T]
# z_nom = [X̄[t][2] for t = 1:T]
# u_nom = [Ū[t][model.idx_u] for t = 1:T-1]
# λ_nom = [Ū[t][model.idx_λ[1]] for t = 1:T-1]
# b_nom = [Ū[t][model.idx_b] for t = 1:T-1]
# ψ_nom = [Ū[t][model.idx_ψ[1]] for t = 1:T-1]
# η_nom = [Ū[t][model.idx_η] for t = 1:T-1]

# plot(hcat(u_nom...)')
# plot(hcat(λ_nom...)', linetype=:steppost)
# plot(hcat(b_nom...)',linetype=:steppost)
# plot(hcat(ψ_nom...)',linetype=:steppost)
# plot(hcat(η_nom...)',linetype=:steppost)
# plot(hcat(U_nom...)',linetype=:steppost)

visualize!(mvis, model, state_to_configuration(X̄), Δt = h)
