include(joinpath(pwd(), "src/models/particle.jl"))
include(joinpath(pwd(), "src/constraints/constraints_contact.jl"))

# Horizon
T = 51

# Time step
tf = 5.0
h = tf / (T-1)

# Bounds
_uu = Inf * ones(model.m)
_uu[model.idx_u] .= 0.0
_ul = zeros(model.m)
ul, uu = control_bounds(model, T, _ul, _uu)

# Initial and final states
q1 = [0.0, 0.0, 1.0]
v1 = [3.0, 5.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q2]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = PenaltyObjective(1000.0, model.m)

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
X0 = [0.01 * rand(model.n) for t = 1:T] #linear_interp(x1, x1, T) # linear interpolation on state
U0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z̄ = solve(prob, copy(Z0), tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model, state_to_configuration(X̄), Δt = h)
