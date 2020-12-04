# Model
include_model("particle")
include_constraints("contact_no_slip")
model = no_slip_model(model)

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
v1 = [1.0, 0.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

x1 = [q1; q2]

xl, xu = state_bounds(model, T, x1 = x1)

# Objective
obj = PenaltyObjective(1.0e5, model.m)

# Constraints
con_contact = contact_no_slip_constraints(model, T)

# Problem
prob = trajectory_optimization_problem(model,
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
x0 = [0.01 * rand(model.n) for t = 1:T] #linear_interpolation(x1, x1, T) # linear interpolation on state
u0 = [0.001 * rand(model.m) for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time z̄ = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

t = 25
q3 = view(x̄[t], model.nq .+ (1:model.nq))
q2 = view(x̄[t], 1:model.nq)
λ = view(ū[t], model.idx_λ)
s = view(ū[t], model.idx_s)
λ_stack = λ[1] * ones(2)
s[1] - (λ_stack' * _P_func(model, q3) * (q3 - q2) / h)[1]


include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
