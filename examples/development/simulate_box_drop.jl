include(joinpath(pwd(), "models/box.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))

# Initial state
mrp = MRP(UnitQuaternion(RotY(pi / 10.0) * RotX(pi / 15.0)))
q1 = [0.0, 0.0, 2.5, mrp.x, mrp.y, mrp.z]
v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
v2 = v1 - G_func(model,q1) * h
q2 = q1 + 0.5 * h * (v1 + v2)

# Simulate contact model one step
function step_contact(model, x1, u1, h)
    # Horizon
    T = 2

    # Objective
    obj = PenaltyObjective(1.0e5, model.m)

    # Constraints
    con_contact = contact_constraints(model, T)

    # Bounds
    _uu = Inf * ones(model.m)
    _uu[model.idx_u] .= u1
    _ul = zeros(model.m)
    _ul[model.idx_u] .= u1
    ul, uu = control_bounds(model, T, _ul, _uu)

    xl, xu = state_bounds(model, T, x1 = x1)

    # Problem
    prob = trajectory_optimization_problem(model,
                   obj,
                   T,
                   h = h,
                   xl = xl,
                   xu = xu,
                   ul = ul,
                   uu = uu,
                   con = con_contact)

    # Trajectory initialization
    x0 = [x1 for t = 1:T]
    u0 = [1.0e-5 * rand(model.m) for t = 1:T-1] # random controls

    # Pack trajectories into vector
    z0 = pack(x0, u0, prob)

    @time z , info = solve(prob, copy(z0), tol = 1.0e-5, c_tol = 1.0e-5)

    @assert check_slack(z, prob) < 1.0e-4
    x, u = unpack(z, prob)

    return x[end]
end

x = [x1]
for t = 1:200
    push!(x, step_contact(model, x[end], zeros(model.nu), h))
    println("step $t")
end

include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)
visualize!(vis, model, state_to_configuration(x), Δt = h)
