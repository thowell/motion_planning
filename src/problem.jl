abstract type Problem end

struct TrajectoryOptimizationProblem <: Problem
    model  # model

    h      # time step
    T::Int # planning horizon

    obj    # objective

    con    # constraints
    xl     # state lower bound
    xu     # state upper bound
    ul     # control lower bound
    uu     # control upper bound

    num_var::Int      # number of decision variables
    num_con::Int      # number of constraints

    idx    # indices
end

function trajectory_optimization_problem(model, obj, T;
        h = 0.0,
        xl = [-1.0 * Inf * ones(model.n) for t = 1:T],
        xu = [Inf * ones(model.n) for t = 1:T],
        ul = [-1.0 * Inf * ones(model.m) for t = 1:T-1],
        uu = [Inf * ones(model.m) for t = 1:T-1],
        con = EmptyConstraints(),
        dynamics = true)

    # indices
    idx = indices(model.n, model.m, T)

    # decision variables
    num_state = model.n * T
    num_control = model.m * (T - 1)
    num_var = num_state + num_control

    # constraints
    dynamics && (con = add_constraints(con, dynamics_constraints(model, T)))

    num_con = con.n

    prob = TrajectoryOptimizationProblem(
            model,
            h,
            T,
            obj,
            con, xl, xu, ul, uu,
            num_var, num_con,
            idx
            )

   return prob
end

function problem(model, obj, T;
        xl = [-1.0 * Inf * ones(model.n) for t = 1:T],
        xu = [Inf * ones(model.n) for t = 1:T],
        ul = [-1.0 * Inf * ones(model.m) for t = 1:T-1],
        uu = [Inf * ones(model.m) for t = 1:T-1],
        h = 0.0,
        con = EmptyConstraints(),
        dynamics = true)

   prob = trajectory_optimization_problem(model, obj, T,
                h = h,
                xl = xl,
                xu = xu,
                ul = ul,
                uu = uu,
                con = con,
                dynamics = dynamics)

   prob_moi = moi_problem(prob)

   return prob_moi
end

function pack(X, U, prob::TrajectoryOptimizationProblem)
    T = prob.T
    idx = prob.idx

    Z = zeros(prob.num_var)

    for t = 1:T
        Z[idx.x[t]] = X[t]

        t == T && continue

        Z[idx.u[t]] = U[t]
    end

    return Z
end

function unpack(Z, prob::TrajectoryOptimizationProblem)
    T = prob.T
    idx = prob.idx

    X = [Z[idx.x[t]] for t = 1:T]
    U = [Z[idx.u[t]] for t = 1:T-1]

    return X, U
end


function moi_problem(prob::TrajectoryOptimizationProblem)
    return MOIProblem(
        prob.num_var,
        prob.num_con,
        primal_bounds(prob),
        constraint_bounds(prob),
        prob)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)
    T = prob.T
    idx = prob.idx

    Zl = -Inf * ones(prob.num_var)
    Zu = Inf * ones(prob.num_var)

    for t = 1:T
        Zl[idx.x[t]] = prob.xl[t]
        Zu[idx.x[t]] = prob.xu[t]

        t == T && continue

        Zl[idx.u[t]] = prob.ul[t]
        Zu[idx.u[t]] = prob.uu[t]
    end

    return Zl, Zu
end

function constraint_bounds(prob::TrajectoryOptimizationProblem)

    cl = zeros(prob.num_con)
    cu = zeros(prob.num_con)

    cu[prob.con.ineq] .= Inf

    return cl, cu
end

function eval_objective(prob::TrajectoryOptimizationProblem, Z)
    objective(Z, prob)
end

function eval_objective_gradient!(∇J, Z, prob::TrajectoryOptimizationProblem)
    ∇J .= 0.0
    objective_gradient!(∇J, Z, prob)
    return nothing
end

function eval_constraint!(c, Z, prob::TrajectoryOptimizationProblem)
    constraints!(c, Z, prob)
    return nothing
end

function eval_constraint_jacobian!(∇c, Z, prob::TrajectoryOptimizationProblem)
    constraints_jacobian!(∇c, Z, prob)
    return nothing
end

function sparsity_jacobian(prob::TrajectoryOptimizationProblem)
    constraints_sparsity(prob)
end

"""
    objective
"""
objective(Z, prob::TrajectoryOptimizationProblem) = objective(Z, prob.obj, prob.idx, prob.T)
objective_gradient!(∇J, Z, prob::TrajectoryOptimizationProblem) = objective_gradient!(∇J, Z, prob.obj, prob.idx, prob.T)

"""
    constraints
"""
constraints!(c, Z, prob::TrajectoryOptimizationProblem) = constraints!(c, Z, prob.con, prob.model, prob.idx, prob.h, prob.T)
constraints_jacobian!(∇c, Z, prob::TrajectoryOptimizationProblem) = constraints_jacobian!(∇c, Z, prob.con, prob.model, prob.idx, prob.h, prob.T)
constraints_sparsity(prob::TrajectoryOptimizationProblem; shift_row = 0, shift_col = 0) = constraints_sparsity(prob.con, prob.model, prob.idx, prob.T; shift_row = shift_row, shift_col = shift_col)
