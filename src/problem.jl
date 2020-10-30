abstract type Problem end

struct TrajectoryOptimizationProblem <: Problem
    T::Int # planning horizon

    N::Int      # number of decision variables

    M::Int      # number of constraints

    xl     # state lower bound
    xu     # state upper bound

    ul     # control lower bound
    uu     # control upper bound

    idx    # indices

    model  # model

    h      # time step

    obj    # objective

    con    # constraints
end

function problem(model, obj, T;
        xl = [-1.0 * Inf * ones(model.n) for t = 1:T],
        xu = [Inf * ones(model.n) for t = 1:T],
        ul = [-1.0 * Inf * ones(model.m) for t = 1:T-1],
        uu = [Inf * ones(model.m) for t = 1:T-1],
        h = 0.0,
        con = EmptyConstraints(),
        dynamics = true)

    # indices
    idx = init_indices(model.n, model.m, T)

    # decision variables
    N_state = model.n * T
    N_control = model.m * (T - 1)
    N = N_state + N_control

    # constraints
    dynamics && (con = add_constraints(con, dynamics_constraints(model, T)))

    M = con.n

    prob = TrajectoryOptimizationProblem(
            T,
            N, M,
            xl, xu,
            ul, uu,
            idx,
            model,
            h,
            obj,
            con)

   prob_moi = moi_problem(prob)

   return prob_moi
end

function pack(X, U, prob::TrajectoryOptimizationProblem)
    T = prob.T
    idx = prob.idx

    Z = zeros(prob.N)

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
        prob.N,
        prob.M,
        primal_bounds(prob),
        constraint_bounds(prob),
        prob)
end


function primal_bounds(prob::TrajectoryOptimizationProblem)
    T = prob.T
    idx = prob.idx

    Zl = -Inf * ones(prob.N)
    Zu = Inf * ones(prob.N)

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
    T = prob.T
    M = prob.M

    cl = zeros(M)
    cu = zeros(M)

    cu[prob.con.ineq] .= Inf

    return cl, cu
end

function eval_objective(prob::TrajectoryOptimizationProblem, Z)
    objective(Z, prob)
end

function eval_objective_gradient!(∇l, Z, prob::TrajectoryOptimizationProblem)
    ∇l .= 0.0
    objective_gradient!(∇l, Z, prob)
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
