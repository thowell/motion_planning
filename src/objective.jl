abstract type Objective end

"""
    quadratic objective
"""
struct QuadraticObjective <: Objective
    Q
    q
    R
    r
    c
end

function quadratic_tracking_objective(Q, R, x, u)
    T = length(Q)
    q = [-2.0 * Q[t] * x[t] for t = 1:T]
    r = [-2.0 * R[t] * u[t] for t = 1:T-1]
    c = [(t == T  ? x[t]' * Q[t] * x[t]
         : x[t]' * Q[t] * x[t] + u[t]' * R[t] * u[t]) for t = 1:T]

    return QuadraticObjective(Q, q, R, r, c)
end

function objective(Z, obj::QuadraticObjective, idx, T)
    J = 0.0

    for t = 1:T
        x = view(Z, idx.x[t])
        Q = obj.Q[t]
        q = obj.q[t]
        c = obj.c[t]

        J += x' * Q * x + q' * x
        J += c

        t == T && continue

        u = view(Z, idx.u[t])
        R = obj.R[t]
        r = obj.r[t]

        J += u' * R * u + r' * u
    end

    return J
end

function objective_gradient!(∇J, Z, obj::QuadraticObjective, idx, T)

    for t = 1:T
        x = view(Z, idx.x[t])
        Q = obj.Q[t]
        q = obj.q[t]
        c = obj.c[t]

        ∇J[idx.x[t]] += 2.0 * Q * x + q

        t == T && continue

        u = view(Z, idx.u[t])
        R = obj.R[t]
        r = obj.r[t]

        ∇J[idx.u[t]] += 2.0 * R * u + r
    end

    return nothing
end

"""
    quadratic time objective
"""
struct QuadraticTimeObjective{T} <: Objective
    obj::QuadraticObjective
    c::T
end

function quadratic_time_tracking_objective(Q, R, x, u, c)
    return QuadraticTimeObjective(quadratic_tracking_objective(Q, R, x, u),c)
end

function objective(Z, obj::QuadraticTimeObjective, idx, T)
    J = 0.0

    for t = 1:T-1
        x = view(Z, idx.x[t])
        Q = obj.obj.Q[t]
        q = obj.obj.q[t]

        u = view(Z, idx.u[t])
        R = obj.obj.R[t]
        r = obj.obj.r[t]

        h = u[end]

        c = obj.obj.c[t]

        J += (x' * Q * x + q' * x) * h
        J += (u' * R * u + r' * u) * h
        J += (c + obj.c) * h
    end

    x = view(Z, idx.x[T])
    Q = obj.obj.Q[T]
    q = obj.obj.q[T]

    J += (x' * Q * x + q' * x)

    return J
end

function objective_gradient!(∇J, Z, obj::QuadraticTimeObjective, idx, T)

    J = 0.0

    for t = 1:T-1
        x = view(Z, idx.x[t])
        Q = obj.obj.Q[t]
        q = obj.obj.q[t]

        u = view(Z, idx.u[t])
        R = obj.obj.R[t]
        r = obj.obj.r[t]

        h = u[end]

        c = obj.obj.c[t]

        ∇J[idx.x[t]] += (2.0 * Q * x + q) * h
        ∇J[idx.u[t]] += (2.0 * R * u + r) * h
        ∇J[idx.u[t][end]] += x' * Q * x + q' * x + u' * R * u + r' * u + c + obj.c
    end

    x = view(Z, idx.x[T])
    Q = obj.obj.Q[T]
    q = obj.obj.q[T]

    ∇J[idx.x[T]] += (2.0 * Q * x + q)

    return nothing
end

"""
    penalty objective
"""
struct PenaltyObjective{T} <: Objective
    α::T
    idx::Int
end

function objective(Z, obj::PenaltyObjective, idx, T)
    J = 0.0
    for t = 1:T-1
        s = Z[idx.u[t][obj.idx]]
        J += s
    end
    return obj.α * J
end

function objective_gradient!(∇J, Z, obj::PenaltyObjective, idx, T)
    for t = 1:T-1
        ∇J[idx.u[t][obj.idx]] += obj.α
    end
    return nothing
end

"""
    multiple objectives
"""
struct MultiObjective <: Objective
    obj::Vector{Objective}
end

function objective(Z, obj::MultiObjective, idx, T)
    return sum([objective(Z, o, idx, T) for o in obj.obj])
end

function objective_gradient!(∇J, Z, obj::MultiObjective, idx, T)
    for o in obj.obj
        objective_gradient!(∇J, Z, o, idx, T)
    end
    return nothing
end


function objective(Z, prob::TrajectoryOptimizationProblem)
    objective(Z, prob.obj, prob.idx, prob.T)
end

function objective_gradient!(∇J, Z, prob::TrajectoryOptimizationProblem)
    objective_gradient!(∇J, Z, prob.obj, prob.idx, prob.T)
end
