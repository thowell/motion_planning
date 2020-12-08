struct DynamicsConstraints <: Constraints
    n::Int
    ineq
    w
end

function dynamics_constraints(model, T;
    w = [zeros(model.d) for t = 1:T-1])

    n = model.n * (T - 1)
    ineq = (1:0)

    return DynamicsConstraints(n, ineq, w)
end

function constraints!(c, Z, con::DynamicsConstraints, model, idx, h, T)
    n = model.n

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])
        x⁺ = view(Z, idx.x[t + 1])

        c[(t-1) * n .+ (1:n)] = fd(model, x⁺, x, u, con.w[t], h, t)
    end

    return nothing
end

function constraints_jacobian!(∇c, Z, con::DynamicsConstraints, model, idx, h, T)
    n = model.n
    m = model.m

    # note: x1 and xT constraints are handled as simple bound constraints
    #       e.g., x1 <= x <= x1, xT <= x <= xT

    shift = 0

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])
        x⁺ = view(Z, idx.x[t + 1])

        dyn_x(z) = fd(model, x⁺, z, u, con.w[t], h, t)
        dyn_u(z) = fd(model, x⁺, x, z, con.w[t], h, t)
        dyn_x⁺(z) = fd(model, z, x, u, con.w[t], h, t)

        r_idx = (t-1) * n .+ (1:n)

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x, x))
        shift += s

        s = n * m
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_u, u))
        shift += s

        s = n * n
        ∇c[shift .+ (1:s)] = vec(ForwardDiff.jacobian(dyn_x⁺, x⁺))
        shift += s
    end

    return nothing
end

function constraints_sparsity(con::DynamicsConstraints, model, idx, T;
    shift_row = 0, shift_col = 0)
    n = model.n

    row = []
    col = []

    for t = 1:T-1
        r_idx = shift_row + (t-1) * n .+ (1:n)
        row_col!(row, col, r_idx, shift_col .+ idx.x[t])
        row_col!(row, col, r_idx, shift_col .+ idx.u[t])
        row_col!(row, col, r_idx, shift_col .+ idx.x[t + 1])
    end

    return collect(zip(row, col))
end
