struct ControlComplementarity <: Constraints
    n
    ineq
    n_stage
end

function constraints!(c, Z, con::ControlComplementarity, model, idx, h, T)
    for t = 1:T-1
        x⁺ = view(Z, idx.x[t+1])
        q⁺ = view(x⁺, model.nq .+ (1:model.nq))
        u = view(Z, idx.u[t])

        c[(t - 1) * con.n_stage .+ (1:1)] = u[model.idx_s] - u[model.idx_u][1] * ϕ_func(model, q⁺)
        c[(t - 1) * con.n_stage + 1 .+ (1:1)] = u[model.idx_s] - u[model.idx_u][2] * ϕ_func(model, q⁺)
    end
    nothing
end

function constraints_jacobian!(∇c, Z, con::ControlComplementarity, model, idx, h, T)

    shift = 0

    for t = 1:T-1
        x⁺ = view(Z, idx.x[t+1])
        q⁺ = view(x⁺, model.nq .+ (1:model.nq))
        u = view(Z, idx.u[t])

        cq1(y) = u[model.idx_s] - u[model.idx_u][1] * ϕ_func(model, y)
        cu1(y) = y[model.idx_s] - y[model.idx_u][1] * ϕ_func(model, q⁺)
        cq2(y) = u[model.idx_s] - u[model.idx_u][2] * ϕ_func(model, y)
        cu2(y) = y[model.idx_s] - y[model.idx_u][2] * ϕ_func(model, q⁺)

        r_idx = (t - 1) * con.n_stage .+ (1:1)

        c_idx = idx.x[t+1][model.nq .+ (1:model.nq)]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            1, model.nq),
            cq1, q⁺)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            1, model.m),
            cu1, u)
        shift += len

        r_idx = (t - 1) * con.n_stage + 1 .+ (1:1)

        c_idx = idx.x[t+1][model.nq .+ (1:model.nq)]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            1, model.nq),
            cq2, q⁺)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            1, model.m),
            cu2, u)
        shift += len
    end

    return nothing
end

function constraints_sparsity(con::ControlComplementarity, model, idx, T;
    shift_row = 0, shift_col = 0)
    row = []
    col = []

    for t = 1:T-1

        r_idx = shift_row + (t - 1) * con.n_stage .+ (1:1)

        c_idx = shift_col .+ idx.x[t+1][model.nq .+ (1:model.nq)]
        row_col!(row, col, r_idx, c_idx)

        c_idx = shift_col .+ idx.u[t]
        row_col!(row, col, r_idx, c_idx)

        r_idx = shift_row + (t - 1) * con.n_stage + 1 .+ (1:1)

        c_idx = shift_col .+ idx.x[t+1][model.nq .+ (1:model.nq)]
        row_col!(row, col, r_idx, c_idx)

        c_idx = shift_col .+ idx.u[t]
        row_col!(row, col, r_idx, c_idx)
    end

    return collect(zip(row, col))
end
