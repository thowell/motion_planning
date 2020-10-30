struct PinnedFoot <: Constraints
    n
    ineq
    q_pinned
end

function pinned_foot_constraint(model, q_pinned, T)
    n = 2 * (T + 1)
    ineq = (1:0)
    PinnedFoot(n, ineq, q_pinned)
end

function constraints!(c, Z, con::PinnedFoot, model, idx, h, T)
    k(y) = kinematics_2(model, y, body = :leg_1, mode = :ee)

    for t = 1:T
        if t == 1
            q⁻ = view(Z, idx.x[t][1:model.nq])
            c[1:2] = k(q⁻) - k(con.q_pinned)
        end
        q = view(Z, idx.x[t][model.nq .+ (1:model.nq)])
        c[2 + (t - 1) * 2 .+ (1:2)] = k(q) - k(con.q_pinned)
    end
    nothing
end

function constraints_jacobian!(∇c, Z, con::PinnedFoot, model, idx, h, T)
    k(y) = kinematics_2(model, y, body = :leg_1, mode = :ee)
    shift = 0
    for t = 1:T
        if t == 1
            q⁻ = view(Z, idx.x[t][1:model.nq])
            r_idx = (1:2)
            c_idx = idx.x[t][1:model.nq]
            len = length(r_idx) * length(c_idx)
            ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(k, q⁻))
            shift += len
        end
        q = view(Z, idx.x[t][model.nq .+ (1:model.nq)])
        r_idx = 2 + (t - 1) * 2 .+ (1:2)
        c_idx = idx.x[t][model.nq .+ (1:model.nq)]
        len = length(r_idx) * length(c_idx)
        ∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(k, q))
        shift += len
    end

    return nothing
end

function constraints_sparsity(con::PinnedFoot, model, idx, T; r_shift = 0)
    row = []
    col = []

    for t = 1:T
        if t == 1
            r_idx = r_shift .+ (1:2)
            c_idx = idx.x[t][1:model.nq]
            row_col!(row, col, r_idx, c_idx)
        end
        r_idx = r_shift + 2 + (t - 1) * 2 .+ (1:2)
        c_idx = idx.x[t][model.nq .+ (1:model.nq)]
        row_col!(row, col, r_idx, c_idx)
    end

    return collect(zip(row, col))
end
