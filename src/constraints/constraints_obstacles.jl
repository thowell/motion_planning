"""
    obstacles
"""

struct ObstacleConstraints <: Constraints
    n
    ineq
    n_stage
end

function constraints!(c, Z, con::ObstacleConstraints, model, idx, h, T)
    for t = 1:T
        x = view(Z, idx.x[t])
        obstacles!(view(c, (t - 1) * con.n_stage .+ (1:con.n_stage)), x)
    end
    nothing
end

function constraints_jacobian!(∇c, Z, con::ObstacleConstraints, model, idx, h, T)
    n = model.n

    c_tmp = zeros(con.n_stage)
    shift = 0

    for t = 1:T
        r_idx = (t - 1) * con.n_stage .+ (1:con.n_stage)
        c_idx = idx.x[t]

        len = length(r_idx) * length(c_idx)
        x = view(Z, idx.x[t])

        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)),
            con.n_stage, n),
            obstacles!, c_tmp, x)

        shift += len
    end

    return nothing
end

function constraints_sparsity(con::ObstacleConstraints, model, idx, T; r_shift = 0)
    n = model.n

    row = []
    col = []

    for t = 1:T
        r_idx = r_shift + (t - 1) * con.n_stage .+ (1:con.n_stage)
        c_idx = idx.x[t]

        row_col!(row, col, r_idx, c_idx)
    end

    return collect(zip(row, col))
end

function circle_obs(x, y, xc, yc, r)
    (x - xc)^2.0 + (y - yc)^2.0 - r^2.0
end
