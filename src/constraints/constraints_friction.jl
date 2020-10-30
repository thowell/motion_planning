"""
	friction
"""
struct FrictionConstraints <: Constraints
    n
    ineq
    n_stage
end

function friction!(c, x⁺, u, model)

    v = x⁺[3]
    β = u[2:3]
    ψ = u[4]
    η = u[5:6]
    s = u[7]

    n = (model.mc + model.mp) * model.g

    c[1] = v + ψ - η[1]
    c[2] = -v + ψ - η[2]
    c[3] = model.μ * n - sum(β)
    c[4] = s - ψ * (model.μ * n - sum(β))
    c[5] = s - β' * η

    return nothing
end

function constraints!(c, Z, con::FrictionConstraints, model, idx, h, T)
    for t = 1:T-1
        x⁺ = view(Z, idx.x[t + 1])
        u = view(Z, idx.u[t])
        friction!(view(c, (t - 1) * con.n_stage .+ (1:con.n_stage)), x⁺, u, model)
    end

    nothing
end

function constraints_jacobian!(∇c, Z, con::FrictionConstraints, model, idx, h, T)
    n = model.n
    m = model.m

    c_tmp = zeros(con.n_stage)
    shift = 0

    for t = 1:T-1
        x⁺ = view(Z, idx.x[t + 1])
        u = view(Z, idx.u[t])

        cx!(c, z) = friction!(c, z, u, model)
        cu!(c, z) = friction!(c, x⁺, z, model)

        r_idx = (t - 1) * con.n_stage .+ (1:con.n_stage)

        c_idx = idx.x[t + 1]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)), con.n_stage, n),
            cx!, c_tmp, x⁺)
        shift += len

        c_idx = idx.u[t]
        len = length(r_idx) * length(c_idx)
        ForwardDiff.jacobian!(reshape(view(∇c, shift .+ (1:len)), con.n_stage, m),
            cu!, c_tmp, u)
        shift += len
    end

    return nothing
end

function constraints_sparsity(con::FrictionConstraints, model, idx, T; r_shift = 0)
    row = []
    col = []

    for t = 1:T-1
        r_idx = r_shift + (t - 1) * con.n_stage .+ (1:con.n_stage)

        c_idx = idx.x[t + 1]
        row_col!(row, col, r_idx, c_idx)

        c_idx = idx.u[t]
        row_col!(row, col, r_idx, c_idx)
    end

    return collect(zip(row, col))
end
