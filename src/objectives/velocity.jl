"""
    finite-difference velocity objective
"""
struct VelocityObjective <: Objective
    Q
    n
    h
end

velocity_objective(Q, n; h = 0.0) = VelocityObjective(Q, n, h)

function objective(Z, obj::VelocityObjective, idx, T)
    n = obj.n

    J = 0.0

    for t = 1:T-1
        q⁻ = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
        h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
        v = (q⁺ - q⁻) ./ h

        J += v' * obj.Q[t] * v
    end

    return J
end

function objective_gradient!(∇J, Z, obj::VelocityObjective, idx, T)

    n = obj.n

    for t = 1:T-1
        q⁻ = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
        h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
        v = (q⁺ - q⁻) ./ h

        dJdv = 2.0 * obj.Q[t] * v
        ∇J[idx.x[t][n .+ (1:n)]] += -1.0 ./ h * dJdv
        ∇J[idx.x[t + 1][n .+ (1:n)]] += 1.0 ./ h * dJdv
        if obj.h == 0.0
            ∇J[idx.u[t][end]] += -1.0 * dJdv' * v ./ h
        end
    end

    return nothing
end
