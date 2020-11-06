"""
    finite-difference acceleration objective
"""
struct AccelerationObjective <: Objective
    Q
    n
    h
end

acceleration_objective(Q, n; h = 0.0) = AccelerationObjective(Q, n, h)

function objective(Z, obj::AccelerationObjective, idx, T)
    n = obj.n

    J = 0.0

    for t = 2:T-1
        # configurations
        q⁻ = view(Z, idx.x[t - 1][n .+ (1:n)])
        q = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])

        # time steps
        h⁻ = obj.h == 0.0 ? view(Z, idx.u[t - 1])[end] : obj.h
        h⁺ = obj.h == 0.0 ? view(Z, idx.u[t])[end] : obj.h

        # velocities
        v⁺ = (q⁺ - q) / h⁺
        v⁻ = (q - q⁻) / h⁻

        # acceleration
        a = (v⁺ - v⁻) / (0.5 * h⁺ + 0.5 * h⁻)

        J += a' * obj.Q[t] * a
    end

    return J
end

function objective_gradient!(∇J, Z, obj::AccelerationObjective, idx, T)

    n = obj.n

    for t = 2:T-1
        # configurations
        q⁻ = view(Z, idx.x[t - 1][n .+ (1:n)])
        q = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])

        # time steps
        h⁻ = obj.h == 0.0 ? view(Z, idx.u[t - 1])[end] : obj.h
        h⁺ = obj.h == 0.0 ? view(Z, idx.u[t])[end] : obj.h

        # velocities
        v⁺ = (q⁺ - q) / h⁺
        v⁻ = (q - q⁻) / h⁻

        # acceleration
        a = (v⁺ - v⁻) / (0.5 * h⁺ + 0.5 * h⁻)

        dJda = 2.0 * obj.Q[t] * a
        dadv⁺ = 1.0 / (0.5 * h⁺ + 0.5 * h⁻)
        dadv⁻ = -1.0 / (0.5 * h⁺ + 0.5 * h⁻)
        dadh⁺ = -0.5 * (v⁺ - v⁻) / (0.5 * h⁺ + 0.5 * h⁻)^2.0
        dadh⁻ = -0.5 * (v⁺ - v⁻) / (0.5 * h⁺ + 0.5 * h⁻)^2.0
        dv⁺dq⁺ = 1.0 / h⁺
        dv⁺dq = -1.0 / h⁺
        dv⁺dh⁺ = -1.0 * v⁺ / h⁺
        dv⁻dq = 1.0 / h⁻
        dv⁻dq⁻ = -1.0 / h⁻
        dv⁻dh⁻ = -1.0 * v⁻ / h⁻

        ∇J[idx.x[t - 1][n .+ (1:n)]] += dJda * dadv⁻ * dv⁻dq⁻
        ∇J[idx.x[t][n .+ (1:n)]] += dJda * (dadv⁺ * dv⁺dq + dadv⁻ * dv⁻dq)
        ∇J[idx.x[t + 1][n .+ (1:n)]] += dJda * dadv⁺ * dv⁺dq⁺
        if obj.h == 0.0
            ∇J[idx.u[t - 1][end]] += dJda' * (dadh⁻ + dadv⁻ * dv⁻dh⁻)
            ∇J[idx.u[t][end]] += dJda' * (dadh⁺ + dadv⁺ * dv⁺dh⁺)
        end
    end

    return nothing
end
