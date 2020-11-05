"""
    finite-difference acceleration objective
"""
struct AccelerationObjective <: Objective
    Q
    n
end

acceleration_objective(Q, n) = AccelerationObjective(Q, n)

function objective(Z, obj::AccelerationObjective, idx, T)
    n = obj.n

    J = 0.0

    for t = 2:T-1
        # configurations
        q⁻ = view(Z, idx.x[t - 1][n .+ (1:n)])
        q = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])

        # time steps
        h⁻ = view(Z, idx.u[t - 1])[end]
        h⁺ = view(Z, idx.u[t])[end]

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
        h⁻ = view(Z, idx.u[t - 1])[end]
        h⁺ = view(Z, idx.u[t])[end]

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
        ∇J[idx.u[t - 1][end]] += dJda' * (dadh⁻ + dadv⁻ * dv⁻dh⁻)
        ∇J[idx.u[t][end]] += dJda' * (dadh⁺ + dadv⁺ * dv⁺dh⁺)
    end

    return nothing
end
