"""
    time-varying LQR
"""
function tvlqr(A, B, Q, R)
    P = [zero(A[1]) for t = 1:T]
    K = [zero(B[1]') for t = 1:T-1]
    P[T] = Q[T]
    for t = T-1:-1:1
        K[t] = (R[t] + B[t]' * P[t+1] *  B[t]) \ (B[t]' * P[t+1] * A[t])
        P[t] = (Q[t] + K[t]' * R[t] * K[t]
                + (A[t] - B[t] * K[t])' * P[t+1] * (A[t] - B[t] * K[t]))
    end
    return K
end

function tvlqr(model, X, U, Q, R, h)
    A, B = jacobians(model, X, U, h)
    h == 0.0 && (R = [r[1:end-1, 1:end-1] for r in R])
    K = tvlqr(A, B, Q, R)
    return K
end

"""
    jacobians along trajectory
"""
function jacobians(model, X, U, h)
    A = []
    B = []

    w = zeros(model.d)
    free_time = (h == 0.0 ? true : false)


    for t = 1:T-1
        x = X[t]
        u = U[t]
        x⁺ = X[t+1]
        free_time && (h = u[end])

        fx(z) = fd(model, x⁺, z, u, w, h, t)
        fu(z) = fd(model, x⁺, x, z, w, h, t)
        fx⁺(z) = fd(model, z, x, u, w, h, t)

        A⁺ = ForwardDiff.jacobian(fx⁺, x⁺)
        push!(A, -1.0 * A⁺ \ ForwardDiff.jacobian(fx, x))
        push!(B, -1.0 * A⁺ \ ForwardDiff.jacobian(fu,
            u)[:, 1:end - (free_time ? 1 : 0)])
    end

    return A, B
end
