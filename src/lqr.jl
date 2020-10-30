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

function tvlqr(model, X, U, Q, R)
    A, B = jacobians(model, X, U)
    K = TVLQR(A, B, Q, R)
    return K
end

"""
    jacobians along trajectory
"""
function jacobians(model, X, U)
    A = []
    B = []

    w = zeros(model.d)

    for t = 1:T-1
        x = X[t]
        u = U[t]
        x⁺ = X[t+1]

        fx(z) = fd(model, x⁺, z, u, w, h, t)
        fu(z) = fd(model, x⁺, x, z, w, h, t)
        fx⁺(z) = fd(model, z, x, u, w, h, t)

        A⁺ = ForwardDiff.jacobian(fx⁺, x⁺)
        push!(A,-1.0 * A⁺ \ ForwardDiff.jacobian(fx, x))
        push!(B,-1.0 * A⁺ \ ForwardDiff.jacobian(fu, u))
    end

    return A, B
end
