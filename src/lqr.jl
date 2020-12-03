"""
    time-varying LQR
"""
function tvlqr(A, B, Q, R)
    T = length(Q)

    P = [zero(A[1]) for t = 1:T]
    K = [zero(B[1]') for t = 1:T-1]
    P[T] = Q[T]

    for t = T-1:-1:1
        K[t] = (R[t] + B[t]' * P[t+1] *  B[t]) \ (B[t]' * P[t+1] * A[t])
        P[t] = (Q[t] + K[t]' * R[t] * K[t]
                + (A[t] - B[t] * K[t])' * P[t+1] * (A[t] - B[t] * K[t]))
    end

    return K, P
end

function tvlqr(model, x, u, Q, R, h)
    A, B = jacobians(model, x, u, h)
    h == 0.0 && (R = [r[1:end-1, 1:end-1] for r in R])
    K, P = tvlqr(A, B, Q, R)
    return K, P
end

"""
    jacobians along trajectory
"""
function jacobians(model, x, u, h)
    A = []
    B = []

    w = zeros(model.d)
    free_time = (h == 0.0 ? true : false)

    T = length(x)

    for t = 1:T-1
        xt = x[t]
        ut = u[t]
        xt⁺ = x[t+1]
        free_time && (h = u[end])

        fx(z) = fd(model, xt⁺, z, ut, w, h, t)
        fu(z) = fd(model, xt⁺, xt, z, w, h, t)
        fx⁺(z) = fd(model, z, xt, ut, w, h, t)

        A⁺ = ForwardDiff.jacobian(fx⁺, xt⁺)
        push!(A, -1.0 * A⁺ \ ForwardDiff.jacobian(fx, xt))
        push!(B, -1.0 * A⁺ \ ForwardDiff.jacobian(fu,
            ut)[:, 1:end - (free_time ? 1 : 0)])
    end

    return A, B
end
