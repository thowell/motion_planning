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

function tvlqr(model::Model{<: Integration, FixedTime}, x, u, h, Q, R)
    A, B = jacobians(model, x, u, h)
    K, P = tvlqr(A, B, Q, R)
    return K, P
end

function tvlqr(model::Model{<: Integration, FreeTime}, x, u, h, Q, R)
    A, B = jacobians(model, x, u, h)
    K, P = tvlqr(A, B, Q, [r[1:end-1, 1:end-1] for r in R])
    return K, P
end

"""
    jacobians along trajectory
"""
function _jacobians(model, x, u, h)
    A = []
    B = []

    w = zeros(model.d)

    T = length(x)

    for t = 1:T-1
        xt = x[t]
        ut = u[t]
        xt⁺ = x[t+1]

        fx(z) = fd(model, xt⁺, z, ut, w, h, t)
        fu(z) = fd(model, xt⁺, xt, z, w, h, t)
        fx⁺(z) = fd(model, z, xt, ut, w, h, t)

        A⁺ = ForwardDiff.jacobian(fx⁺, xt⁺)
        push!(A, -1.0 * A⁺ \ ForwardDiff.jacobian(fx, xt))
        push!(B, -1.0 * A⁺ \ ForwardDiff.jacobian(fu, ut))
    end

    return A, B
end

jacobians(model::Model{<: Integration, FixedTime}, x, u, h) = _jacobians(model, x, u, h)

function jacobians(model::Model{<: Integration, FreeTime}, x, u, h)
    A, B = _jacobians(model, x, u, h)
    T = length(x)
    return A, [B[t][:, 1:end-1] for t = 1:T-1]
end

"""
    projection
        use time-varying LQR to get dynamically feasible solution
"""
function lqr_projection(model::Model{<: Integration, FixedTime}, x̄, ū, h̄, Q, R)
    K, P = tvlqr(model, x̄, ū, h̄, Q, R)

    x_proj = [copy(x̄[1])]
    u_proj = []

    T = length(x̄)

    for t = 1:T-1
        push!(u_proj, ū[t] - K[t] * (x_proj[end] - x̄[t]))
        push!(x_proj, propagate_dynamics(model, x_proj[end], u_proj[end],
            zeros(model.d), h̄[1], t, tol_r = 1.0e-12, tol_d = 1.0e-12))
    end

    return x_proj, u_proj
end

function lqr_projection(model::Model{<: Integration, FreeTime}, x̄, ū, h̄, Q, R)
    K, P = tvlqr(model, x̄, ū, h̄, Q, R)

    x_proj = [copy(x̄[1])]
    u_proj = []

    T = length(x̄)

    for t = 1:T-1
        push!(u_proj, [ū[t][1:end-1] - K[t] * (x_proj[end] - x̄[t]); h̄[1]])
        push!(x_proj, propagate_dynamics(model, x_proj[end], u_proj[end],
            zeros(model.d), h̄[1], t, tol_r = 1.0e-12, tol_d = 1.0e-12))
    end

    return x_proj, u_proj
end
