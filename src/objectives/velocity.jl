"""
    finite-difference velocity objective
"""
struct VelocityObjective <: Objective
    Q
    n
    h

    idx_angle
end

velocity_objective(Q, n; h = 0.0, idx_angle = (1:0)) = VelocityObjective(Q, n, h, idx_angle)

function angle_diff(a1, a2)
    mod(a2 - a1 + pi, 2 * pi) - pi
end

function objective(Z, obj::VelocityObjective, idx, T)
    n = obj.n

    J = 0.0

    for t = 1:T-1
        q⁻ = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
        h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
        v = (q⁺ - q⁻) ./ h
        # v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle))

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
        # v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle),
        #     view(q⁺, obj.idx_angle))

        dJdv = 2.0 * obj.Q[t] * v
        ∇J[idx.x[t][n .+ (1:n)]] += -1.0 ./ h * dJdv
        ∇J[idx.x[t + 1][n .+ (1:n)]] += 1.0 ./ h * dJdv
        if obj.h == 0.0
            ∇J[idx.u[t][end]] += -1.0 * dJdv' * v ./ h
        end
    end

    return nothing
end

# using ForwardDiff
# b1 = -pi / 2.0
# b2 = pi / 2.0 + pi
#
# angle_diff(b1, b2)
#
# tmp1(y) = angle_diff(y, b2)
# tmp2(y) = angle_diff(b1, y)
# ForwardDiff.derivative(tmp1, b1)
# ForwardDiff.derivative(tmp2, b2)
#
# angle_diff.(ones(2), pi * ones(2))
