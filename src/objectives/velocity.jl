"""
    finite-difference velocity objective
"""
struct VelocityObjective <: Objective
    Q
    n
    h

    idx_angle
end

function velocity_objective(Q, n; h = 0.0, idx_angle = (1:0))
    @warn "angle difference not implemented"
    VelocityObjective(Q, n, h, idx_angle)
end

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
        v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h

        J += v' * obj.Q[t] * v
    end

    return J
end

function objective_gradient!(∇J, Z, obj::VelocityObjective, idx, T)
    tmp(y) = objective(y, obj, idx, T)
    ∇J .+= ForwardDiff.gradient(tmp, Z)
    # n = obj.n
    #
    # for t = 1:T-1
    #     q⁻ = view(Z, idx.x[t][n .+ (1:n)])
    #     q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
    #     h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
    #     v = (q⁺ - q⁻) ./ h
    #     # v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle),
    #     #     view(q⁺, obj.idx_angle)) ./ h
    #
    #     dJdv = 2.0 * obj.Q[t] * v
    #     ∇J[idx.x[t][n .+ (1:n)]] += -1.0 ./ h * dJdv
    #     ∇J[idx.x[t + 1][n .+ (1:n)]] += 1.0 ./ h * dJdv
    #     # if obj.h == 0.0
    #     #     ∇J[idx.u[t][end]] += -1.0 * dJdv' * v ./ h
    #     # end
    # end

    return nothing
end

# function objective(Z, obj::VelocityObjective, idx, T)
#     n = obj.n
#
#     J = 0.0
#
#     for t = 1:T-1
#         q⁻ = view(Z, idx.x[t][n .+ (1:n)])
#         q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
#         h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
#         v = (q⁺ - q⁻) ./ h
#         v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h
#         J += v' * obj.Q[t] * v
#     end
#
#     return J
# end
#
# function objective_gradient!(∇J, Z, obj::VelocityObjective, idx, T)
#     # _tmp(y) = objective(y, obj, idx, T)
#     # ∇J .= ForwardDiff.gradient(_tmp, Z)
#     n = obj.n
#
#     for t = 1:T-1
#         q⁻ = view(Z, idx.x[t][n .+ (1:n)])
#         q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
#         h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
#
#         # v = (q⁺ - q⁻) ./ h
#         # # v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle),
#         # #     view(q⁺, obj.idx_angle)) ./ h
#         #
#         # dJdv = 2.0 * obj.Q[t] * v
#
#         function tmp(x⁻, x⁺, g)
#             w = (x⁺ - x⁻) ./ g
#             w[obj.idx_angle] = angle_diff.(view(x⁻, obj.idx_angle), view(x⁺, obj.idx_angle)) ./ g
#             w' * obj.Q[t] * w
#         end
#
#         tmp1(y) = tmp(y, q⁺, h)
#         tmp2(y) = tmp(q⁻, y, h)
#         tmp3(y) = tmp(q⁻, q⁺, y)
#
#         ∇J[idx.x[t][n .+ (1:n)]] += ForwardDiff.gradient(tmp1, q⁻)
#         ∇J[idx.x[t + 1][n .+ (1:n)]] += ForwardDiff.gradient(tmp2, q⁺)
#         if obj.h == 0.0
#             ∇J[idx.u[t][end]] += ForwardDiff.gradient(tmp3, h)[1]
#         end
#     end
#
#     return nothing
# end

# using ForwardDiff
# b1 = -pi / 2.0
# b2 = pi / 2.0 + pi
#
# angle_diff(b1, b2)
#
# tmp1(y) = angle_diff(y, b2)
# tmp2(y) = angle_diff(b1, y)
# for i = 1:1000
#     b1 = rand(1)[1]
#     b2 = rand(1)[1]
#     h1 = rand(1)[1]
#     tmp1(y) = angle_diff(y, b2) / h1
#     tmp2(y) = angle_diff(b1, y) / h1
#     @assert ForwardDiff.derivative(tmp1, b1) == -1.0 / h1
#     @assert ForwardDiff.derivative(tmp2, b2) == 1.0 / h1
# end
#
# angle_diff.(ones(2), pi * ones(2))
