"""
    penalty objective
"""
struct PenaltyObjective{T} <: Objective
    α::T
    idx::Int
end

function objective(Z, obj::PenaltyObjective, idx, T)
    J = 0.0
    for t = 1:T-1
        s = Z[idx.u[t][obj.idx]]
        J += s
    end
    return obj.α * J
end

function objective_gradient!(∇J, Z, obj::PenaltyObjective, idx, T)
    for t = 1:T-1
        ∇J[idx.u[t][obj.idx]] += obj.α
    end
    return nothing
end
