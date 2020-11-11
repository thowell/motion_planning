"""
    nonlinear stage-wise objective
"""
struct NonlinearStageObjective <: Objective
    l_stage
    l_terminal
end

function nonlinear_stage_objective(l_stage, l_terminal)
    return NonlinearStageObjective(l_stage, l_terminal)
end

function objective(Z, obj::NonlinearStageObjective, idx, T)
    J = 0.0

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])

        J += obj.l_stage(x, u, t)
    end

    x = view(Z, idx.x[T])
    J += obj.l_terminal(x)

    return J
end

function objective_gradient!(∇J, Z, obj::NonlinearStageObjective, idx, T)

    for t = 1:T-1
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])

        lx(y) = obj.l_stage(y, u, t)
        lu(y) = obj.l_stage(x, y, t)

        ∇J[idx.x[t]] += ForwardDiff.gradient(lx, x)
        ∇J[idx.u[t]] += ForwardDiff.gradient(lu, u)

    end

    x = view(Z, idx.x[T])
    ∇J[idx.x[T]] += ForwardDiff.gradient(obj.l_terminal, x)

    return nothing
end
